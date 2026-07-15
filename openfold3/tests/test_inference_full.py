# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration tests for inference.

- ``test_protein_only`` / ``test_protein_and_ligand``: two small queries run end-to-end
  (with MSA server + templates), checking the expected output files are written.
- ``test_template_lowers_rmsd``: functional check for PR #306 — with no MSA, supplying a
  template must pull the prediction onto the native fold (low CA-RMSD to the reference),
  whereas without a template the single-sequence model can't find it (high CA-RMSD).
  Parametrized over ``CASES`` so adding a PDB is one row + committing its cif.

All of these require a GPU and downloaded model weights; they skip otherwise.

Run with:
    pytest openfold3/tests/test_inference_full.py
"""

import logging
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import biotite.structure as struc
import numpy as np
import pytest

from openfold3.core.config import config_utils
from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
from openfold3.entry_points.validator import (
    InferenceExperimentConfig,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from openfold3.tests.utils.compare_utils import skip_unless_cuda_available

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

MMCIFS_DIR = Path(__file__).parent / "test_data" / "mmcifs"

protein_only_query = InferenceQuerySet.model_validate(
    {
        "queries": {
            "query1": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A", "B"],
                        "sequence": "XRMKQLEDKVEELLSKNYHLENEVARLKKLVGER",
                    }
                ]
            }
        }
    }
)

protein_and_ligand_query = InferenceQuerySet.model_validate(
    {
        "queries": {
            "query1": {
                "chains": [
                    {
                        "molecule_type": "protein",
                        "chain_ids": ["A", "B"],
                        "sequence": "XRMKQLEDKVEELLSKNYHLENEVARLKKLVGER",
                    },
                    {
                        "molecule_type": "ligand",
                        "chain_ids": ["C"],
                        "smiles": "c1ccccc1O",
                    },
                ]
            }
        }
    }
)

inference_test_yaml_str = textwrap.dedent("""\
    model_update:
      presets:
        - predict
        - low_mem
    """)


def _run_inference_helper(
    query_set,
    output_dir: Path,
    *,
    use_msa_server: bool,
    use_templates: bool,
    num_diffusion_samples: int = 1,
    template_output_dir: Path | None = None,
) -> Path:
    """Run one inference job into ``output_dir`` and return it.

    Skips (``pytest.skip``) if no model checkpoint is available (escalated to a hard
    failure when ``OPENFOLD_SETUP_SCRIPT=1``). ``template_output_dir`` isolates the
    template cache per run (otherwise it lands in a persistent ``/tmp`` dir shared across
    runs and same-sequence queries).
    """
    runner_yaml = output_dir / "runner_config.yaml"
    yaml_str = inference_test_yaml_str
    if template_output_dir is not None:
        yaml_str += textwrap.dedent(f"""\
            template_preprocessor_settings:
              output_directory: {template_output_dir}
            """)
    runner_yaml.write_text(yaml_str)

    with patch("builtins.input", return_value="no"):
        experiment_config = InferenceExperimentConfig(
            **config_utils.load_yaml(runner_yaml)
        )
    runner = InferenceExperimentRunner(
        experiment_config,
        num_diffusion_samples=num_diffusion_samples,
        output_dir=output_dir,
        use_msa_server=use_msa_server,
        use_templates=use_templates,
    )
    try:
        runner.setup()
    except ValueError as e:
        if "is not a valid file or directory" in str(e):
            if os.environ.get("OPENFOLD_SETUP_SCRIPT") == "1":
                raise AssertionError(
                    "No checkpoint files found after running setup script. "
                    "Please check that the download completed successfully."
                ) from None
            logger.warning(
                "No checkpoint files found, skipping. Use the setup script to "
                "download the weights."
            )
            pytest.skip("No checkpoint files available")
        raise

    runner.run(query_set)
    runner.cleanup()

    err_log_dir = output_dir / "logs"
    if err_log_dir.exists():
        raise RuntimeError(
            f"Found error logs in directory {err_log_dir}, "
            "check for errors in inference."
        )
    return output_dir


def _assert_inference_writes_outputs(query_set, tmp_path):
    _run_inference_helper(
        query_set,
        tmp_path,
        use_msa_server=True,
        use_templates=True,
        num_diffusion_samples=1,
    )
    logger.info("Checking output contents at %s", tmp_path)
    seed_dir = tmp_path / "query1" / "seed_42"
    expected_files = [
        "query1_seed_42_sample_1_confidences.json",
        "query1_seed_42_sample_1_confidences_aggregated.json",
        "query1_seed_42_sample_1_model.cif",
        "timing.json",
    ]
    for name in expected_files:
        assert (seed_dir / name).exists(), (
            f"Expected output file not found: {seed_dir / name}"
        )


@skip_unless_cuda_available()
def test_protein_only(tmp_path):
    _assert_inference_writes_outputs(protein_only_query, tmp_path)


@skip_unless_cuda_available()
def test_protein_and_ligand(tmp_path):
    _assert_inference_writes_outputs(protein_and_ligand_query, tmp_path)


# --- Template-effect RMSD test (PR #306) -----------------------------------------------

# Number of diffusion samples per condition. The user's experiments show the samples
# cluster (all near the reference with a template, all far without), so the mean over
# samples is representative and robust.
NUM_DIFFUSION_SAMPLES = 5


@dataclass(frozen=True)
class TemplateRmsdCase:
    """A single-chain protein whose native structure is a committed reference cif.

    ``<pdb_id>.cif`` under ``test_data/mmcifs/`` doubles as the template CIF input and the
    RMSD reference; ``chain`` is compared against (and given as ``template_cif_chain_ids``).
    ``sequence`` is that chain's SEQRES. The three thresholds are per-case bounds on the
    CA-RMSD to the reference, all in Ångström (Å).
    """

    pdb_id: str
    chain: str
    sequence: str
    no_template_rmsd_min_angstrom: (
        float  # CA-RMSD without a template must exceed this (Å)
    )
    with_template_rmsd_max_angstrom: (
        float  # CA-RMSD with a template must be below this (Å)
    )
    rmsd_separation_min_angstrom: float  # required (off - on) CA-RMSD gap (Å)


CASES = [
    # Observed on of3-p2-155k (5 samples each, tightly clustered): off mean ≈ 16.4 Å
    # (15.5-17.5), on mean ≈ 0.26 Å (0.23-0.30). Thresholds keep a wide margin so they
    # tolerate precision/hardware variance but still fail if templates are ignored
    # (then on ≈ off ≈ 16 Å).
    TemplateRmsdCase(
        pdb_id="1a8q",
        chain="A",
        sequence=(
            "PICTTRDGVEIFYKDWGQGRPVVFIHGWPLNGDAWQDQLKAVVDAGYRGIAHDRRGHGHSTPVWDGYDFDT"
            "FADDLNDLLTDLDLRDVTLVAHSMGGGELARYVGRHGTGRLRSAVLLSAIPPVMIKSDKNPDGVPDEVFDA"
            "LKNGVLTERSQFWKDTAEGFFSANRPGNKVTQGNKDAFWYMAMAQTIEGGVRCVDAFGYTDFTEDLKKFDI"
            "PTLVVHGDDDQVVPIDATGRKSAQIIPNAELKVYEGSSHGIAMVPGDKEKFNRDLLEFLNK"
        ),
        no_template_rmsd_min_angstrom=8.0,
        with_template_rmsd_max_angstrom=2.0,
        rmsd_separation_min_angstrom=5.0,
    ),
]


def _ref_cif(case: TemplateRmsdCase) -> Path:
    return MMCIFS_DIR / f"{case.pdb_id}.cif"


def _ca(atom_array, chain: str | None = None):
    """Sorted-by-res_id CA atoms of a chain (or all non-hetero CA if chain is None)."""
    mask = (atom_array.atom_name == "CA") & (~atom_array.hetero)
    if chain is not None:
        mask &= atom_array.chain_id == chain
    ca = atom_array[mask]
    return ca[np.argsort(ca.res_id)]


def _ca_rmsd(pred_cif: Path, ref_cif: Path, ref_chain: str) -> float:
    """Superposition CA-RMSD (Angstroms) of a predicted monomer vs a reference chain."""
    pred = _ca(parse_mmcif(pred_cif).atom_array)  # monomer -> single chain
    ref = _ca(parse_mmcif(ref_cif).atom_array, ref_chain)
    common = np.intersect1d(pred.res_id, ref.res_id)
    pred = pred[np.isin(pred.res_id, common)]
    ref = ref[np.isin(ref.res_id, common)]
    assert len(pred) == len(ref) == len(common), (
        f"CA correspondence mismatch: pred={len(pred)} ref={len(ref)} "
        f"common={len(common)} (duplicate res_ids / unexpected chains?)"
    )
    fitted, _ = struc.superimpose(fixed=ref, mobile=pred)
    return float(struc.rmsd(ref, fitted))


def _make_query(case: TemplateRmsdCase, *, with_template: bool) -> tuple[object, str]:
    chain = {
        "molecule_type": "protein",
        "chain_ids": [case.chain],
        "sequence": case.sequence,
    }
    if with_template:
        chain["template_cif_paths"] = [str(_ref_cif(case))]
        chain["template_cif_chain_ids"] = [case.chain]
    key = f"{case.pdb_id}_template_{'on' if with_template else 'off'}"
    query_set = InferenceQuerySet.model_validate(
        {"queries": {key: {"chains": [chain]}}}
    )
    return query_set, key


def _mean_ca_rmsd(
    case: TemplateRmsdCase, *, with_template: bool, tmp_path: Path
) -> float:
    """Run one condition (no MSA, template on/off) and return mean CA-RMSD over samples."""
    query_set, key = _make_query(case, with_template=with_template)
    out_dir = tmp_path / key
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_inference_helper(
        query_set,
        out_dir,
        use_msa_server=False,
        use_templates=with_template,
        num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
        template_output_dir=out_dir / "template_data",
    )

    seed_dir = out_dir / key / "seed_42"
    sample_cifs = sorted(seed_dir.glob(f"{key}_seed_42_sample_*_model.cif"))
    assert sample_cifs, f"No predicted structures found in {seed_dir}"
    rmsds = [_ca_rmsd(cif, _ref_cif(case), case.chain) for cif in sample_cifs]
    logger.info("%s template=%s per-sample RMSDs: %s", key, with_template, rmsds)
    return float(np.mean(rmsds))


@skip_unless_cuda_available()
@pytest.mark.inference_verification
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.pdb_id)
def test_template_lowers_rmsd(case, tmp_path):
    """Without MSA, a supplied template must lower CA-RMSD to the native fold (PR #306)."""
    rmsd_off = _mean_ca_rmsd(case, with_template=False, tmp_path=tmp_path)
    rmsd_on = _mean_ca_rmsd(case, with_template=True, tmp_path=tmp_path)
    logger.info("%s mean RMSD off=%.2f on=%.2f", case.pdb_id, rmsd_off, rmsd_on)

    assert rmsd_off > case.no_template_rmsd_min_angstrom, (
        f"{case.pdb_id}: expected no-template RMSD > {case.no_template_rmsd_min_angstrom}, "
        f"got {rmsd_off:.2f}"
    )
    assert rmsd_on < case.with_template_rmsd_max_angstrom, (
        f"{case.pdb_id}: expected with-template RMSD < {case.with_template_rmsd_max_angstrom}, "
        f"got {rmsd_on:.2f}"
    )
    assert rmsd_off - rmsd_on > case.rmsd_separation_min_angstrom, (
        f"{case.pdb_id}: template effect too small — off={rmsd_off:.2f} "
        f"on={rmsd_on:.2f} (need gap > {case.rmsd_separation_min_angstrom})"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv"]))
