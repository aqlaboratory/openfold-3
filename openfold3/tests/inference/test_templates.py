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

"""Template-effect RMSD test (PR #306).

``test_template_lowers_rmsd``: with no MSA, supplying a template must pull the prediction
onto the native fold (low CA-RMSD to the reference), whereas without a template the
single-sequence model can't find it (high CA-RMSD). Parametrized over ``CASES`` so adding
a PDB is one row + committing its cif.

Requires an accelerator (CUDA, ROCm or MPS) and downloaded model weights; skips
otherwise.

Run with:
    pytest openfold3/tests/inference/test_templates.py
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from openfold3.core.metrics.alignment import Structure, best_ca_rmsd
from openfold3.tests.inference.helpers import (
    MMCIFS_DIR,
    Mode,
    predicted_structure_cifs,
    query_set_from_chains,
    run_inference,
)
from openfold3.tests.utils.compare_utils import skip_unless_accelerator_available

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Number of diffusion samples per condition. The user's experiments show the samples
# cluster (all near the reference with a template, all far without), so the mean over
# samples is representative and robust.
NUM_DIFFUSION_SAMPLES = 5

#: The two conditions compared. Both run single-sequence, so the template flag is the
#: only difference between them — that is what makes the RMSD gap attributable to it.
TEMPLATE_OFF = Mode(use_msa_server=False, use_templates=False)
TEMPLATE_ON = Mode(use_msa_server=False, use_templates=True)


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


def _make_query(case: TemplateRmsdCase, *, mode: Mode) -> tuple[object, str]:
    """Build the query for one condition, attaching the template only when it is on."""
    chain = {
        "molecule_type": "protein",
        "chain_ids": [case.chain],
        "sequence": case.sequence,
    }
    if mode.use_templates:
        chain["template_cif_paths"] = [str(_ref_cif(case))]
        chain["template_cif_chain_ids"] = [case.chain]
    key = f"{case.pdb_id}_template_{'on' if mode.use_templates else 'off'}"
    return query_set_from_chains(key, chain), key


def _mean_ca_rmsd(case: TemplateRmsdCase, *, mode: Mode, tmp_path: Path) -> float:
    """Run one condition and return the mean CA-RMSD over diffusion samples."""
    query_set, key = _make_query(case, mode=mode)
    out_dir = tmp_path / key
    out_dir.mkdir(parents=True, exist_ok=True)
    run_inference(
        query_set,
        out_dir,
        use_msa_server=mode.use_msa_server,
        use_templates=mode.use_templates,
        num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
        template_output_dir=out_dir / "template_data",
    )

    sample_cifs = predicted_structure_cifs(out_dir, key)
    assert sample_cifs, f"No predicted structures found under {out_dir / key}"
    # The reference is parsed once for the whole batch of samples; the prediction is a
    # monomer, so let its chains be discovered — the chain id the writer emits carries
    # no information here.
    reference = Structure.from_cif(_ref_cif(case))
    rmsds = [
        best_ca_rmsd(Structure.from_cif(cif), reference, ref_chains=(case.chain,)).rmsd
        for cif in sample_cifs
    ]
    logger.info("%s [%s] per-sample RMSDs: %s", key, mode.id, rmsds)
    return float(np.mean(rmsds))


@skip_unless_accelerator_available()
@pytest.mark.inference_verification
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.pdb_id)
def test_template_lowers_rmsd(case, tmp_path):
    """Without MSA, a supplied template must lower CA-RMSD to the native fold (PR #306)."""
    rmsd_off = _mean_ca_rmsd(case, mode=TEMPLATE_OFF, tmp_path=tmp_path)
    rmsd_on = _mean_ca_rmsd(case, mode=TEMPLATE_ON, tmp_path=tmp_path)
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
