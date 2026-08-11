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

"""Pocket-constraint ligand-localization test (PR #324).

1PZP is one of the eight allosteric-ligand benchmark cases from the PR description
(Nittinger et al. 2025): unconstrained OF3 places FTA in the orthosteric site, ~21 Å from
the allosteric site the crystal structure (chain C, res 301) actually shows it in. The
query's ``pocket_constraint`` names that allosteric site's residues.

``test_pocket_constraint_localizes_ligand`` reproduces the PR's own methodology — best
ligand centre-of-mass (COM) distance to the reference over N diffusion samples, in the
frame of the superimposed protein — rather than comparing predictions to each other, so
the numbers here are directly comparable to the PR's reported table. It runs the query
twice, once with pocket-guided proposal sampling enabled and once disabled via
``dataset_config_kwargs.pocket_sampling.enabled`` (see ``PocketSamplingSettings``), and
checks that only the enabled run reliably lands the ligand in the requested pocket.

COM distance evaluates localization within the requested pocket; it does not measure
native ligand pose accuracy (see the PR description) — that is a separate, harder claim
this test does not make.

Requires an accelerator (CUDA, ROCm or MPS) and downloaded model weights; skips
otherwise.

Run with:
    pytest openfold3/tests/inference/test_pocket_constraints.py
"""

import logging
import textwrap
from pathlib import Path

import pytest

import openfold3
from openfold3.core.metrics.alignment import Structure, ligand_pose_metrics
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from openfold3.tests.inference.helpers import (
    MMCIFS_DIR,
    SampleScores,
    measure_samples,
    predicted_structure_cifs,
    run_inference,
)
from openfold3.tests.utils.compare_utils import skip_unless_accelerator_available

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.slow]

#: Repo-root ``examples/`` — present in a source checkout, but not shipped in the wheel
#: (``packages.find`` only picks up ``openfold3*``), so this skips rather than errors
#: when the suite runs from an install.
EXAMPLES_DIR = (
    Path(openfold3.__file__).parent.parent / "examples" / "example_inference_inputs"
)
POCKET_QUERY_JSON = EXAMPLES_DIR / "query_protein_ligand_pocket_constraint.json"

requires_examples = pytest.mark.skipif(
    not EXAMPLES_DIR.is_dir(), reason=f"No examples directory at {EXAMPLES_DIR}"
)

#: Query name inside the example JSON, and the chain ids within it.
QUERY_NAME = "bla_1PZP_FTA"
PROTEIN_CHAIN_ID = "A"
LIGAND_CHAIN_ID = "L"

#: Beta-lactamase (chain A, residues 1-263) with two bound copies of FTA: chain B
#: (res 300) is a distal/orthosteric copy ~16 A from the query's pocket residues, chain C
#: (res 301) is the allosteric copy those residues actually surround (~5.8 A to the
#: pocket-residue centroid) and is what ``pocket_constraint`` targets. Chain A's sequence
#: matches the query byte-for-byte, so residue numbering lines up directly with
#: ``pocket_residues``.
REF_CIF = MMCIFS_DIR / "1pzp.cif"
REF_LIGAND_CHAIN_ID = "C"

#: Full search — the realistic case a pocket constraint is meant for, docking a ligand
#: into a properly folded structure rather than an unconverged single-sequence one.
USE_MSA_SERVER = True
USE_TEMPLATES = True

# PR #324's own best-of-16 numbers for this case (0.63 A / 20.98 A) were measured with
# 16 diffusion samples, we simplify this test to use 5 diffusion samples but increase
# expected min COM from 2 -> 5
NUM_DIFFUSION_SAMPLES = 5
POCKET_SAMPLING_ON_COM_MAX_ANGSTROM = 5.0
POCKET_SAMPLING_OFF_COM_MIN_ANGSTROM = 10.0


def _run(*, extra_yaml: str, out_dir: Path) -> list[Path]:
    """Run one condition and return its predicted sample cifs, in sample order."""
    query_set = InferenceQuerySet.from_json(POCKET_QUERY_JSON)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_inference(
        query_set,
        out_dir,
        use_msa_server=USE_MSA_SERVER,
        use_templates=USE_TEMPLATES,
        num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
        extra_yaml=extra_yaml,
    )
    return predicted_structure_cifs(out_dir, QUERY_NAME)


def _best_of_n_com(sample_cifs: list[Path], *, ref: Structure) -> SampleScores:
    """COM distance of every sample to the allosteric reference ligand, superimposed."""
    metrics = measure_samples(
        sample_cifs,
        lambda pred: ligand_pose_metrics(
            pred=pred,
            ref=ref,
            ref_chains=(PROTEIN_CHAIN_ID,),
            pred_chains=(PROTEIN_CHAIN_ID,),
            pred_ligand_chain=LIGAND_CHAIN_ID,
            ref_ligand_chain=REF_LIGAND_CHAIN_ID,
        ),
        expected_samples=NUM_DIFFUSION_SAMPLES,
    )
    return SampleScores.of(metrics, lambda m: m.centroid_distance)


@requires_examples
@skip_unless_accelerator_available()
@pytest.mark.inference_verification
def test_pocket_constraint_localizes_ligand(tmp_path):
    """Pocket-guided sampling must reliably land the ligand in the requested site.

    Best-of-N ligand COM distance to the allosteric reference, over ``pocket_sampling``
    on and off — the same metric and methodology the PR itself benchmarks with.
    """
    ref = Structure.from_cif(REF_CIF)

    def _make_pocket_sampling_yaml(enabled: bool) -> str:
        return textwrap.dedent(
            f"""\
            dataset_config_kwargs:
              pocket_sampling:
                enabled: {enabled}
            """
        )

    on_cifs = _run(
        extra_yaml=_make_pocket_sampling_yaml(enabled=True), out_dir=(tmp_path / "on")
    )
    off_cifs = _run(
        extra_yaml=_make_pocket_sampling_yaml(enabled=False),
        out_dir=(tmp_path / "off"),
    )

    on_com = _best_of_n_com(on_cifs, ref=ref)
    off_com = _best_of_n_com(off_cifs, ref=ref)
    logger.info(
        "%s COM distance to %s chain %s | pocket_sampling on %s | off %s",
        QUERY_NAME,
        REF_CIF.name,
        REF_LIGAND_CHAIN_ID,
        on_com,
        off_com,
    )

    assert on_com.best < POCKET_SAMPLING_ON_COM_MAX_ANGSTROM, (
        f"{QUERY_NAME}: pocket-guided sampling never localized the ligand — best-of-"
        f"{NUM_DIFFUSION_SAMPLES} COM distance {on_com.best:.2f} Å exceeds the "
        f"{POCKET_SAMPLING_ON_COM_MAX_ANGSTROM} Å ceiling"
    )
    assert off_com.best > POCKET_SAMPLING_OFF_COM_MIN_ANGSTROM, (
        f"{QUERY_NAME}: unguided sampling unexpectedly localized the ligand — best-of-"
        f"{NUM_DIFFUSION_SAMPLES} COM distance {off_com.best:.2f} Å is below the "
        f"{POCKET_SAMPLING_OFF_COM_MIN_ANGSTROM} Å floor, i.e. pocket guidance may "
        "no longer be doing anything"
    )
