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

"""Smoke tests for inference: small queries run end-to-end.

``test_inference_writes_outputs`` runs every case in ``CASES`` against every combination
of ``use_msa_server`` and ``use_templates``, checking the expected output files are
written for each query the case contains. Cases mix in-memory query sets with the query
JSONs under ``examples/example_inference_inputs``, so adding either kind is one row —
and costs four inference runs. See ``test_templates.py`` for the functional check that a
supplied template actually steers the prediction.

These require an accelerator (CUDA, ROCm or MPS) and downloaded model weights; they skip
otherwise.

Run with:
    pytest openfold3/tests/inference/test_inference_full.py

To calibrate a case's ``ca_rmsd_max`` — run just that case in all four modes and read
the measured CA-RMSDs off the log (``log_cli_level`` is WARNING by default, so INFO has
to be asked for):

    pytest openfold3/tests/inference/test_inference_full.py -k ubiquitin \\
        -v --log-cli-level=INFO
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import pytest

import openfold3
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
from openfold3.tests.inference.analysis import best_ca_rmsd
from openfold3.tests.inference.helpers import (
    MMCIFS_DIR,
    MODES,
    Mode,
    prediction_dir,
    prediction_stem,
    query_set_from_chains,
    run_inference,
)
from openfold3.tests.utils.compare_utils import skip_unless_accelerator_available

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

#: Repo-root ``examples/`` — present in a source checkout, but not shipped in the wheel
#: (``packages.find`` only picks up ``openfold3*``), so file-backed cases must tolerate
#: its absence when the suite runs from an install.
EXAMPLES_DIR = (
    Path(openfold3.__file__).parent.parent / "examples" / "example_inference_inputs"
)

requires_examples = pytest.mark.skipif(
    not EXAMPLES_DIR.is_dir(), reason=f"No examples directory at {EXAMPLES_DIR}"
)

# Leucine zipper protein fragment
PROTEIN_CHAIN = {
    "molecule_type": "protein",
    "chain_ids": ["A", "B"],
    "sequence": "XRMKQLEDKVEELLSKNYHLENEVARLKKLVGER",
}

# Phenol
LIGAND_CHAIN = {
    "molecule_type": "ligand",
    "chain_ids": ["C"],
    "smiles": "c1ccccc1O",
}


def _example_query_set(filename: str) -> InferenceQuerySet:
    """Load an example query JSON through the same loader ``run_openfold`` uses."""
    return InferenceQuerySet.from_json(EXAMPLES_DIR / filename)


@dataclass(frozen=True)
class Expectation:
    """How closely one query's prediction must match an experimental structure.

    ``ca_rmsd_max`` maps a :class:`Mode` to an Ångström ceiling on the protein CA-RMSD.
    A mode *absent* from it is run but not scored — the outputs are still checked, the
    test just makes no accuracy claim there. That absence is the point of keying on
    ``Mode``: how close a prediction lands depends strongly on whether it had an MSA and
    a template, so no single ceiling per target is right across all four, and a ceiling
    is only worth asserting once it has been *measured* in that mode.

    ``pred_chains``/``ref_chains`` are matched as sets by :func:`best_ca_rmsd`, so
    interchangeable copies need no particular order.

    To score a query: commit its experimental structure under ``test_data/mmcifs/``,
    then run the case in the mode you want to pin and read the measured CA-RMSD off the
    log line in :func:`_maybe_assert_ca_rmsd`, leaving margin for hardware variance::

        Expectation(
            ref_cif=MMCIFS_DIR / "1ubq.cif",
            ref_chains=("A",),
            ca_rmsd_max={Mode(use_msa_server=True, use_templates=True): 2.0},
        )
    """

    ref_cif: Path
    ref_chains: tuple[str, ...]
    #: Predicted chains to pair with ``ref_chains``. ``None`` discovers every polymer
    #: chain in the prediction, which is what a single-chain query wants — the chain id
    #: the writer emits then carries no weight.
    pred_chains: tuple[str, ...] | None = None
    ca_rmsd_max: Mapping[Mode, float] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceCase:
    """A query set to run, plus optional per-query accuracy expectations.

    ``build_query_set`` is deferred behind a callable so a missing examples directory
    skips the case instead of breaking collection.
    """

    id: str
    build_query_set: Callable[[], InferenceQuerySet]
    #: Query name -> expectation. Queries absent are run but not scored — the default,
    #: since scoring one needs a reference structure committed under
    #: ``test_data/mmcifs/`` and a ceiling measured per mode.
    expectations: Mapping[str, Expectation] = field(default_factory=dict)
    marks: tuple = ()


#: Each case builds an :class:`InferenceQuerySet`, either in memory or from a file in
#: ``examples/example_inference_inputs``. Expected outputs are derived from the query
#: names, so a case may hold any number of queries — adding another example is one row,
#: and costs one inference run per mode.
CASES = [
    InferenceCase(
        id="protein_only",
        build_query_set=partial(query_set_from_chains, "query1", PROTEIN_CHAIN),
    ),
    InferenceCase(
        id="protein_and_ligand",
        build_query_set=partial(
            query_set_from_chains, "query1", PROTEIN_CHAIN, LIGAND_CHAIN
        ),
    ),
    InferenceCase(
        id="ubiquitin",
        build_query_set=partial(_example_query_set, "query_ubiquitin.json"),
        expectations={
            # 1ubq chain A is 76 contiguous residues whose sequence is byte-identical
            # to the query, so every residue takes part in the comparison.
            "ubiquitin": Expectation(
                ref_cif=MMCIFS_DIR / "1ubq.cif",
                ref_chains=("A",),
                # Deliberately unpinned: run the case (see the module docstring) and
                # record the logged CA-RMSD per mode here, with margin. Until then the
                # RMSD is measured and logged but nothing is asserted.
                ca_rmsd_max={},
            )
        },
        marks=(requires_examples,),
    ),
    InferenceCase(
        id="protein_ligand_multiple",
        build_query_set=partial(
            _example_query_set, "query_protein_ligand_multiple.json"
        ),
        marks=(requires_examples,),
    ),
]

CASE_PARAMS = [pytest.param(case, id=case.id, marks=case.marks) for case in CASES]
MODE_PARAMS = [pytest.param(mode, id=mode.id) for mode in MODES]


def _maybe_assert_ca_rmsd(
    case: InferenceCase, query_name: str, mode: Mode, *, pred_cif: Path
) -> None:
    """Score one prediction against its reference, when the case declares one.

    Always logs the measured CA-RMSD when a reference exists, so a run in an unscored
    mode still tells you what ceiling to record for it.
    """
    expectation = case.expectations.get(query_name)
    if expectation is None:
        return

    metrics = best_ca_rmsd(
        pred_cif=pred_cif,
        ref_cif=expectation.ref_cif,
        ref_chains=expectation.ref_chains,
        pred_chains=expectation.pred_chains,
    )
    logger.info(
        "%s [%s] CA-RMSD %.2f Å (gdt_ts %.3f) vs %s",
        query_name,
        mode.id,
        metrics["rmsd"],
        metrics["gdt_ts"],
        expectation.ref_cif.name,
    )

    ceiling = expectation.ca_rmsd_max.get(mode)
    if ceiling is None:
        return
    assert metrics["rmsd"] < ceiling, (
        f"{query_name} [{mode.id}]: CA-RMSD {metrics['rmsd']:.2f} Å against "
        f"{expectation.ref_cif.name} exceeds the {ceiling} Å ceiling for this mode"
    )


@skip_unless_accelerator_available()
@pytest.mark.parametrize("case", CASE_PARAMS)
@pytest.mark.parametrize("mode", MODE_PARAMS)
def test_inference_writes_outputs(case, mode, tmp_path):
    """Every query in the set writes the expected per-sample files, in every mode.

    Then, where the case declares an :class:`Expectation` for that query *and* that
    mode, the prediction is scored against the experimental structure. Without that the
    suite only shows inference ran, not that it produced something consistent with
    experiment.
    """
    query_set = case.build_query_set()
    run_inference(
        query_set,
        tmp_path,
        use_msa_server=mode.use_msa_server,
        use_templates=mode.use_templates,
        num_diffusion_samples=1,
        # Isolate the template cache: without this it lands in a persistent /tmp dir
        # shared across runs, and the template-enabled cases here would see each
        # other's leftovers.
        template_output_dir=tmp_path / "template_data",
    )
    logger.info("Checking output contents at %s", tmp_path)
    for query_name in query_set.queries:
        seed_dir = prediction_dir(tmp_path, query_name)
        stem = prediction_stem(query_name, sample=1)
        expected_files = [
            f"{stem}_confidences.json",
            f"{stem}_confidences_aggregated.json",
            f"{stem}_model.cif",
            "timing.json",
        ]
        for name in expected_files:
            assert (seed_dir / name).exists(), (
                f"Expected output file not found: {seed_dir / name}"
            )
        _maybe_assert_ca_rmsd(
            case, query_name, mode, pred_cif=seed_dir / f"{stem}_model.cif"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv"]))
