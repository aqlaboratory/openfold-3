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
from openfold3.core.metrics.alignment import (
    Structure,
    best_ca_rmsd,
    ligand_pose_metrics,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)
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
class ProteinExpectation:
    """Which protein chains to score, and how closely they must match.

    ``rmsd_max`` bounds the superposition CA-RMSD in Ångström, keyed by :class:`Mode`.
    A mode *absent* from it is measured and logged but not asserted — that absence is
    the point of keying on ``Mode``: how close a prediction lands depends strongly on
    whether it had an MSA and a template, so no single ceiling is right across all four,
    and a ceiling is only worth asserting once it has been *measured* in that mode.

    ``ref_chains`` and ``pred_chains`` are matched as sets by :func:`best_ca_rmsd`, so
    interchangeable copies need no particular order.
    """

    ref_chains: tuple[str, ...]
    #: Predicted chains to pair with ``ref_chains``. ``None`` discovers every polymer
    #: chain in the prediction, which is what a single-chain query wants — the chain id
    #: the writer emits then carries no weight.
    pred_chains: tuple[str, ...] | None = None
    rmsd_max: Mapping[Mode, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LigandExpectation:
    """Which ligand to score, and how close its pose must be.

    ``rmsd_max`` follows the same measured-before-pinned rule as
    :class:`ProteinExpectation`, but bounds the ligand pose RMSD: measured in the frame
    of the superimposed protein, so it asks whether the ligand landed in the right pocket
    the right way round, not merely whether its internal geometry is sane.

    Chains are named rather than discovered: a prediction may carry several ligands, and
    a reference typically carries buffer components and ions too, so which pair is meant
    has to be stated.
    """

    ref_chain: str
    pred_chain: str
    rmsd_max: Mapping[Mode, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Expectation:
    """One query's experimental reference, and what about the prediction must match it.

    The protein is always scored; the ligand only when the query has one with a real
    counterpart in the reference. Both sub-expectations name chains within the same
    ``ref_cif``.

    To score a query: commit its experimental structure under ``test_data/mmcifs/``, then
    run the case in the mode you want to pin and read the measurement off the log line in
    :func:`_maybe_assert_accuracy`, leaving margin for hardware variance::

        Expectation(
            ref_cif=MMCIFS_DIR / "1ubq.cif",
            protein=ProteinExpectation(
                ref_chains=("A",),
                rmsd_max={Mode(use_msa_server=True, use_templates=True): 2.0},
            ),
        )
    """

    ref_cif: Path
    protein: ProteinExpectation
    ligand: LigandExpectation | None = None


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
                protein=ProteinExpectation(
                    ref_chains=("A",),
                    # Measured 2026-07-28 on of3-p2-155k, one diffusion sample, seed 42.
                    # Pinned just above each measurement rather than at a loose common
                    # bound: ubiquitin is easy enough that every mode lands under 1 Å, so
                    # a generous ceiling would accept a real degradation unnoticed.
                    rmsd_max={
                        # measured 0.87 Å (gdt_ts 0.974)
                        Mode(use_msa_server=False, use_templates=False): 0.9,
                        # measured 0.87 Å (gdt_ts 0.974) — same numbers as the row
                        # above: without the MSA server template search has nothing to
                        # draw on (preprocessing returns within the same second), so
                        # both no_msa modes are the same computation. Not bit-identical
                        # though — 7L39 moves between runs across this same pair — they
                        # agree here because a converged prediction is stable.
                        Mode(use_msa_server=False, use_templates=True): 0.9,
                        # measured 0.75 Å (gdt_ts 0.967)
                        Mode(use_msa_server=True, use_templates=False): 0.8,
                        # measured 0.79 Å (gdt_ts 0.967) — only 0.01 Å of headroom, the
                        # first ceiling to revisit if this goes flaky on other hardware
                        Mode(use_msa_server=True, use_templates=True): 0.8,
                    },
                ),
            )
        },
        marks=(requires_examples,),
    ),
    InferenceCase(
        id="query_single_protein_single_ligand",
        build_query_set=partial(
            _example_query_set, "query_single_protein_single_ligand.json"
        ),
        expectations={
            "pdb_7L39": Expectation(
                # T4 lysozyme L99A with toluene. The query encodes the L99A mutation
                # itself (position 99 is A), and 7l39 chain A matches the query sequence
                # residue-for-residue over all 162 modelled positions — no stray point
                # mutants, which is the hazard with T4 lysozyme.
                #
                # Two equally exact alternatives exist, both also monomeric with toluene
                # (MBN) bound: 4W53 (1.56 Å, all 164 residues modelled) and 7L3A (1.11 Å,
                # cryo, toluene the only non-water heteroatom). 7L39 is used because the
                # query names itself after it. They are not interchangeable to the
                # precision we assert at — pairwise CA-RMSD among the three runs
                # 0.24–0.46 Å — so a ceiling recorded here is tied to this reference.
                ref_cif=MMCIFS_DIR / "7l39.cif",
                protein=ProteinExpectation(
                    ref_chains=("A",),
                    # Measured 2026-07-28 on of3-p2-155k, one diffusion sample, seed 42,
                    # over two separate runs:
                    #   no_msa-no_templates  15.29 / 14.85 Å (gdt_ts 0.032 / 0.054)
                    #   no_msa-templates     15.78 / 15.91 Å (gdt_ts 0.039 / 0.039)
                    #   msa-no_templates      0.22 /  0.22 Å (gdt_ts 1.000)
                    #   msa-templates         0.20 /  0.20 Å (gdt_ts 1.000)
                    # The repeat is the evidence for how these are pinned: the converged
                    # modes reproduced to the last printed digit, while the failed ones
                    # moved by up to 0.44 Å between runs.
                    #
                    # Unlike ubiquitin, this target lives or dies on the MSA:
                    # single-sequence the model does not find the fold at all (gdt_ts
                    # 0.03), with an MSA it is essentially exact. Which is the case the
                    # per-mode encoding exists for — one ceiling across all four would
                    # have to be ~16 Å and would then wave through a total collapse of
                    # the MSA path.
                    rmsd_max={
                        # Only the converged modes are pinned, and pinned tight. gdt_ts
                        # is 1.000 in both, i.e. the prediction is locked onto the
                        # reference, so the number is reproducible.
                        Mode(use_msa_server=True, use_templates=False): 0.3,
                        Mode(use_msa_server=True, use_templates=True): 0.3,
                        # The two no_msa modes are deliberately left unpinned. Their
                        # RMSD is the distance to a fold the model never found, which is
                        # not a stable quantity — see the run-to-run spread above.
                        # Pinning a tight ceiling there would buy flakiness and no
                        # signal. Asserting these properly needs a *floor* — "must be
                        # worse than 8 Å without an MSA", the way test_templates.py
                        # proves its template effect — which ProteinExpectation cannot
                        # express yet.
                    },
                ),
                # Toluene (CCD MBN), 7 heavy atoms, sitting in the engineered L99A
                # cavity. The query supplies it as SMILES, so the prediction's atom
                # names are the writer's invention and matching goes through the bond
                # graph; the methyl leaves the ring with a 2-fold flip, which the
                # symmetry search covers. In the reference it is chain D — chain A is
                # the protein, and the other hetero chains are buffer (TRS, BME, Cl).
                ligand=LigandExpectation(
                    ref_chain="D",
                    pred_chain="X",
                    # Measured 2026-07-28, same runs as the CA-RMSD above:
                    #   no_msa-no_templates  25.76 Å (centroid 25.65)
                    #   no_msa-templates     16.45 Å (centroid 16.35)
                    #   msa-no_templates      0.30 Å (centroid  0.27)
                    #   msa-templates         0.27 Å (centroid  0.26)
                    # The pose tracks the fold exactly as expected: with an MSA the
                    # ligand sits sub-Ångström in the cavity, far inside the 2 Å that
                    # normally counts as a correct pose; without one the centroid alone
                    # is 16-26 Å off, i.e. the ligand is not in a pocket at all because
                    # there is no pocket. Only the converged modes are pinned, for the
                    # same reason as the protein.
                    rmsd_max={
                        Mode(use_msa_server=True, use_templates=False): 0.4,
                        Mode(use_msa_server=True, use_templates=True): 0.4,
                    },
                ),
            )
        },
        marks=(requires_examples,),
    ),
]

CASE_PARAMS = [pytest.param(case, id=case.id, marks=case.marks) for case in CASES]
MODE_PARAMS = [pytest.param(mode, id=mode.id) for mode in MODES]


def _assert_within_ceiling(
    measured: float, ceiling: float | None, *, what: str, query_name: str, mode: Mode
) -> None:
    """Enforce one ceiling, when this mode has one pinned.

    ``None`` means the mode is measured but makes no accuracy claim, which is the
    default until a number has actually been observed for it.
    """
    if ceiling is None:
        return
    assert measured < ceiling, (
        f"{query_name} [{mode.id}]: {what} {measured:.2f} Å exceeds the "
        f"{ceiling} Å ceiling for this mode"
    )


def _assert_protein(
    expectation: Expectation,
    query_name: str,
    mode: Mode,
    *,
    pred: Structure,
    ref: Structure,
) -> None:
    """Score the protein fold against the reference."""
    protein = expectation.protein
    metrics = best_ca_rmsd(
        pred=pred,
        ref=ref,
        ref_chains=protein.ref_chains,
        pred_chains=protein.pred_chains,
    )
    logger.info(
        "%s [%s] CA-RMSD %.2f Å (gdt_ts %.3f) vs %s",
        query_name,
        mode.id,
        metrics.rmsd,
        metrics.gdt_ts,
        expectation.ref_cif.name,
    )
    _assert_within_ceiling(
        metrics.rmsd,
        protein.rmsd_max.get(mode),
        what=f"CA-RMSD against {expectation.ref_cif.name}",
        query_name=query_name,
        mode=mode,
    )


def _assert_ligand(
    expectation: Expectation,
    query_name: str,
    mode: Mode,
    *,
    pred: Structure,
    ref: Structure,
) -> None:
    """Score the ligand pose in the frame of the superimposed protein."""
    ligand = expectation.ligand
    assert ligand is not None  # guarded by the caller
    metrics = ligand_pose_metrics(
        pred=pred,
        ref=ref,
        ref_chains=expectation.protein.ref_chains,
        pred_chains=expectation.protein.pred_chains,
        pred_ligand_chain=ligand.pred_chain,
        ref_ligand_chain=ligand.ref_chain,
    )
    logger.info(
        "%s [%s] ligand RMSD %.2f Å (centroid %.2f Å, %d atoms, %d symmetry mappings) "
        "vs %s chain %s",
        query_name,
        mode.id,
        metrics.rmsd,
        metrics.centroid_distance,
        metrics.n_atoms,
        metrics.n_symmetry_mappings,
        expectation.ref_cif.name,
        ligand.ref_chain,
    )
    _assert_within_ceiling(
        metrics.rmsd,
        ligand.rmsd_max.get(mode),
        what=(
            f"ligand RMSD against {expectation.ref_cif.name} chain {ligand.ref_chain}"
        ),
        query_name=query_name,
        mode=mode,
    )


def _maybe_assert_accuracy(
    case: InferenceCase, query_name: str, mode: Mode, *, pred_cif: Path
) -> None:
    """Score one prediction against its reference, when the case declares one.

    Always logs what it measures, whether or not a ceiling is pinned for this mode, so a
    run in an unscored mode still tells you what to record for it.
    """
    expectation = case.expectations.get(query_name)
    if expectation is None:
        return
    # Parsed once here and handed down: the protein and the ligand both need both
    # structures, and the ligand reuses the protein's superposition.
    pred = Structure.from_cif(pred_cif)
    ref = Structure.from_cif(expectation.ref_cif)
    _assert_protein(expectation, query_name, mode, pred=pred, ref=ref)
    if expectation.ligand is not None:
        _assert_ligand(expectation, query_name, mode, pred=pred, ref=ref)


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
        _maybe_assert_accuracy(
            case, query_name, mode, pred_cif=seed_dir / f"{stem}_model.cif"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv"]))
