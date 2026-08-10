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

Accuracy is asserted on the **mean over** ``SCORED_DIFFUSION_SAMPLES`` **diffusion
samples**, never on a single one — see that constant for why a single sample's RMSD is
not a portable quantity. Thresholds follow from the measured distribution rather than
from one observation: see :class:`ProteinExpectation` for the rule.

To calibrate a case — run it in all four modes and read the per-sample values, mean and
sd off the log (``log_cli_level`` is WARNING by default, so INFO has to be asked for),
then apply the rule in :class:`ProteinExpectation`:

    pytest openfold3/tests/inference/test_inference_full.py -k ubiquitin \\
        -v --log-cli-level=INFO

Every measurement is logged before it is asserted, so a run that trips a threshold still
prints everything needed to recalibrate it. Drop ``-x`` when calibrating: fail-fast stops
at the first tripped mode and leaves the others unmeasured.
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
    SCORED_DIFFUSION_SAMPLES,
    Mode,
    SampleScores,
    measure_samples,
    predicted_structure_cifs,
    prediction_dir,
    prediction_stem,
    query_set_from_chains,
    run_inference,
)
from openfold3.tests.utils.compare_utils import skip_unless_accelerator_available

pytestmark = [pytest.mark.slow]


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

    Both bounds apply to the **mean CA-RMSD over diffusion samples**, in Ångström, keyed
    by :class:`Mode`. A mode absent from a mapping is measured and logged but not
    asserted by it — how close a prediction lands depends strongly on whether it had an
    MSA and a template, so no single bound is right across all four, and a bound is only
    worth asserting once it has been *measured* in that mode.

    ``rmsd_max`` is a ceiling: the prediction must be at least this good.
    ``rmsd_min`` is a floor — the prediction must be at least this *bad*. That is not a
    perverse assertion but the positive control for the MSA path: some targets are not
    solvable single-sequence, and pinning "without an MSA this must stay ~15 Å out"
    catches an MSA pipeline that silently stops contributing, which no ceiling can.
    ``test_templates.py`` makes the same argument for templates.

    Calibrating a bound
    -------------------
    Run the mode, read the logged mean and sd, and take::

        ceiling = round_up_0.1(max(mean + 4 * sd / sqrt(n), 1.5 * mean))

    The ``4 * SE`` term covers the noise of the mean itself. The ``1.5 x`` term covers
    what more samples cannot: run-to-run variation, live MSA-server payload differences,
    and the fact that backends do not run identical kernels — the ``mps`` preset
    disables the fused triton triangle kernels that CUDA uses, so the two compute
    genuinely different fp32 results before any noise is drawn.

    A bound is only valid for the backend it was measured on until a *wider*
    distribution turns up elsewhere. Ceilings are one-sided, so calibrating on the widest
    backend seen so far is automatically safe for every more accurate one; the values
    here were measured on MPS, which is currently the widest. Do not tighten a ceiling to
    a narrower backend's numbers — that is exactly how these came to be CUDA-only.

    ``ref_chains`` and ``pred_chains`` are matched as sets by :func:`best_ca_rmsd`, so
    interchangeable copies need no particular order.
    """

    ref_chains: tuple[str, ...]
    #: Predicted chains to pair with ``ref_chains``. ``None`` discovers every polymer
    #: chain in the prediction, which is what a single-chain query wants — the chain id
    #: the writer emits then carries no weight.
    pred_chains: tuple[str, ...] | None = None
    rmsd_max: Mapping[Mode, float] = field(default_factory=dict)
    rmsd_min: Mapping[Mode, float] = field(default_factory=dict)


@dataclass(frozen=True)
class LigandExpectation:
    """Which ligand to score, and how close its pose must be.

    ``rmsd_max`` follows the same mean-over-samples convention and the same calibration
    rule as :class:`ProteinExpectation`, but bounds the ligand pose RMSD: measured in the
    frame of the superimposed protein, so it asks whether the ligand landed in the right
    pocket the right way round, not merely whether its internal geometry is sane.

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
    run the case in the mode you want to pin, read the logged mean and sd, and apply the
    calibration rule in :class:`ProteinExpectation`::

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
    #: ``test_data/mmcifs/`` and a bound measured per mode.
    expectations: Mapping[str, Expectation] = field(default_factory=dict)
    marks: tuple = ()

    @property
    def num_diffusion_samples(self) -> int:
        """Samples to draw: enough to average over when scored, otherwise one.

        A case with no expectations only proves inference ran and wrote its files, and
        one sample proves that as well as eight — so the unscored cases stay cheap.
        """
        return SCORED_DIFFUSION_SAMPLES if self.expectations else 1


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
                    # Measured 2026-08-04 on of3-p2-155k, MPS, 8 samples, seed 42, as
                    # mean ± sd. Ubiquitin is a *wide* fixture, not an inaccurate one:
                    # single-sequence it scores avg_plddt 78.6 / ptm 0.67, so the model
                    # is genuinely unsure and the samples scatter accordingly (gdt_ts
                    # 0.888-0.974). Averaging over 8 is what makes it assertable at all.
                    rmsd_max={
                        # 1.197 ± 0.273
                        Mode(use_msa_server=False, use_templates=False): 1.8,
                        # 1.197 ± 0.273 — identical to the row above, to every decimal
                        # place: without the MSA server template search has nothing to
                        # draw on, so both no_msa modes are literally the same
                        # computation. (Verified on MPS, where inference is run-to-run
                        # deterministic at a fixed seed.)
                        Mode(use_msa_server=False, use_templates=True): 1.8,
                        # 1.224 ± 0.456
                        Mode(use_msa_server=True, use_templates=False): 1.9,
                        # 1.193 ± 0.447. Note the MSA buys ubiquitin nothing here — all
                        # four modes land within 0.03 Å of each other — while nearly
                        # doubling the spread. Unexplained; the wider ceiling on the msa
                        # modes is that extra variance, not a laxer accuracy claim.
                        Mode(use_msa_server=True, use_templates=True): 1.9,
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
                    # Measured 2026-08-04 on of3-p2-155k, MPS, 8 samples, seed 42, as
                    # mean ± sd:
                    #   no_msa-no_templates  16.002 ± 0.388 (gdt_ts 0.042)
                    #   no_msa-templates     16.002 ± 0.388 (gdt_ts 0.042)
                    #   msa-no_templates      0.367 ± 0.049 (gdt_ts 0.993)
                    #   msa-templates         0.247 ± 0.065 (gdt_ts 0.999)
                    #
                    # Unlike ubiquitin, this target lives or dies on the MSA:
                    # single-sequence the model does not find the fold at all (gdt_ts
                    # 0.04), with an MSA it is essentially exact. Which is what the
                    # per-mode encoding exists for — one bound across all four would
                    # have to be ~24 Å and would then wave through a total collapse of
                    # the MSA path.
                    rmsd_max={
                        Mode(use_msa_server=True, use_templates=False): 0.6,
                        # Tighter than its no_templates twin because the template
                        # measurably helps here (0.247 vs 0.367) — each mode is bounded
                        # by its own distribution, not by a shared worst case.
                        Mode(use_msa_server=True, use_templates=True): 0.4,
                    },
                    # The converse claim, and the one no ceiling can make: without an
                    # MSA this target must *stay* unsolved. If the MSA path silently
                    # stopped contributing, every mode would collapse to this one and a
                    # ceiling-only test would still pass. Pinned at half the measured
                    # 16.0 Å, the same margin test_templates.py uses for its template
                    # floor — the failure it guards against is a 40x move, so the exact
                    # value is not delicate.
                    rmsd_min={
                        Mode(use_msa_server=False, use_templates=False): 8.0,
                        Mode(use_msa_server=False, use_templates=True): 8.0,
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
                    # Measured 2026-08-04, same runs as the CA-RMSD above, mean ± sd:
                    #   no_msa-no_templates  20.217 ± 3.727 (centroid 20.1)
                    #   no_msa-templates     20.212 ± 3.719 (centroid 20.1)
                    #   msa-no_templates      0.240 ± 0.045 (centroid  0.21)
                    #   msa-templates         0.427 ± 0.304 (centroid  0.23)
                    # The pose tracks the fold exactly as expected: with an MSA the
                    # ligand sits sub-Ångström in the cavity, far inside the 2 Å that
                    # normally counts as a correct pose; without one the centroid alone
                    # is ~20 Å off, i.e. the ligand is not in a pocket at all because
                    # there is no pocket. No floor is pinned here — the protein's floor
                    # already asserts the no-MSA modes fail, and a ligand cannot find a
                    # pocket that was never folded.
                    rmsd_max={
                        Mode(use_msa_server=True, use_templates=False): 0.4,
                        # Six times the spread of its no_templates twin, and the outlier
                        # is orientational rather than positional: the centroid holds at
                        # 0.16-0.29 Å across all 8 samples while the RMSD reaches 1.06 Å
                        # on two of them. The ring is in the cavity but turned within it,
                        # beyond the 2-fold flip the symmetry search folds out. So this
                        # ceiling bounds a pose that is correct by any practical
                        # criterion; it is wide because toluene's orientation in a
                        # hydrophobic cavity is genuinely underdetermined, not because
                        # the prediction is worse.
                        Mode(use_msa_server=True, use_templates=True): 0.9,
                    },
                ),
            )
        },
        marks=(requires_examples,),
    ),
]

CASE_PARAMS = [pytest.param(case, id=case.id, marks=case.marks) for case in CASES]
MODE_PARAMS = [pytest.param(mode, id=mode.id) for mode in MODES]


def _assert_within_band(
    scores: SampleScores,
    *,
    ceiling: float | None,
    floor: float | None = None,
    what: str,
    query_name: str,
    mode: Mode,
) -> None:
    """Enforce whichever bounds this mode pins, on the mean over samples.

    ``None`` on either side means the mode is measured but makes no claim in that
    direction, which is the default until a number has been observed for it.
    """
    measured = scores.mean
    if ceiling is not None:
        assert measured < ceiling, (
            f"{query_name} [{mode.id}]: mean {what} {measured:.2f} Å exceeds the "
            f"{ceiling} Å ceiling for this mode — {scores}"
        )
    if floor is not None:
        assert measured > floor, (
            f"{query_name} [{mode.id}]: mean {what} {measured:.2f} Å is below the "
            f"{floor} Å floor for this mode, i.e. the prediction is better than this "
            f"mode should be able to manage — {scores}"
        )


def _assert_protein(
    expectation: Expectation,
    query_name: str,
    mode: Mode,
    *,
    sample_cifs: list[Path],
    expected_samples: int,
    ref: Structure,
) -> None:
    """Score the protein fold against the reference, averaged over diffusion samples."""
    protein = expectation.protein
    metrics = measure_samples(
        sample_cifs,
        lambda pred: best_ca_rmsd(
            pred=pred,
            ref=ref,
            ref_chains=protein.ref_chains,
            pred_chains=protein.pred_chains,
        ),
        expected_samples=expected_samples,
    )
    ca_rmsd = SampleScores.of(metrics, lambda m: m.rmsd)
    # gdt_ts comes free with the RMSD and is logged for context, but is not asserted:
    # it saturates on a converged target (7L39 spans 0.991-0.998 while its RMSD moves
    # by 50%), so it would add a calibration surface without adding sensitivity.
    gdt_ts = SampleScores.of(metrics, lambda m: m.gdt_ts)
    logger.info(
        "%s [%s] CA-RMSD %s vs %s | gdt_ts %s",
        query_name,
        mode.id,
        ca_rmsd,
        expectation.ref_cif.name,
        gdt_ts,
    )
    _assert_within_band(
        ca_rmsd,
        ceiling=protein.rmsd_max.get(mode),
        floor=protein.rmsd_min.get(mode),
        what=f"CA-RMSD against {expectation.ref_cif.name}",
        query_name=query_name,
        mode=mode,
    )


def _assert_ligand(
    expectation: Expectation,
    query_name: str,
    mode: Mode,
    *,
    sample_cifs: list[Path],
    expected_samples: int,
    ref: Structure,
) -> None:
    """Score the ligand pose in the frame of the superimposed protein, over samples."""
    ligand = expectation.ligand
    assert ligand is not None  # guarded by the caller
    metrics = measure_samples(
        sample_cifs,
        lambda pred: ligand_pose_metrics(
            pred=pred,
            ref=ref,
            ref_chains=expectation.protein.ref_chains,
            pred_chains=expectation.protein.pred_chains,
            pred_ligand_chain=ligand.pred_chain,
            ref_ligand_chain=ligand.ref_chain,
        ),
        expected_samples=expected_samples,
    )
    rmsd = SampleScores.of(metrics, lambda m: m.rmsd)
    centroid = SampleScores.of(metrics, lambda m: m.centroid_distance)
    logger.info(
        "%s [%s] ligand RMSD %s (centroid %s) vs %s chain %s",
        query_name,
        mode.id,
        rmsd,
        centroid,
        expectation.ref_cif.name,
        ligand.ref_chain,
    )
    _assert_within_band(
        rmsd,
        ceiling=ligand.rmsd_max.get(mode),
        what=(
            f"ligand RMSD against {expectation.ref_cif.name} chain {ligand.ref_chain}"
        ),
        query_name=query_name,
        mode=mode,
    )


def _maybe_assert_accuracy(
    case: InferenceCase,
    query_name: str,
    mode: Mode,
    *,
    output_dir: Path,
) -> None:
    """Score one prediction against its reference, when the case declares one.

    Always logs what it measures, whether or not this mode pins a bound, so a run in an
    unpinned mode still tells you what to record for it — and a run that trips a bound
    still prints the distribution needed to recalibrate.
    """
    expectation = case.expectations.get(query_name)
    if expectation is None:
        return
    sample_cifs = predicted_structure_cifs(output_dir, query_name)
    # Parsed once here and handed down: every sample scores against the same reference,
    # and parsing is the expensive step (see core.metrics.alignment).
    ref = Structure.from_cif(expectation.ref_cif)
    _assert_protein(
        expectation,
        query_name,
        mode,
        sample_cifs=sample_cifs,
        expected_samples=case.num_diffusion_samples,
        ref=ref,
    )
    if expectation.ligand is not None:
        _assert_ligand(
            expectation,
            query_name,
            mode,
            sample_cifs=sample_cifs,
            expected_samples=case.num_diffusion_samples,
            ref=ref,
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
        num_diffusion_samples=case.num_diffusion_samples,
        # Isolate the template cache: without this it lands in a persistent /tmp dir
        # shared across runs, and the template-enabled cases here would see each
        # other's leftovers.
        template_output_dir=tmp_path / "template_data",
    )
    logger.info("Checking output contents at %s", tmp_path)

    msa_output_root = tmp_path / "msas"
    msa_run_directories = [
        path for path in msa_output_root.iterdir() if path.name != "raw"
    ]
    assert len(msa_run_directories) == 1
    msa_run_directory = msa_run_directories[0]
    expected_msa_directories = [
        msa_run_directory / ("main" if mode.use_msa_server else "dummy")
    ]
    if mode.use_msa_server:
        expected_msa_directories.append(
            msa_output_root / "raw" / msa_run_directory.name / "main"
        )
    for directory in expected_msa_directories:
        assert directory.is_dir(), f"Expected MSA directory not found: {directory}"

    for query_name in query_set.queries:
        seed_dir = prediction_dir(tmp_path, query_name)
        assert (seed_dir / "timing.json").exists(), (
            f"Expected output file not found: {seed_dir / 'timing.json'}"
        )
        # Every sample writes its own trio, so a case that draws several must find all
        # of them — checking only sample 1 would let a partial write through, and the
        # accuracy mean is computed over exactly this set.
        for sample in range(1, case.num_diffusion_samples + 1):
            stem = prediction_stem(query_name, sample=sample)
            for name in (
                f"{stem}_confidences.json",
                f"{stem}_confidences_aggregated.json",
                f"{stem}_model.cif",
            ):
                assert (seed_dir / name).exists(), (
                    f"Expected output file not found: {seed_dir / name}"
                )
        _maybe_assert_accuracy(case, query_name, mode, output_dir=tmp_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv"]))
