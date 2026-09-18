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
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from biotite.structure.io import pdbx

from openfold3.core.metrics.alignment import Structure, best_ca_rmsd
from openfold3.tests.inference.helpers import (
    MMCIFS_DIR,
    SCORED_DIFFUSION_SAMPLES,
    Mode,
    SampleScores,
    measure_samples,
    predicted_structure_cifs,
    query_set_from_chains,
    run_inference,
)
from openfold3.tests.utils.compare_utils import skip_unless_accelerator_available

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Number of diffusion samples per condition. The samples cluster (all near the reference
# with a template, all far without), so the mean over samples is representative and
# robust. Shared with test_inference_full so the two cannot drift — see
# SCORED_DIFFUSION_SAMPLES for why a single sample is not a portable measurement.
NUM_DIFFUSION_SAMPLES = SCORED_DIFFUSION_SAMPLES

#: The two conditions compared. Both run single-sequence, so the template flag is the
#: only difference between them — that is what makes the RMSD gap attributable to it.
TEMPLATE_OFF = Mode(use_msa_server=False, use_templates=False)
TEMPLATE_ON = Mode(use_msa_server=False, use_templates=True)

pytestmark = [pytest.mark.slow]


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
    # Observed on of3-p2-155k, tightly clustered, at both sample counts:
    #   N=5           off mean ≈ 16.4 Å (15.5-17.5), on mean ≈ 0.26 Å (0.23-0.30)
    #   N=8, MPS      off 16.58 ± 0.68,              on 0.26 ± 0.02
    # Raising the sample count moved neither mean, and the MPS numbers match hardware
    # that does not share its RNG stream — so this case is genuinely insensitive to the
    # draw, not merely reproducible within one. Thresholds keep a wide margin so they
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
    # Human HCK (SH3-SH2-kinase) templated on c-Src 1Y57 — a *homologous* template
    # (61.7% identity to 1Y57_A, with indels), unlike 1a8q's self-template. This is the
    # realistic user-supplied-template case, and the one that regressed in issue #294:
    # gapped alignment columns reached the template cache, the query/template residue
    # counts disagreed, and the template was silently dropped.
    #
    # Expect partial improvement, not the near-native fit of a self-template. The
    # on-condition samples are widely spread, so the mean, not any single sample, is the
    # meaningful quantity. If templates are ignored again, on collapses onto off
    # (~23.7 Å) and both the max and separation checks fail.
    #
    # Three runs on of3-p2-155k, 5 samples each (one on the original hardware, two on
    # an NVIDIA GB10) put the means in a narrow band:
    #     off mean  23.39 / 23.70 / 23.83   (per-sample 21.8-26.1)
    #     on  mean  12.70 / 12.74 / 12.78   (per-sample  8.6-22.8)
    #     off - on  10.61 / 11.00 / 11.09
    # Those three agree to 0.08 Å, which is less reassuring than it looks: all three ran
    # on CUDA-family hardware, which shares one RNG stream, so they replay the same draw.
    # A fourth run at 8 samples on MPS — an independent stream — is the real check:
    #     off 23.77 ± 1.25   on 12.07 ± 2.83   off - on 11.70
    # It agrees to within 5%, so the case is genuinely insensitive to the draw. Note the
    # on-condition is not the tidy bimodal it first appeared: MPS spreads continuously
    # over 8.2-15.5 with nothing near the no-template value.
    #
    # Bounds leave >=32% headroom to the nearest observation. The with-template max is
    # the tightest of the three at 3.9 SE above the measured mean — right at the 4 SE
    # that test_inference_full's rule asks for. That rule's second term (1.5x the mean)
    # would demand 18.1 Å here, which would collide with the 18.0 Å off floor and make
    # the pair meaningless; it is a stand-in for *unmeasured* cross-backend offset, and
    # for this case that offset is now measured at 5%. So 16.0 stands. The separation
    # check is the robust claim regardless: off and on move together with hardware, and
    # it holds with ~2x margin.
    TemplateRmsdCase(
        pdb_id="1y57",
        chain="A",
        sequence=(
            "DIIVVALYDYEAIHHEDLSFQKGDQMVVLEESGEWWKARSLATRKEGYIPSNYVARVDSLETEEWFFKGIS"
            "RKDAERQLLAPGNMLGSFMIRDSETTKGSYSLSVRDYDPRQGDTVKHYKIRTLDNGGFYISPRSTFSTLQE"
            "LVDHYKKGNDGLCQKLSVPCMSSKPQKPWEKDAWEIPRESLKLEKKLGAGQFGEVWMATYNKHTKVAVKTM"
            "KPGSMSVEAFLAEANVMKTLQHDKLVKLHAVVTKEPIYIITEFMAKGSLLDFLKSDEGSKQPLPKLIDFSA"
            "QIAEGMAFIEQRNYIHRDLRAANILVSASLVCKIADFGLARVIEDNEYTAREGAKFPIKWTAPEAINFGSF"
            "TIKSDVWSFGILLMEIVTYGRIPYPGMSNPEVIRALERGYRMPRPENCPEELYNIMMRCWKNRPEERPTFE"
            "YIQSVLDDFYTATESQYQQQP"
        ),
        no_template_rmsd_min_angstrom=18.0,
        with_template_rmsd_max_angstrom=16.0,
        rmsd_separation_min_angstrom=6.0,
    ),
    # The target from issue #406, where CIF-direct templates were reported to be ignored
    # entirely. Kept as a case because the reporter's own measurements are the reference
    # point for the provenance tests below, and because it is a self-template like 1a8q:
    # the near-native fit makes "was this template used?" unambiguous.
    #
    # One run on of3-p2-155k, 8 samples, CUDA (GB10):
    #     off 22.08 +- 1.74  (per-sample 18.95-23.95)
    #     on   0.47 +- 0.02  (per-sample  0.44-0.52)
    #     off - on 21.61
    # The reporter measured the same shape on an A10G via the pip package -- off
    # 20.07 +- 1.08, on 0.14 +- 0.01 -- so the effect reproduces across two backends and
    # two builds even though the absolute numbers differ. Unlike 1y57 above, this has
    # been measured on one backend only, so the bounds are set generously rather than
    # tuned: the off floor sits well under the lowest sample drawn (18.95), the on
    # ceiling is ~4x the measured mean, and the separation ~2.7x the measured gap.
    TemplateRmsdCase(
        pdb_id="6tel",
        chain="A",
        sequence=(
            "GPGEKLELRLKSPVGAEPAVYPWPLPVYDKHHDAAHEIIETIRWVCEEIPDLKLAMENYVL"
            "IDYDTKSFESMQRLCDKYNRAIDSIHQLWKGTTQPMKLNTRPSTGLLRHILQQVYNHSVTD"
            "PEKLNNYEPFSPEVYGETSFDLVAQMIDEIKMTDDDLFVDLGSGVGQVVLQVAAATNCKHH"
            "YGVEKADIPAKYAETMDREFRKWMKWYGKKHAEYTLERGDFLSEEWRERIANTSVIFVNNF"
            "AFGPEVDHQLKERFANMKEGGRIVSSKPFAPLNFRINSRNLSDIGTIMRVVELSPLKGSVS"
            "WTGKPVSYYLHTIDRTILENYFSSLKNPG"
        ),
        no_template_rmsd_min_angstrom=10.0,
        with_template_rmsd_max_angstrom=2.0,
        rmsd_separation_min_angstrom=8.0,
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

    # The reference is parsed once for the whole batch of samples; the prediction is a
    # monomer, so let its chains be discovered — the chain id the writer emits carries
    # no information here.
    reference = Structure.from_cif(_ref_cif(case))
    metrics = measure_samples(
        predicted_structure_cifs(out_dir, key),
        lambda pred: best_ca_rmsd(pred, reference, ref_chains=(case.chain,)),
        expected_samples=NUM_DIFFUSION_SAMPLES,
    )
    rmsds = SampleScores.of(metrics, lambda m: m.rmsd)
    logger.info("%s [%s] CA-RMSD %s", key, mode.id, rmsds)
    return rmsds.mean


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


# ---------------------------------------------------------------------------
# Template provenance: the pinned CIF is the coordinate source (issue #406)
# ---------------------------------------------------------------------------
#
# `test_template_lowers_rmsd` above pins a template file whose name happens to be its
# PDB ID and whose contents happen to match the deposited entry, so it cannot tell
# whether the prediction used the *provided file* or a PDB copy resolved from its
# filename. Issue #406 reported that it was in fact the latter: `template_cif_paths`
# fed the alignment only, while coordinates came from `<structure_directory>/<stem>.cif`
# (downloaded when absent). Two user-visible consequences, and one test for each:
#
#   * a template whose filename is not a PDB ID was silently dropped, and
#   * an edited or non-deposited template was silently replaced by the PDB version.
#
# Both are measured against the same clean-template run, so no new RMSD thresholds are
# introduced: the case's own `with_template_rmsd_max_angstrom` and
# `rmsd_separation_min_angstrom` carry over.

#: Per-atom noise applied to the perturbed template, in Angstrom. Deliberately *not* a
#: rigid translation: template features are built from intra-template geometry, which a
#: rigid motion leaves untouched. Only a per-atom displacement actually destroys the
#: fold, which is what makes "did these coordinates get used?" observable at all.
PERTURBATION_NOISE_ANGSTROM = 10.0

#: Fixed so the perturbed template is identical on every run and across machines; the
#: measurement varies with the diffusion draw, and this keeps the input from varying too.
PERTURBATION_SEED = 406


def _scrambled_template_cif(
    case: TemplateRmsdCase, destination: Path, *, noise_angstrom: float
) -> Path:
    """Copy the case's reference cif to `destination`, displacing every atom.

    Rewrites the `atom_site` coordinate columns in place rather than rebuilding the
    file, so the scrambled copy is byte-for-byte the original except for the numbers
    that matter: same atoms in the same order, same entity and revision metadata the
    template pipeline reads for sequences and release dates.
    """
    cif = pdbx.CIFFile.read(str(_ref_cif(case)))
    atom_site = cif.block["atom_site"]
    rng = np.random.default_rng(PERTURBATION_SEED)
    for axis in ("Cartn_x", "Cartn_y", "Cartn_z"):
        coordinates = atom_site[axis].as_array(float)
        atom_site[axis] = pdbx.CIFColumn(
            coordinates + rng.normal(0.0, noise_angstrom, coordinates.shape)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cif.write(str(destination))
    return destination


def _pinned_template_cif(case: TemplateRmsdCase, destination: Path) -> Path:
    """Copy the case's reference cif to `destination`, contents untouched."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(_ref_cif(case), destination)
    return destination


def _offline_template_yaml(case: TemplateRmsdCase, out_dir: Path) -> str:
    """Runner settings that stage the deposited entry and forbid fetching.

    The deposited structure is put exactly where a filename-keyed lookup would find it,
    so the defect this guards against has every opportunity to occur; fetching is then
    disabled so the outcome cannot depend on the network. A correct pipeline ignores
    both and reads the pinned file.
    """
    structure_dir = out_dir / "deposited_structures"
    _pinned_template_cif(case, structure_dir / f"{case.pdb_id}.cif")
    return textwrap.dedent(f"""\
        template_preprocessor_settings:
          output_directory: {out_dir / "template_data"}
          structure_directory: {structure_dir}
          fetch_missing_structures: false
        """)


def _mean_ca_rmsd_with_template(
    case: TemplateRmsdCase,
    *,
    template_cif: Path,
    key: str,
    tmp_path: Path,
) -> float:
    """Mean CA-RMSD for one pinned template file, single-sequence."""
    out_dir = tmp_path / key
    out_dir.mkdir(parents=True, exist_ok=True)
    query_set = query_set_from_chains(
        key,
        {
            "molecule_type": "protein",
            "chain_ids": [case.chain],
            "sequence": case.sequence,
            "template_cif_paths": [str(template_cif)],
            "template_cif_chain_ids": [case.chain],
        },
    )
    run_inference(
        query_set,
        out_dir,
        use_msa_server=False,
        use_templates=True,
        num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
        extra_yaml=_offline_template_yaml(case, out_dir),
    )
    reference = Structure.from_cif(_ref_cif(case))
    rmsds = SampleScores.of(
        measure_samples(
            predicted_structure_cifs(out_dir, key),
            lambda pred: best_ca_rmsd(pred, reference, ref_chains=(case.chain,)),
            expected_samples=NUM_DIFFUSION_SAMPLES,
        ),
        lambda m: m.rmsd,
    )
    logger.info("%s CA-RMSD %s", key, rmsds)
    return rmsds.mean


@skip_unless_accelerator_available()
@pytest.mark.inference_verification
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.pdb_id)
def test_template_filename_does_not_change_the_prediction(case, tmp_path):
    """The same template under a non-PDB-ID filename must work just as well.

    Byte-identical contents, pinned twice under two names. Any difference between the
    two runs is attributable to the filename alone, which should carry no meaning.
    Before the fix the non-PDB-ID copy resolved to nothing, was dropped without an
    error, and the query fell back to the single-sequence (no-template) prediction.

    The chosen name also carries underscores, which is not incidental: template IDs are
    `f"{entry_id}_{chain_id}"` and the entry ID here is the filename stem, so a stem
    containing "_" once crashed the run outright. Ordinary names for a user-supplied
    template -- `model_1_rank_2`, `6TEL_relaxed` -- all hit that.

    Observed on of3-p2-155k, 8 samples, CUDA (GB10), as (pdb-name, custom-name) means:
        1a8q   0.16 / 0.16     per-sample values identical
        1y57  12.05 / 12.04    per-sample agreeing to <=0.02 A
        6tel   0.47 / 0.47     per-sample values identical
    The bound asserted below is the case's own with-template ceiling rather than that
    agreement: matching draws are what a correct pipeline *should* produce from
    identical inputs and a fixed seed, but tying a regression test to sample-for-sample
    reproducibility would make it a determinism test for whatever backend it runs on --
    note 1y57 already drifts in the third digit here.

    Regression test for https://github.com/aqlaboratory/openfold-3/issues/406
    """
    deposited_name = _pinned_template_cif(
        case, tmp_path / "pinned" / f"{case.pdb_id}.cif"
    )
    custom_name = _pinned_template_cif(
        case, tmp_path / "pinned" / f"{case.pdb_id}_not_a_pdb_id.cif"
    )

    rmsd_pdb_name = _mean_ca_rmsd_with_template(
        case,
        template_cif=deposited_name,
        key=f"{case.pdb_id}_pdb_name",
        tmp_path=tmp_path,
    )
    rmsd_custom_name = _mean_ca_rmsd_with_template(
        case,
        template_cif=custom_name,
        key=f"{case.pdb_id}_custom_name",
        tmp_path=tmp_path,
    )
    logger.info(
        "%s mean RMSD pdb_name=%.2f custom_name=%.2f",
        case.pdb_id,
        rmsd_pdb_name,
        rmsd_custom_name,
    )

    assert rmsd_pdb_name < case.with_template_rmsd_max_angstrom, (
        f"{case.pdb_id}: the control run did not use its template — "
        f"got {rmsd_pdb_name:.2f}, need < {case.with_template_rmsd_max_angstrom}"
    )
    assert rmsd_custom_name < case.with_template_rmsd_max_angstrom, (
        f"{case.pdb_id}: template dropped because its filename is not a PDB ID — "
        f"got {rmsd_custom_name:.2f}, need < {case.with_template_rmsd_max_angstrom} "
        f"(same file named {case.pdb_id}.cif gave {rmsd_pdb_name:.2f})"
    )


@skip_unless_accelerator_available()
@pytest.mark.inference_verification
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.pdb_id)
def test_template_coordinates_come_from_the_provided_file(case, tmp_path):
    """Scrambling the pinned template's coordinates must degrade the prediction.

    The negative control for the test above: if the provided coordinates are the ones
    being featurized, destroying them has to cost accuracy. Before the fix it cost
    nothing — the deposited entry was loaded by filename and the edit never reached the
    model, so a 10 A scrambled template scored the same as a pristine one.

    Observed on of3-p2-155k, 8 samples, CUDA (GB10), as clean -> scrambled means:
        1a8q   0.16 -> 16.19    gap 16.03 against 5.0 required
        1y57  12.05 -> 22.44    gap 10.39 against 6.0 required  (tightest, 1.7x)
        6tel   0.47 -> 21.55    gap 21.08 against 8.0 required
    In each the scrambled figure lands on that case's own *no-template* value (~16.4,
    ~23.7 and 22.08, recorded above). That is the expected ceiling rather than a lucky
    draw: a
    template whose internal geometry is noise carries no more information than no
    template at all, so the prediction cannot degrade past the no-template baseline and
    the margin will not shrink for reasons to do with what is being tested.

    6tel is the issue's own target, so this pair is the direct analogue of the table in
    the report: there `template_nonpdbname` collapsed onto no-template (20.14 vs 20.07)
    and `template_pdbname_perturbed` was indistinguishable from a pristine template
    (0.15 vs 0.14). Both relations are now inverted.

    Regression test for https://github.com/aqlaboratory/openfold-3/issues/406
    """
    clean = _pinned_template_cif(case, tmp_path / "pinned" / f"{case.pdb_id}.cif")
    perturbed = _scrambled_template_cif(
        case,
        tmp_path / "scrambled" / f"{case.pdb_id}.cif",
        noise_angstrom=PERTURBATION_NOISE_ANGSTROM,
    )

    rmsd_clean = _mean_ca_rmsd_with_template(
        case, template_cif=clean, key=f"{case.pdb_id}_clean", tmp_path=tmp_path
    )
    rmsd_perturbed = _mean_ca_rmsd_with_template(
        case, template_cif=perturbed, key=f"{case.pdb_id}_perturbed", tmp_path=tmp_path
    )
    logger.info(
        "%s mean RMSD clean=%.2f perturbed=%.2f",
        case.pdb_id,
        rmsd_clean,
        rmsd_perturbed,
    )

    assert rmsd_clean < case.with_template_rmsd_max_angstrom, (
        f"{case.pdb_id}: the control run did not use its template — "
        f"got {rmsd_clean:.2f}, need < {case.with_template_rmsd_max_angstrom}"
    )
    assert rmsd_perturbed - rmsd_clean > case.rmsd_separation_min_angstrom, (
        f"{case.pdb_id}: scrambling the pinned template changed nothing, so its "
        f"coordinates were not the ones used — clean={rmsd_clean:.2f} "
        f"perturbed={rmsd_perturbed:.2f} (need gap > "
        f"{case.rmsd_separation_min_angstrom})"
    )
