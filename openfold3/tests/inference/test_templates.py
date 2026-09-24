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

import pytest

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
