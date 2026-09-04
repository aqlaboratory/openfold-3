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

"""Chemical-steering ligand-geometry test.

``test_steering_repairs_ligand_internal_geometry`` runs one query twice, with
``dataset_config_kwargs.steering.enabled`` on and off, and measures the internal
geometry of the predicted ligand in each arm. Nothing else differs, so the
difference is attributable to steering.

The case is FKBP12 with rapamycin (PDB 1FKB), the ligand given by its CCD code
``RAP`` rather than a SMILES string so the chemistry comes from the PDB
component dictionary. Rapamycin is a 31-membered macrocycle with 65 heavy
atoms: flexible enough that a single-sequence prediction strains it, which is
what gives the test something to detect. The protein is 107 residues, so both
arms together run in about two minutes.

What is measured, and why
-------------------------
The metric is the worst violation of the ligand's RDKit distance-geometry
bounds -- how far, in Angstrom, any restrained atom pair sits outside its
permitted window. This is the same family of check PoseBusters applies for
internal ligand validity (bond lengths, bond angles, internal steric clash),
computed the same way, from ``GetMoleculeBoundsMatrix`` on the reference
molecule.

It is also, deliberately and openly, the quantity steering optimizes. That
makes this an end-to-end demonstration that guidance reaches the sampler and
changes the output chemistry, not independent evidence that the chemistry is
*right*. The independent check is the ring geometry asserted alongside it, and
foldsteer's PoseBusters benchmark (66.0% -> 89.3% valid) is the external
evidence.

Two facts worth recording, both measured on this case:

* The violations are entirely nonbonded 1-4 and 1-5 pairs -- torsional strain,
  atoms folded closer than any reachable torsion allows (e.g. C6-O4 at 3.42 A
  against a 4.28 A bound). Bond lengths and bond angles were already correct in
  both arms, so this term is not fixing those here.
* The *unbuffered* bounds matrix is not a usable yardstick: RDKit's own
  generated conformer for this molecule violates it by up to 2.12 A over 15
  pairs, worse than the predictions do. The buffered bounds the package
  actually restrains against are satisfied exactly by that conformer, which is
  what makes "zero violation" a meaningful target rather than an unreachable
  one (``openfold3/steering/tests/test_featurization.py`` pins that).

The metrics themselves live in ``openfold3.core.metrics.ligand_geometry`` and
are unit-tested in ``openfold3/tests/test_ligand_geometry.py``, which needs
neither an accelerator nor weights.

Requires an accelerator (CUDA, ROCm or MPS) and downloaded model weights; skips
otherwise.

Run with:
    pytest openfold3/tests/inference/test_chemical_steering.py
"""

import logging
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from openfold3.core.data.primitives.structure.query import (
    StructureWithReferenceMolecules,
    structure_with_ref_mols_from_query,
)
from openfold3.core.data.primitives.structure.tokenization import (
    add_token_positions,
    tokenize_atom_array,
)
from openfold3.core.metrics.alignment import Structure
from openfold3.core.metrics.ligand_geometry import (
    mean_ring_torsion,
    saturated_ring_atom_names,
    worst_bounds_violation,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
    Query,
)
from openfold3.steering.featurization import build_context
from openfold3.steering.potentials import DistanceBoundsPotential
from openfold3.tests.inference.helpers import (
    SampleScores,
    measure_samples,
    predicted_structure_cifs,
    run_inference,
)
from openfold3.tests.utils.compare_utils import skip_unless_accelerator_available

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.slow]

QUERY_NAME = "fkbp12_rapamycin"
PROTEIN_CHAIN_ID = "A"
LIGAND_CHAIN_ID = "B"

#: Human FKBP12 (UniProt P62942) with rapamycin — the 1FKB complex.
CHAINS = [
    {
        "molecule_type": "protein",
        "chain_ids": [PROTEIN_CHAIN_ID],
        "sequence": (
            "MGVQVETISPGDGRTFPKRGQTCVVHYTGMLEDGKKFDSSRDRNKPFKFMLGKQEVIRGWEEGVAQM"
            "SVGQRAKLTISPDYAYGATGHPGIIPPHATLVFDVELLKLE"
        ),
    },
    {
        "molecule_type": "ligand",
        "chain_ids": [LIGAND_CHAIN_ID],
        "ccd_codes": ["RAP"],
    },
]

#: Run single-sequence.
USE_MSA_SERVER = False
USE_TEMPLATES = False

#: Fewer samples than SCORED_DIFFUSION_SAMPLES: this asserts on a within-sample
#: geometric property, not on accuracy against an experimental structure, and
#: the separation between the arms is two orders of magnitude wide.
NUM_DIFFUSION_SAMPLES = 5

# Calibrated on one run of 5 samples per arm, of3-ob-2025-06-30-174k on a
# CUDA GB10:
#   steering off  worst violation 0.74, 0.86, 0.88, 1.00, 1.15 Å (mean 0.93),
#                 10-13 violated pairs per sample
#   steering on   worst violation 0.000 x4, 0.003 Å (mean 0.001),
#                 0-1 violated pairs per sample
# The thresholds sit an order of magnitude inside that gap on both sides, so
# they tolerate hardware and precision variance but still fail if guidance
# stops reaching the sampler (then on collapses onto off).
STEERING_ON_VIOLATION_MAX_ANGSTROM = 0.10
STEERING_OFF_VIOLATION_MIN_ANGSTROM = 0.30

#: A saturated all-carbon six-ring sits in a chair, ~55° of torsion at every
#: bond. Guidance must not flatten it toward 0° — the failure mode Tom
#: Goddard's badchem survey documents for predicted ligands, where cyclohexane
#: came back planar:
#: https://www.rbvi.ucsf.edu/chimerax/data/ligchem-feb2026/badchem.html
#: Measured ~53-55° in both arms here; the floor is well below that.
SATURATED_RING_TORSION_MIN_DEGREES = 45.0


def _query_structure() -> StructureWithReferenceMolecules:
    """The tokenized query structure and its reference molecules."""
    structure = structure_with_ref_mols_from_query(
        Query.model_validate({"chains": CHAINS})
    )
    tokenize_atom_array(structure.atom_array)
    add_token_positions(structure.atom_array)
    return structure


def _ligand_coords_by_name(prediction: Structure) -> dict[str, np.ndarray]:
    """Predicted ligand coordinates keyed by atom name, as the ring metrics
    want them."""
    ligand = prediction.heavy_atoms(LIGAND_CHAIN_ID)
    return {
        str(name): np.asarray(coord, dtype=float)
        for name, coord in zip(ligand.atom_name, ligand.coord, strict=True)
    }


def _ligand_coordinates(
    prediction: Structure, structure: StructureWithReferenceMolecules
) -> torch.Tensor:
    """Predicted ligand coordinates, laid out on the query's global atom axis.

    Mapped by atom name rather than by position, so this does not depend on the
    predicted cif listing atoms in the query's order. Non-ligand atoms stay at
    the origin and are never read: the restraints are intramolecular, so every
    index they carry is a ligand atom.
    """
    ligand = prediction.heavy_atoms(LIGAND_CHAIN_ID)
    coords_by_name = {
        str(name): np.asarray(coord, dtype=np.float32)
        for name, coord in zip(ligand.atom_name, ligand.coord, strict=True)
    }
    atom_array = structure.atom_array
    coords = torch.zeros((len(atom_array), 3), dtype=torch.float32)
    for index, (name, chain) in enumerate(
        zip(atom_array.atom_name, atom_array.chain_id, strict=True)
    ):
        if str(chain) == LIGAND_CHAIN_ID:
            coords[index] = torch.from_numpy(coords_by_name[str(name)])
    return coords


def _run(*, steering_enabled: bool, out_dir: Path) -> list[Path]:
    """Run one condition and return its predicted sample cifs, in sample order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_inference(
        InferenceQuerySet.model_validate({"queries": {QUERY_NAME: {"chains": CHAINS}}}),
        out_dir,
        use_msa_server=USE_MSA_SERVER,
        use_templates=USE_TEMPLATES,
        num_diffusion_samples=NUM_DIFFUSION_SAMPLES,
        extra_yaml=textwrap.dedent(f"""\
            dataset_config_kwargs:
              steering:
                enabled: {str(steering_enabled).lower()}
            """),
    )
    return predicted_structure_cifs(out_dir, QUERY_NAME)


@skip_unless_accelerator_available()
@pytest.mark.inference_verification
def test_steering_repairs_ligand_internal_geometry(tmp_path: Path) -> None:
    """Steering must clear the predicted ligand's distance-bounds violations."""
    structure = _query_structure()
    restraints = build_context(
        structure.atom_array,
        structure.processed_reference_mols,
        n_atoms=len(structure.atom_array),
    ).restraints[DistanceBoundsPotential.name]
    rings = saturated_ring_atom_names(structure.processed_reference_mols[-1].mol)

    def _measure(sample_cifs: list[Path]) -> tuple[SampleScores, SampleScores]:
        measurements = measure_samples(
            sample_cifs,
            lambda prediction: (
                worst_bounds_violation(
                    _ligand_coordinates(prediction, structure), restraints
                ),
                mean_ring_torsion(_ligand_coords_by_name(prediction), rings),
            ),
            expected_samples=NUM_DIFFUSION_SAMPLES,
        )
        return (
            SampleScores.of(measurements, lambda pair: pair[0]),
            SampleScores.of(measurements, lambda pair: pair[1]),
        )

    on_violation, on_torsion = _measure(
        _run(steering_enabled=True, out_dir=tmp_path / "on")
    )
    off_violation, off_torsion = _measure(
        _run(steering_enabled=False, out_dir=tmp_path / "off")
    )
    logger.info(
        "%s worst distance-bounds violation (Å) | steering on %s | off %s",
        QUERY_NAME,
        on_violation,
        off_violation,
    )
    logger.info(
        "%s mean saturated-ring |torsion| (°) | steering on %s | off %s",
        QUERY_NAME,
        on_torsion,
        off_torsion,
    )

    assert on_violation.mean < STEERING_ON_VIOLATION_MAX_ANGSTROM, (
        f"{QUERY_NAME}: steering left the ligand strained — mean worst violation "
        f"{on_violation.mean:.3f} Å over {NUM_DIFFUSION_SAMPLES} samples exceeds "
        f"the {STEERING_ON_VIOLATION_MAX_ANGSTROM} Å ceiling"
    )
    assert off_violation.mean > STEERING_OFF_VIOLATION_MIN_ANGSTROM, (
        f"{QUERY_NAME}: the unsteered ligand was already clean — mean worst "
        f"violation {off_violation.mean:.3f} Å is below the "
        f"{STEERING_OFF_VIOLATION_MIN_ANGSTROM} Å floor, so this case no longer "
        "demonstrates anything about steering"
    )
    assert on_torsion.mean > SATURATED_RING_TORSION_MIN_DEGREES, (
        f"{QUERY_NAME}: steering flattened the saturated ring — mean |torsion| "
        f"{on_torsion.mean:.1f}° is below the "
        f"{SATURATED_RING_TORSION_MIN_DEGREES}° floor a chair conformation sits "
        "well above"
    )
