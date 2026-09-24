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

"""Unit tests for :mod:`openfold3.core.metrics.alignment`.

Everything runs against committed mmCIFs under ``test_data/mmcifs/``, so the fixtures are
real structures rather than hand-built arrays. Where a test needs a "prediction", one is
derived from a reference by applying a known transformation — a rigid motion, a
translation of one chain, a symmetry relabelling — which makes the expected answer exact
and independent of the code under test.

Run with:
    pytest openfold3/tests/core/metrics/test_alignment.py
"""

import math
from functools import cache
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest
import torch

import openfold3
from openfold3.core.metrics.alignment import (
    MAX_PERMUTED_CHAINS,
    Structure,
    _best_ca_assignment,
    best_ca_rmsd,
    ligand_pose_metrics,
    symmetry_mappings,
)
from openfold3.core.utils.geometry.kabsch_alignment import apply_transformation

MMCIFS_DIR = Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs"

#: Coordinates round-trip through float32 in the mmCIF, and the superposition itself is
#: iterative, so "exactly zero" lands around 1e-6 Å.
ZERO_RMSD_TOL = 1e-4


@cache
def load_structure(pdb_id: str) -> Structure:
    """Parse a committed reference, once per session — parsing dominates runtime."""
    return Structure.from_cif(MMCIFS_DIR / f"{pdb_id}.cif")


def rotation_about_z(radians: float) -> np.ndarray:
    cos, sin = math.cos(radians), math.sin(radians)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def moved(
    structure: Structure,
    *,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> Structure:
    """A copy of *structure* under a rigid motion — a perfect prediction of itself."""
    array = structure.atom_array.copy()
    coord = array.coord
    if rotation is not None:
        coord = coord @ np.asarray(rotation).T
    if translation is not None:
        coord = coord + np.asarray(translation)
    array.coord = coord
    return Structure(path=structure.path, atom_array=array)


def with_chain_translated(
    structure: Structure, chain: str, offset: np.ndarray
) -> Structure:
    """A copy with a single chain displaced, leaving every other chain in place."""
    array = structure.atom_array.copy()
    mask = array.chain_id == chain
    array.coord[mask] = array.coord[mask] + np.asarray(offset)
    return Structure(path=structure.path, atom_array=array)


def with_chain_relabelled(
    structure: Structure, chain: str, mapping: tuple[int, ...]
) -> Structure:
    """A copy whose *chain* coordinates are permuted by a graph automorphism.

    The molecule is geometrically unchanged, only its atom order differs — so a
    symmetry-aware comparison must score it as identical.
    """
    array = structure.atom_array.copy()
    indices = np.where(array.chain_id == chain)[0]
    array.coord[indices] = array.coord[indices][list(mapping)]
    return Structure(path=structure.path, atom_array=array)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

CA_COUNT_CASES = [
    pytest.param("1ubq", {"A": 76}, id="monomer"),
    pytest.param("1hf9", {"A": 41, "B": 41}, id="homodimer"),
    pytest.param("4i6p", {"A": 84, "B": 82}, id="dimer-with-unequal-gaps"),
    pytest.param(
        "1kd8",
        {"A": 35, "B": 35, "C": 35, "D": 35, "E": 34, "F": 35},
        id="hexamer",
    ),
]


@pytest.mark.parametrize(("pdb_id", "expected"), CA_COUNT_CASES)
def test_ca_positions_by_chain_counts(pdb_id, expected):
    positions = load_structure(pdb_id).ca_positions_by_chain
    assert {chain: len(res) for chain, res in positions.items()} == expected


def test_from_cif_keeps_the_source_path():
    structure = load_structure("1ubq")
    assert structure.path == MMCIFS_DIR / "1ubq.cif"
    assert len(structure.atom_array) > 0


def test_ca_positions_are_cached():
    structure = Structure.from_cif(MMCIFS_DIR / "1ubq.cif")
    assert structure.ca_positions_by_chain is structure.ca_positions_by_chain


def test_ca_positions_reject_duplicate_residues():
    """Two CAs for one residue would silently drop residues, so it must raise."""
    array = load_structure("1ubq").atom_array
    ca = array[(array.atom_name == "CA") & (~array.hetero)]
    duplicated = Structure(
        path=Path("duplicated.cif"), atom_array=struc.concatenate([ca, ca])
    )
    with pytest.raises(ValueError, match="more than one CA"):
        _ = duplicated.ca_positions_by_chain


HEAVY_ATOM_CASES = [
    pytest.param("7l39", "D", "MBN", 7, id="toluene"),
    pytest.param("7l39", "B", "TRS", 8, id="tris-buffer"),
    pytest.param("7l39", "E", "BME", 4, id="mercaptoethanol"),
    pytest.param("4zey", "B", "SO4", 5, id="sulfate"),
    pytest.param("3u8v", "C", "NI", 1, id="nickel-ion"),
]


@pytest.mark.parametrize(("pdb_id", "chain", "res_name", "n_atoms"), HEAVY_ATOM_CASES)
def test_heavy_atoms_selects_the_ligand(pdb_id, chain, res_name, n_atoms):
    selected = load_structure(pdb_id).heavy_atoms(chain)
    assert len(selected) == n_atoms
    assert set(map(str, selected.res_name)) == {res_name}
    assert "H" not in set(map(str, selected.element))


def test_heavy_atoms_excludes_water():
    """7l39 chain F is solvent only, so nothing is left to compare."""
    with pytest.raises(ValueError, match="no non-water heavy atoms"):
        load_structure("7l39").heavy_atoms("F")


def test_heavy_atoms_unknown_chain_names_the_file():
    with pytest.raises(ValueError, match="7l39.cif"):
        load_structure("7l39").heavy_atoms("ZZ")


# ---------------------------------------------------------------------------
# best_ca_rmsd
# ---------------------------------------------------------------------------

IDENTITY_CASES = [
    pytest.param("1ubq", ("A",), id="monomer"),
    pytest.param("1a8q", ("A",), id="long-monomer"),
    pytest.param("1hf9", ("A", "B"), id="homodimer"),
    pytest.param("1kd8", ("A", "B", "C", "D"), id="four-copies"),
]


@pytest.mark.parametrize(("pdb_id", "ref_chains"), IDENTITY_CASES)
def test_structure_against_itself_scores_zero(pdb_id, ref_chains):
    structure = load_structure(pdb_id)
    metrics = best_ca_rmsd(
        structure, structure, ref_chains=ref_chains, pred_chains=ref_chains
    )
    assert metrics.rmsd == pytest.approx(0.0, abs=ZERO_RMSD_TOL)
    assert metrics.gdt_ts == pytest.approx(1.0)
    assert metrics.gdt_ha == pytest.approx(1.0)


RIGID_MOTION_CASES = [
    pytest.param(None, np.array([10.0, -5.0, 3.0]), id="translation-only"),
    pytest.param(rotation_about_z(0.7), None, id="rotation-only"),
    pytest.param(
        rotation_about_z(2.5),
        np.array([-4.0, 8.0, 12.0]),
        id="rotation-and-translation",
    ),
    pytest.param(
        rotation_about_z(math.pi), np.array([100.0, 100.0, 100.0]), id="far-displaced"
    ),
]


@pytest.mark.parametrize(("rotation", "translation"), RIGID_MOTION_CASES)
def test_rmsd_is_invariant_under_rigid_motion(rotation, translation):
    """Superposition must remove any rigid motion, however large."""
    reference = load_structure("1ubq")
    prediction = moved(reference, rotation=rotation, translation=translation)
    metrics = best_ca_rmsd(prediction, reference, ref_chains=("A",))
    assert metrics.rmsd == pytest.approx(0.0, abs=ZERO_RMSD_TOL)


SCRAMBLE_CASES = [
    pytest.param("1hf9", ("A", "B"), ("B", "A"), id="dimer-swapped"),
    pytest.param("1kd8", ("A", "B", "C"), ("C", "A", "B"), id="trimer-rotated"),
    pytest.param(
        "1kd8", ("A", "B", "C", "D"), ("D", "C", "B", "A"), id="four-copies-reversed"
    ),
]


@pytest.mark.parametrize(("pdb_id", "pred_chains", "ref_chains"), SCRAMBLE_CASES)
def test_finds_the_matching_chain_assignment(pdb_id, pred_chains, ref_chains):
    """Chains are paired as sets, so a scrambled reference order still scores zero."""
    structure = load_structure(pdb_id)
    metrics = best_ca_rmsd(
        structure, structure, ref_chains=ref_chains, pred_chains=pred_chains
    )
    assert metrics.rmsd == pytest.approx(0.0, abs=ZERO_RMSD_TOL)


def test_pred_chains_default_to_every_polymer_chain():
    structure = load_structure("1ubq")
    metrics = best_ca_rmsd(structure, structure, ref_chains=("A",))
    assert metrics.rmsd == pytest.approx(0.0, abs=ZERO_RMSD_TOL)


BIOTITE_AGREEMENT_CASES = [
    pytest.param("1kd8", "A", "B", id="1kd8-A-vs-B"),
    pytest.param("1kd8", "C", "D", id="1kd8-C-vs-D"),
    pytest.param("1hf9", "A", "B", id="1hf9-A-vs-B"),
    pytest.param("3u8v", "A", "B", id="3u8v-A-vs-B"),
    pytest.param("4i6p", "A", "B", id="4i6p-A-vs-B"),
]


@pytest.mark.parametrize(("pdb_id", "pred_chain", "ref_chain"), BIOTITE_AGREEMENT_CASES)
def test_best_ca_rmsd_agrees_with_biotite_superimposition(
    pdb_id, pred_chain, ref_chain
):
    """Cross-check the repo primitive against an independent implementation."""
    structure = load_structure(pdb_id)
    array = structure.atom_array

    def _sorted_ca(chain):
        selected = array[
            (array.atom_name == "CA") & (~array.hetero) & (array.chain_id == chain)
        ]
        return selected[np.argsort(selected.res_id)]

    pred, ref = _sorted_ca(pred_chain), _sorted_ca(ref_chain)
    shared = np.intersect1d(pred.res_id, ref.res_id)
    pred = pred[np.isin(pred.res_id, shared)]
    ref = ref[np.isin(ref.res_id, shared)]
    fitted, _ = struc.superimpose(fixed=ref, mobile=pred)
    expected = float(struc.rmsd(ref, fitted))

    measured = best_ca_rmsd(
        structure, structure, ref_chains=(ref_chain,), pred_chains=(pred_chain,)
    ).rmsd
    assert measured == pytest.approx(expected, abs=1e-4)


def test_only_residues_present_in_both_are_scored():
    """4i6p chain B is missing two residues that chain A models; the pair still scores."""
    structure = load_structure("4i6p")
    positions = structure.ca_positions_by_chain
    shared = set(positions["A"]) & set(positions["B"])
    assert len(shared) == 82 < len(positions["A"])
    metrics = best_ca_rmsd(structure, structure, ref_chains=("B",), pred_chains=("A",))
    assert metrics.rmsd > 0.0


REJECTION_CASES = [
    pytest.param(
        "1hf9", ("A",), ("A", "B"), "Need equal chain counts", id="unequal-chain-counts"
    ),
    pytest.param("1ubq", ("A",), ("ZZ",), "has no chain", id="unknown-reference-chain"),
    pytest.param("1ubq", ("ZZ",), ("A",), "has no chain", id="unknown-predicted-chain"),
]


@pytest.mark.parametrize(
    ("pdb_id", "pred_chains", "ref_chains", "message"), REJECTION_CASES
)
def test_rejects_unusable_chain_selections(pdb_id, pred_chains, ref_chains, message):
    structure = load_structure(pdb_id)
    with pytest.raises(ValueError, match=message):
        best_ca_rmsd(
            structure, structure, ref_chains=ref_chains, pred_chains=pred_chains
        )


def test_rejects_more_chains_than_brute_force_allows():
    """5kc1 has 12 polymer chains; 12! assignments is not a search worth attempting."""
    structure = load_structure("5kc1")
    assert len(structure.ca_positions_by_chain) > MAX_PERMUTED_CHAINS
    with pytest.raises(ValueError, match="Refusing to brute-force"):
        best_ca_rmsd(
            structure, structure, ref_chains=tuple(structure.ca_positions_by_chain)
        )


def test_rejects_chains_with_no_residues_in_common():
    """5kc1 chains A and B share no residue ids, so there is nothing to superimpose."""
    structure = load_structure("5kc1")
    positions = structure.ca_positions_by_chain
    assert not set(positions["A"]) & set(positions["B"])
    with pytest.raises(ValueError, match="No residues in common"):
        best_ca_rmsd(structure, structure, ref_chains=("B",), pred_chains=("A",))


# ---------------------------------------------------------------------------
# CaAlignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("rotation", "translation"), RIGID_MOTION_CASES)
def test_alignment_transformation_maps_prediction_onto_reference(rotation, translation):
    """The returned transform is what carries a ligand into the reference frame."""
    reference = load_structure("1ubq")
    prediction = moved(reference, rotation=rotation, translation=translation)
    alignment = _best_ca_assignment(prediction, reference, ref_chains=("A",))

    predicted_coords = prediction.atom_array.coord.astype(float)
    restored = apply_transformation(
        positions=torch.from_numpy(predicted_coords),
        transformation=alignment.transformation,
    ).numpy()
    assert np.allclose(restored, reference.atom_array.coord, atol=1e-3)


# ---------------------------------------------------------------------------
# symmetry_mappings
# ---------------------------------------------------------------------------

#: Expected automorphism counts follow from the chemistry, not from running the code:
#: a sulfate's four equivalent oxygens give 4! and nitrate's three give 3!, tris permutes
#: its three hydroxymethyl arms, ethylene glycol and toluene each have a single 2-fold,
#: and mercaptoethanol is asymmetric.
SYMMETRY_CASES = [
    pytest.param("7l39", "E", 1, id="mercaptoethanol-asymmetric"),
    pytest.param("3u8v", "C", 1, id="single-atom-ion"),
    pytest.param("5kc1", "GB", 2, id="ethylene-glycol-2fold"),
    pytest.param("7l39", "D", 2, id="toluene-ring-flip"),
    pytest.param("5kc1", "N", 6, id="nitrate-3-factorial"),
    pytest.param("7l39", "B", 6, id="tris-three-arms"),
    pytest.param("4zey", "B", 24, id="sulfate-4-factorial"),
]


@pytest.mark.parametrize(("pdb_id", "chain", "expected"), SYMMETRY_CASES)
def test_symmetry_group_size(pdb_id, chain, expected):
    ligand = load_structure(pdb_id).heavy_atoms(chain)
    assert len(symmetry_mappings(ligand, ligand)) == expected


@pytest.mark.parametrize(("pdb_id", "chain", "expected"), SYMMETRY_CASES)
def test_symmetry_mappings_are_bijections(pdb_id, chain, expected):
    ligand = load_structure(pdb_id).heavy_atoms(chain)
    for mapping in symmetry_mappings(ligand, ligand):
        assert sorted(mapping) == list(range(len(ligand)))


def test_symmetry_mappings_are_empty_for_different_molecules():
    """Toluene is not a subgraph of mercaptoethanol, so nothing matches."""
    toluene = load_structure("7l39").heavy_atoms("D")
    mercaptoethanol = load_structure("7l39").heavy_atoms("E")
    assert symmetry_mappings(mercaptoethanol, toluene) == []


# ---------------------------------------------------------------------------
# ligand_pose_metrics
# ---------------------------------------------------------------------------


def test_identical_pose_scores_zero():
    structure = load_structure("7l39")
    metrics = ligand_pose_metrics(
        structure,
        structure,
        ref_chains=("A",),
        pred_chains=("A",),
        pred_ligand_chain="D",
        ref_ligand_chain="D",
    )
    assert metrics.rmsd == pytest.approx(0.0, abs=ZERO_RMSD_TOL)
    assert metrics.centroid_distance == pytest.approx(0.0, abs=ZERO_RMSD_TOL)
    assert metrics.n_atoms == 7
    assert metrics.n_symmetry_mappings == 2


LIGAND_OFFSET_CASES = [
    pytest.param(np.array([1.0, 0.0, 0.0]), 1.0, id="1A-along-x"),
    pytest.param(np.array([0.0, 2.0, 0.0]), 2.0, id="2A-along-y"),
    pytest.param(np.array([3.0, 4.0, 0.0]), 5.0, id="5A-diagonal"),
    pytest.param(np.array([-6.0, 0.0, 8.0]), 10.0, id="10A-out-of-pocket"),
]


@pytest.mark.parametrize(("offset", "expected"), LIGAND_OFFSET_CASES)
def test_displacing_only_the_ligand_shows_up_in_full(offset, expected):
    """The protein still superimposes exactly, so the whole offset lands on the ligand.

    Symmetry cannot absorb a rigid displacement, so both the pose RMSD and the
    mapping-free centroid distance must equal the displacement exactly.
    """
    reference = load_structure("7l39")
    prediction = with_chain_translated(reference, "D", offset)
    metrics = ligand_pose_metrics(
        prediction,
        reference,
        ref_chains=("A",),
        pred_chains=("A",),
        pred_ligand_chain="D",
        ref_ligand_chain="D",
    )
    assert metrics.rmsd == pytest.approx(expected, abs=1e-3)
    assert metrics.centroid_distance == pytest.approx(expected, abs=1e-3)


#: Toluene (MBN), 7l39 chain D, in file order: a methyl carbon ``C`` on a benzene ring
#: whose carbons run ``C1`` (the one bearing the methyl) round to ``C6``.
TOLUENE_ATOM_ORDER = ("C", "C1", "C2", "C3", "C4", "C5", "C6")

#: The same molecule with the ring read the other way round. ``C``, ``C1`` and the para
#: carbon ``C4`` lie on the mirror axis and stay put; the two pairs either side of it
#: swap — ``C2``<->``C6`` and ``C3``<->``C5``. This is toluene's only non-trivial
#: automorphism.
TOLUENE_RING_FLIP = ("C", "C1", "C6", "C5", "C4", "C3", "C2")

#: Smallest atom movement among the non-symmetry relabellings below (~0.74 Å). Scoring
#: any of them near zero would mean a genuine mismatch had been wrongly forgiven.
MIN_MISMATCH_RMSD = 0.5


def toluene_indices(atom_order: tuple[str, ...]) -> tuple[int, ...]:
    """An atom order as indices into ``TOLUENE_ATOM_ORDER``.

    That index form is what :func:`with_chain_relabelled` and :func:`symmetry_mappings`
    both speak; names are used above because a flip is legible and ``(0, 1, 6, 5, 4, 3,
    2)`` is not.
    """
    return tuple(TOLUENE_ATOM_ORDER.index(name) for name in atom_order)


def test_symmetry_mappings_finds_the_ring_flip():
    """Toluene has exactly two automorphisms: the identity and the mirror.

    Pins the atom order the constants above are written against, so they cannot quietly
    drift out of step with the committed cif.
    """
    ligand = load_structure("7l39").heavy_atoms("D")
    assert tuple(str(name) for name in ligand.atom_name) == TOLUENE_ATOM_ORDER

    found = {
        tuple(TOLUENE_ATOM_ORDER[index] for index in mapping)
        for mapping in symmetry_mappings(ligand, ligand)
    }
    assert found == {TOLUENE_ATOM_ORDER, TOLUENE_RING_FLIP}


RELABELLING_CASES = [
    pytest.param(TOLUENE_ATOM_ORDER, True, id="unchanged"),
    pytest.param(TOLUENE_RING_FLIP, True, id="ring-flip"),
    # Ortho and meta carbons are not interchangeable with each other, so swapping a
    # neighbouring pair is a real error however the molecule is read.
    pytest.param(
        ("C", "C1", "C3", "C2", "C4", "C5", "C6"), False, id="adjacent-carbons-swapped"
    ),
    # Benzene alone would be invariant under this; the methyl is what breaks it.
    pytest.param(
        ("C", "C2", "C3", "C4", "C5", "C6", "C1"), False, id="ring-rotated-under-methyl"
    ),
    pytest.param(
        ("C4", "C1", "C2", "C3", "C", "C5", "C6"), False, id="methyl-swapped-into-ring"
    ),
    # Reversed about the wrong axis: close enough to the true mirror that the symmetry
    # search cuts the error roughly in half, but it never reaches zero.
    pytest.param(
        ("C", "C6", "C5", "C4", "C3", "C2", "C1"), False, id="ring-reversed-wrong-axis"
    ),
]


@pytest.mark.parametrize(("atom_order", "is_symmetry"), RELABELLING_CASES)
def test_relabelling_is_forgiven_only_when_it_is_a_symmetry(atom_order, is_symmetry):
    """Reordering a ligand's atoms is free exactly when the reordering is a symmetry.

    Each case rewrites chain D's coordinates into ``atom_order`` — the pose is untouched,
    only which atom sits where changes. Toluene's mirror must score zero; anything else
    must not, or the metric would forgive genuinely misplaced atoms.
    """
    reference = load_structure("7l39")
    ligand = reference.heavy_atoms("D")
    prediction = with_chain_relabelled(reference, "D", toluene_indices(atom_order))

    naive = float(
        np.sqrt(
            ((prediction.heavy_atoms("D").coord - ligand.coord) ** 2).sum(-1).mean()
        )
    )
    if atom_order != TOLUENE_ATOM_ORDER:
        assert naive > MIN_MISMATCH_RMSD, (
            "the relabelling must actually move atoms for this to prove anything"
        )

    metrics = ligand_pose_metrics(
        prediction,
        reference,
        ref_chains=("A",),
        pred_chains=("A",),
        pred_ligand_chain="D",
        ref_ligand_chain="D",
    )

    if is_symmetry:
        assert metrics.rmsd == pytest.approx(0.0, abs=ZERO_RMSD_TOL)
    else:
        assert metrics.rmsd > MIN_MISMATCH_RMSD

    # Reordering atoms cannot move their centre of mass, so the mapping-free half of the
    # metric stays at zero throughout: every case above is the right pocket, and the RMSD
    # signal is purely about which atom went where.
    assert metrics.centroid_distance == pytest.approx(0.0, abs=ZERO_RMSD_TOL)
    # Searching symmetries can only ever improve the score, never worsen it.
    assert metrics.rmsd <= naive + ZERO_RMSD_TOL


def test_rejects_ligands_of_different_size():
    structure = load_structure("7l39")
    with pytest.raises(ValueError, match="Ligand atom count mismatch"):
        ligand_pose_metrics(
            structure,
            structure,
            ref_chains=("A",),
            pred_chains=("A",),
            pred_ligand_chain="B",
            ref_ligand_chain="D",
        )


def test_rejects_ligands_that_do_not_match_as_graphs():
    """Same atom count, unrelated topology: EDO (C2O2) against NO3 (NO3)."""
    structure = load_structure("5kc1")
    with pytest.raises(ValueError, match="No graph match"):
        ligand_pose_metrics(
            structure,
            structure,
            ref_chains=("A",),
            pred_chains=("A",),
            pred_ligand_chain="GB",
            ref_ligand_chain="N",
        )
