import networkx as nx
import numpy as np
import pytest
from biotite.structure import Atom, AtomArray, BondList, BondType

from openfold3.core.data.primitives.structure.cleanup import (
    filter_fully_atomized_bonds,
    prefilter_bonds,
)
from openfold3.core.data.resources.residues import MoleculeType
from tests.custom_assert_utils import assert_atomarray_equal
from tests.data_utils import create_atomarray_with_bondlist

# This helps conciseness to avoid Ruff line-break
PROTEIN = MoleculeType.PROTEIN
LIGAND = MoleculeType.LIGAND
DNA = MoleculeType.DNA


# -- Test Case Data ---

# Atom array that covers all the following properties:
# - intra-chain dative bond
# - inter-chain dative bond
# - intra-chain consecutive polymer bonds
# - intra-chain nonconsecutive polymer bonds
# - inter-chain polymer bonds
# - 2 long bonds (longer than 2.4 Å)

# atom_name is only set for user interpretability and irrelevant for the test

# fmt: off
all_atoms=[
    # Protein chain A
    Atom([0, 0, 0], chain_id="A", res_id=1, atom_name="backbone", molecule_type_id=PROTEIN), # noqa: E501
    Atom([0, 0, 0], chain_id="A", res_id=1, atom_name="sidechain", molecule_type_id=PROTEIN), # noqa: E501
    Atom([0, 0, 0], chain_id="A", res_id=2, atom_name="backbone", molecule_type_id=PROTEIN), # noqa: E501
    Atom([0, 0, 0], chain_id="A", res_id=2, atom_name="sidechain", molecule_type_id=PROTEIN), # noqa: E501
    Atom([0, 0, 0], chain_id="A", res_id=3, atom_name="backbone", molecule_type_id=PROTEIN), # noqa: E501
    Atom([0, 0, 0], chain_id="A", res_id=3, atom_name="sidechain", molecule_type_id=PROTEIN), # noqa: E501

    # Ligand chain B
    Atom([0, 0, 0], chain_id="B", res_id=1, atom_name="ligand", molecule_type_id=LIGAND), # noqa: E501
    Atom([0, 0, 2.41], chain_id="B", res_id=1, atom_name="ligand", molecule_type_id=LIGAND), # noqa: E501
    Atom([0, 0, 2.41], chain_id="B", res_id=1, atom_name="ligand", molecule_type_id=LIGAND), # noqa: E501

    # DNA chain C
    Atom([2.41, 0, 0], chain_id="C", res_id=1, atom_name="backbone", molecule_type_id=DNA), # noqa: E501
    Atom([2.41, 0, 0], chain_id="C", res_id=1, atom_name="sidechain", molecule_type_id=DNA), # noqa: E501
    Atom([0, 0, 0], chain_id="C", res_id=2, atom_name="backbone", molecule_type_id=DNA), # noqa: E501
    Atom([0, 0, 0], chain_id="C", res_id=2, atom_name="sidechain", molecule_type_id=DNA), # noqa: E501
    Atom([0, 0, 0], chain_id="C", res_id=3, atom_name="backbone", molecule_type_id=DNA), # noqa: E501
    Atom([0, 0, 0], chain_id="C", res_id=3, atom_name="sidechain", molecule_type_id=DNA), # noqa: E501
    Atom([0, 0, 0], chain_id="C", res_id=4, atom_name="backbone", molecule_type_id=DNA), # noqa: E501
    Atom([0, 0, 0], chain_id="C", res_id=4, atom_name="sidechain", molecule_type_id=DNA), # noqa: E501
    Atom([0, 0, 0], chain_id="C", res_id=5, atom_name="backbone", molecule_type_id=DNA), # noqa: E501
    Atom([0, 0, 0], chain_id="C", res_id=5, atom_name="sidechain", molecule_type_id=DNA), # noqa: E501
]
# fmt: on

# - References for special bonds -

# Chain B
intra_chain_long_dative = (6, 7, BondType.COORDINATION)

# Chain A to Chain B
inter_chain_dative = (3, 6, BondType.COORDINATION)

# Chain C
intra_chain_poly_link = (14, 18, BondType.SINGLE)

# Chain A to Chain C
inter_chain_poly_link = (5, 12, BondType.SINGLE)

# Chain C
long_bond_2 = (9, 11, BondType.SINGLE)

long_bond_set = {intra_chain_long_dative, long_bond_2}

# - Bond set -
bond_set = set(
    (
        # -- Intra-chain standard bonds --
        # Chain A
        (0, 1, BondType.SINGLE),
        (0, 2, BondType.SINGLE),
        (2, 3, BondType.SINGLE),
        (2, 4, BondType.SINGLE),
        (4, 5, BondType.SINGLE),
        # Chain B
        (7, 8, BondType.SINGLE),
        # Chain C
        (9, 10, BondType.SINGLE),
        long_bond_2,
        (11, 12, BondType.SINGLE),
        (11, 13, BondType.SINGLE),
        (13, 14, BondType.SINGLE),
        (13, 15, BondType.SINGLE),
        (15, 16, BondType.SINGLE),
        (15, 17, BondType.SINGLE),
        (17, 18, BondType.SINGLE),
        # -- Intra-chain dative bond --
        intra_chain_long_dative,
        # -- Inter-chain dative bond --
        inter_chain_dative,
        # -- Intra-chain nonconsecutive polymer crosslink --
        intra_chain_poly_link,
        # -- Inter-chain polymer crosslink --
        inter_chain_poly_link,
    )
)

# Sorting the bonds here is important for unit-test equivalence
atom_array_filter_bonds = create_atomarray_with_bondlist(
    all_atoms, np.array(sorted(bond_set))
)

# -- Test Cases --

# - Case 1: all active -
# remove_inter_chain_dative: True
# remove_inter_chain_poly_links: True
# remove_intra_chain_poly_links: True
# remove_longer_than: 2.4
case_all_true_bond_set = (
    bond_set
    - {inter_chain_dative, inter_chain_poly_link, intra_chain_poly_link}
    - long_bond_set
)
case_all_true_bond_array = np.array(sorted(case_all_true_bond_set))

# - Case 2: all inactive -
# remove_inter_chain_dative: False
# remove_inter_chain_poly_links: False
# remove_intra_chain_poly_links: False
# remove_longer_than: None
case_all_false_bond_set = bond_set
case_all_false_bond_array = np.array(sorted(case_all_false_bond_set))

# - Case 3: only inter-chain dative active -
# remove_inter_chain_dative: True
# remove_inter_chain_poly_links: False
# remove_intra_chain_poly_links: False
# remove_longer_than: None
case_only_inter_chain_dative = bond_set - {inter_chain_dative}
case_only_inter_chain_dative_bond_array = np.array(sorted(case_only_inter_chain_dative))

# - Case 4: only inter-chain poly links active -
# remove_inter_chain_dative: False
# remove_inter_chain_poly_links: True
# remove_intra_chain_poly_links: False
# remove_longer_than: None
case_only_inter_chain_poly_links = bond_set - {inter_chain_poly_link}
case_only_inter_chain_poly_links_bond_array = np.array(
    sorted(case_only_inter_chain_poly_links)
)

# - Case 5: only intra-chain poly links active -
# remove_inter_chain_dative: False
# remove_inter_chain_poly_links: False
# remove_intra_chain_poly_links: True
# remove_longer_than: None
case_only_intra_chain_poly_links = bond_set - {intra_chain_poly_link}
case_only_intra_chain_poly_links_bond_array = np.array(
    sorted(case_only_intra_chain_poly_links)
)

# - Case 6: only longer_than active -
# remove_inter_chain_dative: False
# remove_inter_chain_poly_links: False
# remove_intra_chain_poly_links: False
# remove_longer_than: 2.4
case_only_longer_than = bond_set - long_bond_set
case_only_longer_than_bond_array = np.array(sorted(case_only_longer_than))


@pytest.mark.parametrize(
    [
        "atom_array",
        "remove_inter_chain_dative",
        "remove_inter_chain_poly_links",
        "remove_intra_chain_poly_links",
        "remove_longer_than",
        "expected_bondlist",
    ],
    [
        # All True
        (
            atom_array_filter_bonds,
            True,
            True,
            True,
            2.4,
            case_all_true_bond_array,
        ),
        # All False
        (
            atom_array_filter_bonds,
            False,
            False,
            False,
            None,
            case_all_false_bond_array,
        ),
        # Only remove inter-chain dative
        (
            atom_array_filter_bonds,
            True,
            False,
            False,
            None,
            case_only_inter_chain_dative_bond_array,
        ),
        # Only remove inter-chain polymer links
        (
            atom_array_filter_bonds,
            False,
            True,
            False,
            None,
            case_only_inter_chain_poly_links_bond_array,
        ),
        # Only remove intra-chain polymer links
        (
            atom_array_filter_bonds,
            False,
            False,
            True,
            None,
            case_only_intra_chain_poly_links_bond_array,
        ),
        # Only remove longer than 2.4
        (
            atom_array_filter_bonds,
            False,
            False,
            False,
            2.4,
            case_only_longer_than_bond_array,
        ),
    ],
    ids=[
        "all_true",
        "all_false",
        "only_rm_inter_chain_dative",
        "only_rm_inter_chain_poly_links",
        "only_rm_intra_chain_poly_links",
        "only_rm_longer_than_2.4",
    ],
)
def test_prefilter_bonds(
    atom_array: AtomArray,
    remove_inter_chain_dative: bool,
    remove_inter_chain_poly_links: bool,
    remove_intra_chain_poly_links: bool,
    remove_longer_than: float | None,
    expected_bondlist: BondList | np.ndarray,
):
    """Tests whether the bond prefiltering works as expected."""

    # Create a copy of the atom array to avoid modifying the original
    atom_array_expected = atom_array.copy()

    # Set the expected bond list
    atom_array_expected.bonds = BondList(len(atom_array_expected), expected_bondlist)

    atom_array_filtered = prefilter_bonds(
        atom_array=atom_array,
        remove_inter_chain_dative=remove_inter_chain_dative,
        remove_inter_chain_poly_links=remove_inter_chain_poly_links,
        remove_intra_chain_poly_links=remove_intra_chain_poly_links,
        remove_longer_than=remove_longer_than,
    )

    assert_atomarray_equal(
        atom_array_filtered,
        atom_array_expected,
    )


# -- Test Case Data ---

# - Case 1: no atomized atoms at all -
atom_array_no_atomized = create_atomarray_with_bondlist(
    atoms=[
        Atom([0, 0, 0], chain_id="A", res_id=1, is_atomized=False),
        Atom([0, 0, 0], chain_id="A", res_id=1, is_atomized=False),
        Atom([0, 0, 0], chain_id="A", res_id=1, is_atomized=False),
        Atom([0, 0, 0], chain_id="A", res_id=2, is_atomized=False),
        Atom([0, 0, 0], chain_id="A", res_id=2, is_atomized=False),
        Atom([0, 0, 0], chain_id="A", res_id=3, is_atomized=False),
        Atom([0, 0, 0], chain_id="A", res_id=3, is_atomized=False),
    ],
    bondlist=np.array(
        sorted(
            [
                (0, 1, BondType.SINGLE),
                (1, 2, BondType.SINGLE),
                (2, 3, BondType.SINGLE),
                (3, 4, BondType.SINGLE),
                (4, 5, BondType.SINGLE),
                (5, 6, BondType.SINGLE),
            ]
        )
    ),
)
expected_bondlist_no_atomized = np.array([], dtype=int).reshape(0, 3)

# - Case 2: non-atomized residue binding to atomized ligand -
atom_array_some_atomized = create_atomarray_with_bondlist(
    atoms=[
        Atom([0, 0, 0], chain_id="A", res_id=1, is_atomized=False),
        Atom([0, 0, 0], chain_id="A", res_id=1, is_atomized=False),
        Atom([0, 0, 0], chain_id="B", res_id=1, is_atomized=True),
        Atom([0, 0, 0], chain_id="B", res_id=1, is_atomized=True),
        Atom([0, 0, 0], chain_id="B", res_id=1, is_atomized=True),
    ],
    bondlist=np.array(
        sorted(
            [
                (0, 1, BondType.SINGLE),
                (1, 2, BondType.SINGLE),
                (2, 3, BondType.SINGLE),
                (3, 4, BondType.SINGLE),
            ]
        )
    ),
)
expected_bondlist_some_atomized = np.array(
    [
        (2, 3, BondType.SINGLE),
        (3, 4, BondType.SINGLE),
    ],
    dtype=int,
)

# - Case 3: all atoms are atomized -

# make bondlist fully connected
G = nx.complete_graph(8)
edges = np.array(G.edges())
bondlist_all_atomized = np.hstack(
    (edges, np.full((edges.shape[0], 1), BondType.SINGLE, dtype=int))
)

atom_array_all_atomized = create_atomarray_with_bondlist(
    atoms=[
        Atom([0, 0, 0], chain_id="A", res_id=1, is_atomized=True),
        Atom([0, 0, 0], chain_id="A", res_id=1, is_atomized=True),
        Atom([0, 0, 0], chain_id="A", res_id=1, is_atomized=True),
        Atom([0, 0, 0], chain_id="A", res_id=2, is_atomized=True),
        Atom([0, 0, 0], chain_id="A", res_id=2, is_atomized=True),
        Atom([0, 0, 0], chain_id="B", res_id=1, is_atomized=True),
        Atom([0, 0, 0], chain_id="B", res_id=1, is_atomized=True),
        Atom([0, 0, 0], chain_id="B", res_id=2, is_atomized=True),
    ],
    bondlist=bondlist_all_atomized,
)


@pytest.mark.parametrize(
    ["atom_array", "expected_bondlist"],
    [
        (atom_array_no_atomized, expected_bondlist_no_atomized),
        (atom_array_some_atomized, expected_bondlist_some_atomized),
        (atom_array_all_atomized, bondlist_all_atomized),
    ],
    ids=["no_atomized_bonds", "some_atomized_bonds", "all_atomized_bonds"],
)
def test_filter_fully_atomized_bonds(
    atom_array: AtomArray, expected_bondlist: BondList | np.ndarray
):
    """Tests whether filtering of fully atomized bonds works as expected."""

    # Create a copy of the atom array to avoid modifying the original
    atom_array_expected = atom_array.copy()

    # Set the expected bond list
    atom_array_expected.bonds = BondList(len(atom_array_expected), expected_bondlist)

    # Apply the filter
    filtered_bondlist = filter_fully_atomized_bonds(
        atom_array=atom_array,
    )

    # Assert that the filtered bond list matches the expected bond list
    assert_atomarray_equal(
        filtered_bondlist,
        atom_array_expected,
    )
