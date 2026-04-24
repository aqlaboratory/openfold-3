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

"""Regression tests for the data-pipeline optimizations documented in
`dev_scripts/data_pipeline_optimizations.md`.

Each test re-implements the pre-optimization (legacy) version of a function
inline and asserts that the new implementation produces equivalent output on
a variety of inputs. The legacy implementations are intentionally duplicated
here rather than referenced from the main source tree; they exist only as
reference oracles for the refactors and should not be used at runtime.
"""

import gc
import sys
import time
import tracemalloc
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from biotite.structure import AtomArray, BondList

from openfold3.core.data.primitives.featurization.msa import vstack_pad_msa_arrays
from openfold3.core.data.primitives.permutation.mol_labels import (
    chain_connected_molecule_iter,
)
from openfold3.core.data.primitives.sequence.msa import (
    MsaArray,
    cap_seqs_per_species,
)
from openfold3.core.data.resources.patches import get_molecule_indices
from openfold3.core.data.resources.residues import (
    MOLECULE_TYPE_TO_RESIDUES_1,
    MOLECULE_TYPE_TO_RESIDUES_POS_MAP,
    MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_1,
    STANDARD_RESIDUES_WITH_GAP_1,
    MoleculeType,
    map_str_array_to_idx_array,
)

# ---------------------------------------------------------------------------
# §1. cap_seqs_per_species
# ---------------------------------------------------------------------------


def _legacy_cap_seqs_per_species(msa: MsaArray, max_seq_per_species: int) -> MsaArray:
    """Pre-optimization implementation (groupby loop + multi_concatenate)."""
    metadata = msa.metadata
    if "species_id" not in metadata.columns:
        raise ValueError(
            "MsaArray metadata must contain species_id column to cap sequences per "
            "species."
        )

    groups = []
    for _, block in metadata.groupby("species_id", sort=False):
        block_length = min(len(block), max_seq_per_species)
        rows = block.index[:block_length]
        groups.append(
            MsaArray(
                msa=msa.msa[rows + 1],
                deletion_matrix=msa.deletion_matrix[rows + 1],
                metadata=metadata.iloc[rows].reset_index(drop=True),
            )
        )
    if not groups:
        raise ValueError("No sequences found in MsaArray to cap per species.")
    filtered_msa_array = MsaArray.multi_concatenate(groups, axis=0)
    return MsaArray.multi_concatenate(
        [
            MsaArray(
                msa=msa.msa[0:1, :],
                deletion_matrix=msa.deletion_matrix[0:1, :],
                metadata=pd.DataFrame(),
            ),
            filtered_msa_array,
        ],
        axis=0,
    )


def _build_species_msa(
    species_ids: list[int], seq_len: int = 4, seed: int = 0
) -> MsaArray:
    """Build an MsaArray with query row + one row per species_id entry."""
    rng = np.random.default_rng(seed)
    n_rows = len(species_ids) + 1  # +1 query
    alphabet = np.array(list("ACGT-"))
    msa = alphabet[rng.integers(0, len(alphabet), size=(n_rows, seq_len))]
    msa[0, :] = "A"  # deterministic query
    deletion = rng.integers(0, 3, size=(n_rows, seq_len))
    metadata = pd.DataFrame({"species_id": species_ids})
    return MsaArray(msa=msa, deletion_matrix=deletion, metadata=metadata)


class TestCapSeqsPerSpecies:
    @pytest.mark.parametrize(
        "species_ids, cap",
        [
            ([1, 1, 2, 2, 3], 2),  # each species has <= cap
            ([1, 1, 1, 2, 2, 2, 3], 2),  # cap truncates within species
            ([1, 2, 3, 4, 5], 10),  # cap >> group sizes
            ([1, 1, 1, 1, 1], 1),  # single species
            ([5, 3, 5, 3, 2, 5], 2),  # non-contiguous species
        ],
    )
    def test_matches_legacy(self, species_ids, cap):
        msa = _build_species_msa(species_ids)
        expected = _legacy_cap_seqs_per_species(msa, cap)
        actual = cap_seqs_per_species(msa, cap)

        # The new impl preserves original MSA row order; the legacy groupby
        # implementation reorders rows into species-blocks. Both keep the same
        # set of rows per species — downstream pairing uses species lookups,
        # not row position. Normalize by stable-sorting on species_id before
        # comparing (query row stays pinned at position 0).
        def _sorted_by_species(
            m: MsaArray,
        ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
            order = np.argsort(m.metadata["species_id"].to_numpy(), kind="stable")
            # +1 for query offset on msa/deletion rows; metadata has no query entry
            msa_rows = np.concatenate(([0], order + 1))
            return (
                m.msa[msa_rows],
                m.deletion_matrix[msa_rows],
                m.metadata.iloc[order].reset_index(drop=True),
            )

        a_msa, a_del, a_meta = _sorted_by_species(actual)
        e_msa, e_del, e_meta = _sorted_by_species(expected)
        np.testing.assert_array_equal(a_msa, e_msa)
        np.testing.assert_array_equal(a_del, e_del)
        pd.testing.assert_frame_equal(a_meta, e_meta)


# ---------------------------------------------------------------------------
# §2. create_main — dedup block (hash-based vs sort-based)
# ---------------------------------------------------------------------------


def _legacy_unique_idx(main_msa_redundant: np.ndarray) -> np.ndarray:
    """Pre-optimization dedup path: np.unique(return_index=True) + sort."""
    main_view = main_msa_redundant.view(
        np.dtype(
            (np.void, main_msa_redundant.dtype.itemsize * main_msa_redundant.shape[1])
        )
    )
    _, unique_idx = np.unique(main_view, return_index=True)
    unique_idx.sort()
    return unique_idx


def _new_unique_idx(main_msa_redundant: np.ndarray) -> np.ndarray:
    """Post-optimization dedup path (mirrors create_main)."""
    main_view = main_msa_redundant.view(
        np.dtype(
            (np.void, main_msa_redundant.dtype.itemsize * main_msa_redundant.shape[1])
        )
    ).ravel()
    return pd.Series(main_view).drop_duplicates().index.to_numpy()


class TestCreateMainDedup:
    @pytest.mark.parametrize(
        "rows",
        [
            [[0, 1, 2], [0, 1, 2], [3, 4, 5], [0, 1, 2], [6, 7, 8]],  # dupes
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],  # all unique
            [[1, 1, 1]] * 5,  # all identical
            [[i, i + 1, i + 2] for i in range(20)],  # big unique
        ],
    )
    def test_dedup_matches_legacy(self, rows):
        arr = np.array(rows, dtype=np.int32)
        legacy = _legacy_unique_idx(arr)
        new = _new_unique_idx(arr)
        np.testing.assert_array_equal(legacy, new)
        # Verify the resulting dedup'd arrays are identical
        np.testing.assert_array_equal(arr[legacy], arr[new])


# ---------------------------------------------------------------------------
# §2b. create_main — paired-MSA membership block (isin shape-hygiene cleanup)
# ---------------------------------------------------------------------------


def _legacy_isin_mask(main: np.ndarray, paired: np.ndarray) -> np.ndarray:
    """Pre-optimization: 2-D void view + np.squeeze on the trailing axis."""
    main_view = main.view(np.dtype((np.void, main.dtype.itemsize * main.shape[1])))
    paired_view = paired.view(
        np.dtype((np.void, paired.dtype.itemsize * paired.shape[1]))
    )
    return np.squeeze(~np.isin(main_view, paired_view), axis=-1)


def _new_isin_mask(main: np.ndarray, paired: np.ndarray) -> np.ndarray:
    """Post-optimization: ravel'd 1-D void views, no squeeze needed."""
    main_view = main.view(
        np.dtype((np.void, main.dtype.itemsize * main.shape[1]))
    ).ravel()
    paired_view = paired.view(
        np.dtype((np.void, paired.dtype.itemsize * paired.shape[1]))
    ).ravel()
    return ~np.isin(main_view, paired_view)


class TestCreateMainIsin:
    @pytest.mark.parametrize(
        "main_rows, paired_rows",
        [
            # Some overlap
            ([[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[3, 4, 5], [9, 9, 9]]),
            # No overlap
            ([[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]),
            # Full overlap
            ([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]),
            # Empty paired
            ([[1, 2, 3], [4, 5, 6]], [[0, 0, 0]][:0]),  # zero rows, same cols
            # Single main row, overlaps
            ([[7, 8, 9]], [[7, 8, 9]]),
        ],
    )
    def test_isin_mask_matches_legacy(self, main_rows, paired_rows):
        main = np.array(main_rows, dtype=np.int32).reshape(-1, 3)
        paired = np.array(paired_rows, dtype=np.int32).reshape(-1, 3)
        legacy = _legacy_isin_mask(main, paired)
        new = _new_isin_mask(main, paired)
        np.testing.assert_array_equal(legacy, new)
        # Both should select the same subset of main rows.
        np.testing.assert_array_equal(main[legacy], main[new])


# ---------------------------------------------------------------------------
# §3. map_str_array_to_idx_array
# ---------------------------------------------------------------------------


def _legacy_map_str_array_to_idx_array(
    msa_array: np.ndarray, molecule_type: MoleculeType
) -> np.ndarray:
    """Pre-optimization implementation (bool-mask scan per residue)."""
    if isinstance(molecule_type, str):
        molecule_type = MoleculeType(MoleculeType[molecule_type].value)
    msa_idx = np.full(
        msa_array.shape,
        np.where(np.array(STANDARD_RESIDUES_WITH_GAP_1) == "-")[0].item(),
        dtype=np.int64,
    )
    for residue, idx in MOLECULE_TYPE_TO_RESIDUES_POS_MAP[molecule_type].items():
        if residue != "-":
            msa_idx[msa_array == residue] = idx
    msa_idx[~np.isin(msa_array, MOLECULE_TYPE_TO_RESIDUES_1[molecule_type])] = (
        MOLECULE_TYPE_TO_RESIDUES_POS_MAP[molecule_type][
            MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_1[molecule_type]
        ]
    )
    return msa_idx


class TestMapStrArrayToIdxArray:
    @pytest.mark.parametrize(
        "molecule_type",
        [
            MoleculeType.PROTEIN,
            MoleculeType.RNA,
            # DNA omitted: its residues_pos_map is keyed by 2-letter codes
            # ("DA", "DG", …) while MOLECULE_TYPE_TO_UNKNOWN_RESIDUES_1[DNA] is
            # "N", so both legacy and new impls raise KeyError on the unknown
            # lookup. Pre-existing issue; function is only invoked on <U1
            # arrays (PROTEIN/RNA) in practice.
            MoleculeType.LIGAND,
        ],
    )
    @pytest.mark.parametrize("shape", [(1, 10), (5, 20), (32, 7)])
    def test_matches_legacy(self, molecule_type, shape):
        rng = np.random.default_rng(42)
        # Alphabet includes all molecule-type residues plus "-" plus a few
        # unknown ASCII chars to exercise the fallback path.
        alphabet = list(MOLECULE_TYPE_TO_RESIDUES_1[molecule_type]) + ["-", "Z", "O"]
        arr = np.array(
            [
                [alphabet[i] for i in rng.integers(0, len(alphabet), size=shape[1])]
                for _ in range(shape[0])
            ],
            dtype="<U1",
        )
        expected = _legacy_map_str_array_to_idx_array(arr, molecule_type)
        actual = map_str_array_to_idx_array(arr, molecule_type)

        # Legacy is int64, new is int16 — use array_equal which ignores dtype.
        np.testing.assert_array_equal(actual, expected)

    def test_handles_non_contiguous_input(self):
        """Non-contig slices should hit the ascontiguousarray guard."""
        rng = np.random.default_rng(0)
        alphabet = list(MOLECULE_TYPE_TO_RESIDUES_1[MoleculeType.PROTEIN]) + ["-"]
        arr = np.array(
            [
                [alphabet[i] for i in rng.integers(0, len(alphabet), size=20)]
                for _ in range(8)
            ],
            dtype="<U1",
        )
        # Make a non-contiguous slice by column-striding
        sliced = arr[:, ::2]
        assert not sliced.flags.c_contiguous

        expected = _legacy_map_str_array_to_idx_array(sliced, MoleculeType.PROTEIN)
        actual = map_str_array_to_idx_array(sliced, MoleculeType.PROTEIN)
        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# §4. get_molecule_indices
# ---------------------------------------------------------------------------


def _legacy_get_molecule_indices(atom_array: AtomArray) -> list[np.ndarray]:
    """Pre-optimization implementation (networkx)."""
    bonds = atom_array.bonds
    g = bonds.as_graph()

    all_atoms = np.arange(len(atom_array))
    atoms_in_graph = np.unique(list(g.nodes))
    singleton_components = np.setdiff1d(all_atoms, atoms_in_graph)
    singleton_components_formatted = [np.array([atom]) for atom in singleton_components]

    connected = nx.connected_components(g)
    connected_formatted = [np.sort(list(c)) for c in connected]

    all_components = singleton_components_formatted + connected_formatted
    return sorted(all_components, key=lambda x: x[0])


def _make_atom_array(
    n_atoms: int, chain_ids: list[str], bonds: list[tuple]
) -> AtomArray:
    """Build a minimal AtomArray with the given chain IDs and bonds."""
    assert len(chain_ids) == n_atoms
    arr = AtomArray(n_atoms)
    arr.coord = np.zeros((n_atoms, 3))
    arr.chain_id[:] = chain_ids
    bond_arr = (
        np.array(bonds, dtype=np.uint32) if bonds else np.empty((0, 2), np.uint32)
    )
    arr.bonds = BondList(n_atoms, bond_arr)
    return arr


def _components_equal(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    if len(a) != len(b):
        return False
    return all(np.array_equal(x, y) for x, y in zip(a, b))


class TestGetMoleculeIndices:
    @pytest.mark.parametrize(
        "n_atoms, bonds, description",
        [
            (6, [(0, 1), (1, 2), (3, 4)], "two components + one singleton"),
            (5, [], "all singletons"),
            (4, [(0, 1), (1, 2), (2, 3)], "one linear chain"),
            (8, [(0, 1), (2, 3), (3, 4), (5, 6), (6, 7)], "multiple components"),
            (3, [(0, 2)], "non-adjacent bond"),
        ],
    )
    def test_matches_legacy(self, n_atoms, bonds, description):
        atom_array = _make_atom_array(n_atoms, ["A"] * n_atoms, bonds)
        expected = _legacy_get_molecule_indices(atom_array)
        actual = get_molecule_indices(atom_array)
        assert _components_equal(actual, expected), (
            f"mismatch for {description}: expected={expected}, actual={actual}"
        )


# ---------------------------------------------------------------------------
# §5. chain_connected_molecule_iter
# ---------------------------------------------------------------------------


def _legacy_chain_connected_molecule_iter(atom_array):
    """Pre-optimization implementation (BondList merge)."""
    from biotite.structure import get_chain_starts

    atom_array_pseudo_copy = atom_array[:]
    n_atoms = len(atom_array)

    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=True)
    root_atoms = chain_starts[:-1]
    root_atom_repeated = np.repeat(root_atoms, np.diff(chain_starts))
    root_atom_bond_pairs = np.column_stack((root_atom_repeated, np.arange(n_atoms)))

    chain_connected_bond_list = BondList(n_atoms, bonds=root_atom_bond_pairs)
    atom_array_pseudo_copy.bonds = atom_array_pseudo_copy.bonds.merge(
        chain_connected_bond_list
    )

    # Uses the legacy get_molecule_indices via the new public one — semantics
    # identical, see TestGetMoleculeIndices for equivalence.
    for molecule_indices in _legacy_get_molecule_indices(atom_array_pseudo_copy):
        yield atom_array[molecule_indices]


class TestChainConnectedMoleculeIter:
    @pytest.mark.parametrize(
        "n_atoms, chain_ids, bonds, description",
        [
            # Three chains, no inter-chain bonds → three components
            (
                9,
                ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
                [(0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8)],
                "three fully bonded chains",
            ),
            # Chain with a disconnected atom — chain-star should still glue it
            (
                6,
                ["A", "A", "A", "B", "B", "B"],
                [(0, 1), (3, 4)],  # atoms 2 and 5 dangling
                "chains with internal disconnect",
            ),
            # Inter-chain bond merges two chains into one component
            (
                6,
                ["A", "A", "A", "B", "B", "B"],
                [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
                "inter-chain bond merges chains",
            ),
            # Single chain
            (4, ["A"] * 4, [(0, 1), (2, 3)], "single chain, internal disconnect"),
        ],
    )
    def test_matches_legacy(self, n_atoms, chain_ids, bonds, description):
        atom_array = _make_atom_array(n_atoms, chain_ids, bonds)

        legacy = [
            mol.coord.shape[0]
            for mol in _legacy_chain_connected_molecule_iter(atom_array)
        ]
        new = [mol.coord.shape[0] for mol in chain_connected_molecule_iter(atom_array)]
        assert legacy == new, f"component sizes differ for {description}"

        # Rebuild atom arrays (generators exhausted above) and compare atom-ID contents
        atom_array2 = _make_atom_array(n_atoms, chain_ids, bonds)
        atom_array3 = _make_atom_array(n_atoms, chain_ids, bonds)
        legacy_mols = list(_legacy_chain_connected_molecule_iter(atom_array2))
        new_mols = list(chain_connected_molecule_iter(atom_array3))

        # Order and content of each component must match
        for lm, nm in zip(legacy_mols, new_mols, strict=True):
            np.testing.assert_array_equal(lm.chain_id, nm.chain_id)
            np.testing.assert_array_equal(lm.coord, nm.coord)


# ---------------------------------------------------------------------------
# §6. vstack_pad_msa_arrays
# ---------------------------------------------------------------------------


def _legacy_vstack_pad_msa_arrays(msa_array_collection, chain_id):
    """Pre-optimization implementation (sequential concatenate + pad)."""
    msa_array_vstack = msa_array_collection.chain_id_to_query_seq[chain_id]

    if msa_array_collection.row_counts.n_rows_paired_subsampled > 0:
        msa_array_vstack = msa_array_vstack.concatenate(
            msa_array_collection.chain_id_to_paired_msa[chain_id], axis=0
        )

    if len(msa_array_collection.row_counts.n_rows_main_subsampled) > 0:
        msa_array_vstack = msa_array_vstack.concatenate(
            msa_array_collection.chain_id_to_main_msa[chain_id], axis=0
        )

    msa_array_vstack, msa_array_vstack_mask = msa_array_vstack.pad(
        target_length=msa_array_collection.row_counts.n_rows_total, axis=0
    )
    return msa_array_vstack, msa_array_vstack_mask


def _build_msa_array(n_rows: int, seq_len: int, seed: int) -> MsaArray:
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT-"))
    msa = alphabet[rng.integers(0, len(alphabet), size=(n_rows, seq_len))]
    deletion = rng.integers(0, 3, size=(n_rows, seq_len))
    return MsaArray(msa=msa, deletion_matrix=deletion, metadata=pd.DataFrame())


def _build_collection(
    chain_id: str,
    seq_len: int,
    n_paired: int,
    n_main: int,
    target_length: int,
):
    query = _build_msa_array(1, seq_len, seed=0)
    paired = _build_msa_array(n_paired, seq_len, seed=1) if n_paired > 0 else None
    main = _build_msa_array(n_main, seq_len, seed=2) if n_main > 0 else None

    row_counts = SimpleNamespace(
        n_rows_total=target_length,
        n_rows_paired_subsampled=n_paired,
        n_rows_main_subsampled={"rep": n_main} if n_main > 0 else {},
    )
    collection = SimpleNamespace(
        chain_id_to_query_seq={chain_id: query},
        chain_id_to_paired_msa={chain_id: paired} if paired is not None else {},
        chain_id_to_main_msa={chain_id: main} if main is not None else {},
        row_counts=row_counts,
    )
    return collection


class TestVstackPadMsaArrays:
    @pytest.mark.parametrize(
        "n_paired, n_main, target_length, description",
        [
            (3, 5, 15, "query + paired + main, pads"),
            (0, 5, 10, "no paired, query + main, pads"),
            (3, 0, 8, "query + paired only, pads"),
            (0, 0, 5, "query only, pads"),
            (2, 3, 6, "exact fit, no padding"),
            (10, 10, 21, "larger case, pads"),
        ],
    )
    def test_matches_legacy(self, n_paired, n_main, target_length, description):
        chain_id = "A"
        seq_len = 7
        collection = _build_collection(
            chain_id, seq_len, n_paired, n_main, target_length
        )
        # Rebuild a second identical collection since the legacy call may
        # mutate (it doesn't, but be defensive).
        collection2 = _build_collection(
            chain_id, seq_len, n_paired, n_main, target_length
        )

        legacy_msa, legacy_mask = _legacy_vstack_pad_msa_arrays(collection, chain_id)
        new_msa, new_mask = vstack_pad_msa_arrays(collection2, chain_id)

        np.testing.assert_array_equal(new_msa.msa, legacy_msa.msa)
        np.testing.assert_array_equal(
            new_msa.deletion_matrix, legacy_msa.deletion_matrix
        )
        np.testing.assert_array_equal(new_mask, legacy_mask)


# ---------------------------------------------------------------------------
# Performance + memory benchmarks
#
# Run with `pytest -s -k TestPerformance …` to see printed results. Assertions
# enforce that the new implementation is at least as memory-light and at least
# as fast as the legacy baseline (with modest timing slack for CI jitter).
# ---------------------------------------------------------------------------


def _measure_perf(fn, args_factory, repeats: int = 5) -> tuple[float, int]:
    """Measure min runtime (s) and peak tracemalloc allocation (bytes).

    Args:
        fn: Callable to benchmark.
        args_factory: Zero-arg callable returning a fresh `(args, kwargs)` tuple
            for each invocation (so inputs aren't reused/warmed across runs).
        repeats: Timing repeats; min is reported.

    Notes:
        tracemalloc tracks the Python heap. Numpy buffer allocations may or may
        not show up depending on numpy's allocator config, but both legacy and
        new impls use numpy the same way, so the *delta* remains meaningful.
        If any caller relies on accurate absolute numbers, switch to psutil RSS.
    """
    # Warmup (primes imports, JIT caches in pandas, LUT cache, etc.)
    args, kwargs = args_factory()
    fn(*args, **kwargs)

    # Memory pass
    gc.collect()
    tracemalloc.start()
    args, kwargs = args_factory()
    tracemalloc.reset_peak()
    fn(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Timing pass — separate so tracemalloc overhead doesn't distort timings
    times = []
    for _ in range(repeats):
        args, kwargs = args_factory()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)

    return min(times), peak


def _report(name: str, t_old: float, m_old: int, t_new: float, m_new: int) -> None:
    # Bypass pytest capture so output appears even without `-s`
    line = (
        f"\n[bench] {name:40s} "
        f"time: {t_old * 1000:7.2f}ms → {t_new * 1000:7.2f}ms "
        f"({t_old / max(t_new, 1e-9):5.1f}x)  "
        f"peak_mem: {m_old / 1024:8.1f}KB → {m_new / 1024:8.1f}KB "
        f"({m_old / max(m_new, 1):5.1f}x)"
    )
    sys.__stdout__.write(line)
    sys.__stdout__.flush()


# Slack bounds. Timing slack absorbs CI shared-runner jitter. Memory slack
# absorbs small tracemalloc artifacts (view-object wrappers, interned bytes)
# on refactors that are memory-neutral rather than memory-wins (e.g. the isin
# shape-hygiene cleanup). Assertions catch gross regressions; the printed
# ratios convey the real win magnitude.
_TIME_SLACK = 1.25
_MEM_SLACK = 1.05


class TestPerformanceAndMemory:
    def test_cap_seqs_per_species(self):
        rng = np.random.default_rng(0)
        n_species, per_species, cap, seq_len = 200, 30, 5, 64
        species_ids = np.repeat(np.arange(n_species), per_species).tolist()
        rng.shuffle(species_ids)

        def factory():
            alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY-"))
            msa = alphabet[
                rng.integers(0, len(alphabet), size=(len(species_ids) + 1, seq_len))
            ]
            deletion = rng.integers(0, 3, size=(len(species_ids) + 1, seq_len))
            metadata = pd.DataFrame({"species_id": species_ids})
            m = MsaArray(msa=msa, deletion_matrix=deletion, metadata=metadata)
            return (m, cap), {}

        t_old, m_old = _measure_perf(_legacy_cap_seqs_per_species, factory)
        t_new, m_new = _measure_perf(cap_seqs_per_species, factory)
        _report("cap_seqs_per_species", t_old, m_old, t_new, m_new)
        assert t_new <= t_old * _TIME_SLACK
        assert m_new <= m_old * _MEM_SLACK

    def test_create_main_dedup(self):
        rng = np.random.default_rng(1)
        n_rows, n_cols = 4000, 96

        def factory():
            # ~30% duplicates
            base = rng.integers(0, 5, size=(n_rows // 2, n_cols), dtype=np.int32)
            dup_idx = rng.integers(0, base.shape[0], size=n_rows - base.shape[0])
            arr = np.concatenate([base, base[dup_idx]], axis=0)
            rng.shuffle(arr)
            return (arr,), {}

        t_old, m_old = _measure_perf(_legacy_unique_idx, factory)
        t_new, m_new = _measure_perf(_new_unique_idx, factory)
        _report("create_main.dedup", t_old, m_old, t_new, m_new)
        assert t_new <= t_old * _TIME_SLACK
        assert m_new <= m_old * _MEM_SLACK

    def test_create_main_isin(self):
        rng = np.random.default_rng(2)
        n_main, n_paired, n_cols = 4000, 500, 96

        def factory():
            main = rng.integers(0, 5, size=(n_main, n_cols), dtype=np.int32)
            paired = rng.integers(0, 5, size=(n_paired, n_cols), dtype=np.int32)
            return (main, paired), {}

        t_old, m_old = _measure_perf(_legacy_isin_mask, factory)
        t_new, m_new = _measure_perf(_new_isin_mask, factory)
        _report("create_main.isin", t_old, m_old, t_new, m_new)
        assert t_new <= t_old * _TIME_SLACK
        assert m_new <= m_old * _MEM_SLACK

    def test_map_str_array_to_idx_array(self):
        rng = np.random.default_rng(3)
        n_rows, n_cols = 5000, 500
        alphabet = list(MOLECULE_TYPE_TO_RESIDUES_1[MoleculeType.PROTEIN]) + ["-"]
        alphabet_arr = np.array(alphabet, dtype="<U1")

        def factory():
            arr = alphabet_arr[
                rng.integers(0, len(alphabet_arr), size=(n_rows, n_cols))
            ]
            return (arr, MoleculeType.PROTEIN), {}

        t_old, m_old = _measure_perf(_legacy_map_str_array_to_idx_array, factory)
        t_new, m_new = _measure_perf(map_str_array_to_idx_array, factory)
        _report("map_str_array_to_idx_array", t_old, m_old, t_new, m_new)
        assert t_new <= t_old * _TIME_SLACK
        assert m_new <= m_old * _MEM_SLACK

    def test_get_molecule_indices(self):
        rng = np.random.default_rng(4)
        n_atoms = 2000
        # ~1500 edges split across several components
        n_edges = 1500
        src = rng.integers(0, n_atoms, size=n_edges, dtype=np.uint32)
        dst = (src + rng.integers(1, 5, size=n_edges, dtype=np.uint32)) % n_atoms
        bond_pairs = np.column_stack((src, dst))

        def factory():
            arr = AtomArray(n_atoms)
            arr.coord = np.zeros((n_atoms, 3))
            arr.chain_id[:] = "A"
            arr.bonds = BondList(n_atoms, bond_pairs.copy())
            return (arr,), {}

        t_old, m_old = _measure_perf(_legacy_get_molecule_indices, factory)
        t_new, m_new = _measure_perf(get_molecule_indices, factory)
        _report("get_molecule_indices", t_old, m_old, t_new, m_new)
        assert t_new <= t_old * _TIME_SLACK
        assert m_new <= m_old * _MEM_SLACK

    def test_chain_connected_molecule_iter(self):
        rng = np.random.default_rng(5)
        n_atoms = 2000
        n_chains = 10
        chain_ids = np.repeat(
            np.array([f"C{i}" for i in range(n_chains)]), n_atoms // n_chains
        )
        # Pad to n_atoms if not divisible
        if len(chain_ids) < n_atoms:
            chain_ids = np.concatenate(
                [chain_ids, np.full(n_atoms - len(chain_ids), f"C{n_chains - 1}")]
            )
        n_edges = 1500
        src = rng.integers(0, n_atoms, size=n_edges, dtype=np.uint32)
        dst = (src + rng.integers(1, 10, size=n_edges, dtype=np.uint32)) % n_atoms
        bond_pairs = np.column_stack((src, dst))

        def factory():
            arr = AtomArray(n_atoms)
            arr.coord = np.zeros((n_atoms, 3))
            arr.chain_id[:] = chain_ids
            arr.bonds = BondList(n_atoms, bond_pairs.copy())
            return (arr,), {}

        def consume_legacy(arr):
            return [
                m.coord.shape[0] for m in _legacy_chain_connected_molecule_iter(arr)
            ]

        def consume_new(arr):
            return [m.coord.shape[0] for m in chain_connected_molecule_iter(arr)]

        t_old, m_old = _measure_perf(consume_legacy, factory)
        t_new, m_new = _measure_perf(consume_new, factory)
        _report("chain_connected_molecule_iter", t_old, m_old, t_new, m_new)
        assert t_new <= t_old * _TIME_SLACK
        assert m_new <= m_old * _MEM_SLACK

    def test_vstack_pad_msa_arrays(self):
        seq_len, n_paired, n_main, target = 256, 200, 800, 1500

        def factory():
            collection = _build_collection("A", seq_len, n_paired, n_main, target)
            return (collection, "A"), {}

        t_old, m_old = _measure_perf(_legacy_vstack_pad_msa_arrays, factory)
        t_new, m_new = _measure_perf(vstack_pad_msa_arrays, factory)
        _report("vstack_pad_msa_arrays", t_old, m_old, t_new, m_new)
        assert t_new <= t_old * _TIME_SLACK
        assert m_new <= m_old * _MEM_SLACK


# ---------------------------------------------------------------------------
# Scaling benchmarks (gated behind `-m slow`)
#
# For each optimization with a meaningful complexity story (O(N log N) → O(N),
# or eliminated per-item Python overhead), measure time-vs-size across 3–4
# input scales. A speedup ratio that grows with input size is the signature of
# a better asymptotic complexity; a flat ratio is a constant-factor win.
#
# Run with: `pytest -s -m slow -k TestScaling`
# ---------------------------------------------------------------------------


def _time(fn, args_factory, repeats: int = 3) -> float:
    args, kwargs = args_factory()
    fn(*args, **kwargs)  # warmup
    times = []
    for _ in range(repeats):
        args, kwargs = args_factory()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return min(times)


def _report_scaling(name: str, header: list[str], rows: list[tuple]) -> None:
    col_widths = [
        max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))
    ]
    border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    out = [f"\n[scale] {name}", border]
    out.append(
        "| " + " | ".join(f"{h:<{w}}" for h, w in zip(header, col_widths)) + " |"
    )
    out.append(border)
    for row in rows:
        out.append(
            "| " + " | ".join(f"{str(c):<{w}}" for c, w in zip(row, col_widths)) + " |"
        )
    out.append(border)
    sys.__stdout__.write("\n".join(out) + "\n")
    sys.__stdout__.flush()


@pytest.mark.slow
class TestScaling:
    def test_map_str_array_to_idx_array_scaling(self):
        """Legacy: O(N × alphabet_size) alphabet-mask loop. New: O(N) LUT gather."""
        sizes = [(100, 100), (1000, 500), (5000, 500)]
        alphabet = list(MOLECULE_TYPE_TO_RESIDUES_1[MoleculeType.PROTEIN]) + ["-"]
        alphabet_arr = np.array(alphabet, dtype="<U1")

        rows = []
        ratios = []
        for n_rows, n_cols in sizes:
            rng = np.random.default_rng(0)

            def factory(rng=rng, n_rows=n_rows, n_cols=n_cols):
                arr = alphabet_arr[
                    rng.integers(0, len(alphabet_arr), size=(n_rows, n_cols))
                ]
                return (arr, MoleculeType.PROTEIN), {}

            t_old = _time(_legacy_map_str_array_to_idx_array, factory)
            t_new = _time(map_str_array_to_idx_array, factory)
            ratio = t_old / max(t_new, 1e-9)
            ratios.append(ratio)
            rows.append(
                (
                    f"{n_rows}×{n_cols}",
                    f"{t_old * 1000:.2f}ms",
                    f"{t_new * 1000:.2f}ms",
                    f"{ratio:.1f}x",
                )
            )
        _report_scaling(
            "map_str_array_to_idx_array (O(N×A) → O(N))",
            ["shape", "legacy", "new", "speedup"],
            rows,
        )
        # Alphabet size A is a constant (~22 for PROTEIN), so both impls are
        # effectively O(N). We expect a large *flat* speedup ratio (~alphabet
        # size × constant factors), not a growing one. Assert it stays high
        # and doesn't collapse at scale.
        assert ratios[-1] >= ratios[0] * 0.7, (
            f"speedup degraded with size: {ratios[0]:.1f}x → {ratios[-1]:.1f}x"
        )
        assert min(ratios) > 5.0, f"speedup too small: {ratios}"

    def test_create_main_dedup_scaling(self):
        """Legacy: O(N log N × L) sort-based. New: O(N × L) hash-based."""
        sizes = [1_000, 10_000, 50_000]
        n_cols = 96

        rows = []
        ratios = []
        for n_rows in sizes:
            rng = np.random.default_rng(0)

            def factory(rng=rng, n_rows=n_rows, n_cols=n_cols):
                base = rng.integers(0, 5, size=(n_rows // 2, n_cols), dtype=np.int32)
                dup_idx = rng.integers(0, base.shape[0], size=n_rows - base.shape[0])
                arr = np.concatenate([base, base[dup_idx]], axis=0)
                rng.shuffle(arr)
                return (arr,), {}

            t_old = _time(_legacy_unique_idx, factory)
            t_new = _time(_new_unique_idx, factory)
            ratio = t_old / max(t_new, 1e-9)
            ratios.append(ratio)
            rows.append(
                (
                    f"{n_rows}",
                    f"{t_old * 1000:.2f}ms",
                    f"{t_new * 1000:.2f}ms",
                    f"{ratio:.1f}x",
                )
            )
        _report_scaling(
            "create_main.dedup (O(N log N) → O(N))",
            ["n_rows", "legacy", "new", "speedup"],
            rows,
        )
        assert ratios[-1] > ratios[0], (
            f"speedup did not grow with size: {ratios[0]:.1f}x → {ratios[-1]:.1f}x"
        )

    def test_cap_seqs_per_species_scaling(self):
        """Legacy: N_species per-group allocations + 2 concats. New: single pass."""
        # (n_species, per_species) tuples
        sizes = [(10, 10), (100, 30), (500, 20)]

        rows = []
        ratios = []
        for n_species, per_species in sizes:
            rng = np.random.default_rng(0)
            species_ids = np.repeat(np.arange(n_species), per_species).tolist()
            rng.shuffle(species_ids)
            seq_len = 64

            def factory(rng=rng, species_ids=species_ids, seq_len=seq_len):
                alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY-"))
                msa = alphabet[
                    rng.integers(0, len(alphabet), size=(len(species_ids) + 1, seq_len))
                ]
                deletion = rng.integers(0, 3, size=(len(species_ids) + 1, seq_len))
                metadata = pd.DataFrame({"species_id": list(species_ids)})
                m = MsaArray(msa=msa, deletion_matrix=deletion, metadata=metadata)
                return (m, 5), {}

            t_old = _time(_legacy_cap_seqs_per_species, factory)
            t_new = _time(cap_seqs_per_species, factory)
            ratio = t_old / max(t_new, 1e-9)
            ratios.append(ratio)
            rows.append(
                (
                    f"{n_species}×{per_species}",
                    f"{t_old * 1000:.2f}ms",
                    f"{t_new * 1000:.2f}ms",
                    f"{ratio:.1f}x",
                )
            )
        _report_scaling(
            "cap_seqs_per_species (N_species allocs → single pass)",
            ["sp×per", "legacy", "new", "speedup"],
            rows,
        )
        assert ratios[-1] > ratios[0], (
            f"speedup did not grow with size: {ratios[0]:.1f}x → {ratios[-1]:.1f}x"
        )

    def test_chain_connected_molecule_iter_scaling(self):
        """Legacy: BondList.merge over n_atoms synthetic bonds. New: direct edges."""
        sizes = [500, 2_000, 5_000]
        n_chains = 10

        rows = []
        ratios = []
        for n_atoms in sizes:
            rng = np.random.default_rng(0)
            chain_ids = np.repeat(
                np.array([f"C{i}" for i in range(n_chains)]), n_atoms // n_chains
            )
            if len(chain_ids) < n_atoms:
                chain_ids = np.concatenate(
                    [chain_ids, np.full(n_atoms - len(chain_ids), f"C{n_chains - 1}")]
                )
            n_edges = min(n_atoms * 3 // 4, 5000)
            src = rng.integers(0, n_atoms, size=n_edges, dtype=np.uint32)
            dst = (src + rng.integers(1, 10, size=n_edges, dtype=np.uint32)) % n_atoms
            bond_pairs = np.column_stack((src, dst))

            def factory(n_atoms=n_atoms, chain_ids=chain_ids, bond_pairs=bond_pairs):
                arr = AtomArray(n_atoms)
                arr.coord = np.zeros((n_atoms, 3))
                arr.chain_id[:] = chain_ids
                arr.bonds = BondList(n_atoms, bond_pairs.copy())
                return (arr,), {}

            def consume_legacy(arr):
                return [
                    m.coord.shape[0] for m in _legacy_chain_connected_molecule_iter(arr)
                ]

            def consume_new(arr):
                return [m.coord.shape[0] for m in chain_connected_molecule_iter(arr)]

            t_old = _time(consume_legacy, factory)
            t_new = _time(consume_new, factory)
            ratio = t_old / max(t_new, 1e-9)
            ratios.append(ratio)
            rows.append(
                (
                    f"{n_atoms}",
                    f"{t_old * 1000:.2f}ms",
                    f"{t_new * 1000:.2f}ms",
                    f"{ratio:.1f}x",
                )
            )
        _report_scaling(
            "chain_connected_molecule_iter (BondList.merge eliminated)",
            ["n_atoms", "legacy", "new", "speedup"],
            rows,
        )
        assert ratios[-1] > ratios[0], (
            f"speedup did not grow with size: {ratios[0]:.1f}x → {ratios[-1]:.1f}x"
        )

    def test_get_molecule_indices_scaling(self):
        """Legacy: networkx (Python DFS). New: scipy.sparse connected_components."""
        sizes = [500, 2_000, 5_000]

        rows = []
        ratios = []
        for n_atoms in sizes:
            rng = np.random.default_rng(0)
            n_edges = min(n_atoms * 3 // 4, 5000)
            src = rng.integers(0, n_atoms, size=n_edges, dtype=np.uint32)
            dst = (src + rng.integers(1, 5, size=n_edges, dtype=np.uint32)) % n_atoms
            bond_pairs = np.column_stack((src, dst))

            def factory(n_atoms=n_atoms, bond_pairs=bond_pairs):
                arr = AtomArray(n_atoms)
                arr.coord = np.zeros((n_atoms, 3))
                arr.chain_id[:] = "A"
                arr.bonds = BondList(n_atoms, bond_pairs.copy())
                return (arr,), {}

            t_old = _time(_legacy_get_molecule_indices, factory)
            t_new = _time(get_molecule_indices, factory)
            ratio = t_old / max(t_new, 1e-9)
            ratios.append(ratio)
            rows.append(
                (
                    f"{n_atoms}",
                    f"{t_old * 1000:.2f}ms",
                    f"{t_new * 1000:.2f}ms",
                    f"{ratio:.1f}x",
                )
            )
        _report_scaling(
            "get_molecule_indices (networkx DFS → scipy C)",
            ["n_atoms", "legacy", "new", "speedup"],
            rows,
        )
        # For get_molecule_indices the win is primarily constant-factor (both are
        # linear in N+E), so we assert speedup *doesn't degrade* rather than
        # strictly grows. A ≥0.8× ratio-of-ratios is fine.
        assert ratios[-1] >= ratios[0] * 0.8, (
            f"speedup degraded with size: {ratios[0]:.1f}x → {ratios[-1]:.1f}x"
        )
