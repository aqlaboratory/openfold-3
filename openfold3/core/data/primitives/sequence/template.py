"""Primitives for processing templates alignments."""

import dataclasses
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import biotite.structure as struc
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile

from openfold3.core.data.io.sequence.fasta import read_multichain_fasta
from openfold3.core.data.io.sequence.template import TemplateHit
from openfold3.core.data.io.structure.cif import (
    parse_mmcif,
)
from openfold3.core.data.io.structure.pdb import parse_protein_monomer_pdb_tmp
from openfold3.core.data.primitives.caches.format import DatasetCache
from openfold3.core.data.primitives.quality_control.logging_utils import (
    TEMPLATE_PROCESS_LOGGER,
)
from openfold3.core.data.primitives.structure.metadata import (
    get_cif_block,
    get_release_date,
)
from openfold3.core.data.resources.residues import get_with_unknown_3_to_1


# Shared
class _DatedQueryEntry(NamedTuple):
    """Query PDB+chain ID, release date tuple."""

    query_pdb_chain_id: str
    query_release_date: str


class _TemplateQueryEntry(NamedTuple):
    """Representative PDB+chain ID, _DatedQueryEntry tuple."""

    rep_pdb_chain_id: str
    dated_query: _DatedQueryEntry | list[_DatedQueryEntry]


@dataclasses.dataclass(frozen=False)
class _TemplateQueryIterator:
    """Dataclass for iterating over queries to filter templates for

    Attributes:
        entries (list[_TemplateQueryEntry]):
            List of tuples of chain - representative ID pairs for template cache
            construction and core training filtering OR a list of tuples of
            representative chain IDs to the list of corresponding PDB IDs for
            distillation training sets and inference sets.
    """

    entries: list[_TemplateQueryEntry]


def parse_representatives(
    dataset_cache: DatasetCache,
    is_core_train: bool,
    single_moltype: str | None,
) -> _TemplateQueryIterator:
    """Extracts the chain ID mappings and release dates from the dataset cache.

    Note: behaves differently for core training sets and distillation/inference sets to
    reduce the runtimes for the latter.

    Args:
        dataset_cache (DatasetCache):
            The precomputed dataset cache.
        is_core_train (bool):
            Parser mode. One of "construct", "filter_core_train", "filter_distillation".
        single_moltype (str | None):
            Constant molecule type to be used if the molecule_type field is not
            available in the input cache chain data. Can be used for caches whose every
            sample contains the same molecule type.

    Returns:
        _TemplateDataIterator:
            The list of tuples of chain - representative ID pairs for core training or a
            dict mapping representative chain IDs to the list of corresponding PDB IDs
            for distillation training sets and inference sets
    """
    if is_core_train:
        return _TemplateQueryIterator(
            entries=[
                _TemplateQueryEntry(
                    chain_data.alignment_representative_id,
                    _DatedQueryEntry(
                        f"{pdb_id}_{chain_id}",
                        getattr(entry_data, "release_date", None),
                    ),
                )
                for pdb_id, entry_data in dataset_cache.structure_data.items()
                for chain_id, chain_data in entry_data.chains.items()
                if (getattr(chain_data, "molecule_type", single_moltype) == "PROTEIN")
            ],
        )
    else:
        entries = {}
        for pdb_id, entry_data in dataset_cache.structure_data.items():
            for chain_id, chain_data in entry_data.chains.items():
                if getattr(chain_data, "molecule_type", single_moltype) == "PROTEIN":
                    rep_id = chain_data.alignment_representative_id
                    if rep_id not in entries:
                        entries[rep_id] = []
                        entries[rep_id].append(
                            _DatedQueryEntry(
                                f"{pdb_id}_{chain_id}",
                                getattr(entry_data, "release_date", None),
                            )
                        )
                    else:
                        entries[rep_id].append(
                            _DatedQueryEntry(
                                f"{pdb_id}_{chain_id}",
                                getattr(entry_data, "release_date", None),
                            )
                        )
        return _TemplateQueryIterator(
            entries=[
                _TemplateQueryEntry(rep, dated_query)
                for rep, dated_query in entries.items()
            ]
        )


# Template cache construction
def check_sequence(
    query_seq: str,
    hit: TemplateHit,
    max_subseq: float = 0.95,
    min_align: float = 0.1,
    min_len: int = 10,
) -> bool:
    """Applies sequence filters to template hits following AF3 SI Section 2.4.

    Args:
        query_seq (str):
            The query sequence.
        hit (TemplateHit):
            Candidate template hit.
        max_subseq (float, optional):
            Maximum allowed sequence idenity between query and hit. Defaults to 0.95.
        min_align (float, optional):
            Minimum required query coverage of the template hit. Defaults to 0.1.
        min_len (int, optional):
            Minimum required template hit length. Defaults to 10 residues.

    Returns:
        bool:
            Whether the hit passes the sequence filters.
    """
    hit_seq = hit.hit_sequence.replace("-", "")
    return (
        ((len(hit_seq) / len(query_seq)) > max_subseq)
        | ((hit.aligned_cols / len(query_seq)) < min_align)
        | (len(hit_seq) < min_len)
    )


def parse_release_date(cif_file: CIFFile) -> datetime:
    """Parses the release date of a structure from its CIF file.

    Args:
        cif_file (CIFFile):
            Parsed mmCIF file containing the structure.

    Returns:
        datetime:
            The release date of the structure.
    """
    cif_data = get_cif_block(cif_file)
    return datetime.strptime(
        get_release_date(cif_data).strftime("%Y-%m-%d"), "%Y-%m-%d"
    )


def parse_structure(
    structures_path: Path, query_pdb_id: str, file_format: str = "bcif"
) -> tuple[CIFFile | None, AtomArray | None]:
    """Attempts to parse the structure of a given PDB ID from a target directory.

    Args:
        structures_path (Path):
            Path to the directory containing structures in mmCIF format.
        query_pdb_id (str):
            The PDB ID of the structure to parse.
        file_format (str, optional):
            The file format of the structure. Defaults to "bcif".

    Returns:
        tuple[CIFFile | None, AtomArray | None]:
            The parsed CIF file and atom array of the structure, or None if the
            structure is not found.
    """
    try:
        cif_file, atom_array = parse_mmcif(
            structures_path / Path(f"{query_pdb_id}.{file_format}")
        )
    except FileNotFoundError:
        cif_file, atom_array = None, None
    return cif_file, atom_array


def match_query_chain_and_sequence(
    query_structures_directory: Path,
    query: TemplateHit,
    query_pdb_id: str,
    query_chain_id: str,
    query_seq_load_logic: str,
    query_file_format: str,
    query_structure_filename: str,
    s3_profile: str | None,
) -> bool:
    """Checks if the query sequences in the CIF and template alignment file match.

    Note: The chain-ID to sequence mapping is attempted to be extracted first from the
    preprocessed fasta file for the sanitized assembly, then from the CIF or PDB file if
    the fasta file is not found.

    Args:
        query_structures_directory (Path):
            Path to the directory containing query structures in mmCIF format. The PDB
            IDs of the query CIF files need to match the PDB IDs for the query (1st row)
            in the alignment file for it to have any templates.
        query (TemplateHit):
            The query TemplateHit from the alignment.
        query_pdb_id (str):
            The PDB ID of the query.
        query_chain_id (str):
            The chain ID of the query.
        query_seq_load_logic (str):
            Whether to load the query sequence from the preprocessed fasta file or from
            a structure file.
        query_file_format (str):
            The file format of the query structure from which the sequence is parsed if
            load_logic is set to 'structure'.
        query_structure_filename (str):
            The filename of the query structure file. Only used if seq_load_logic is set
            to 'structure'.
        s3_profile (str | None):
            The AWS profile to use for downloading the CIF file from S3. Only used if
            seq_load_logic is set to 'structure'.

    Returns:
        bool:
            Whether the query sequence-structure pair is invalid.
    """
    is_query_invalid = True

    # TODO: rework this logic, currently only 2 options are supported
    # Get the query sequence from the structure
    if query_seq_load_logic == "fasta":
        chain_id_seq_map = read_multichain_fasta(
            query_structures_directory / Path(f"{query_pdb_id}.fasta")
        )
        query_seq_structure = chain_id_seq_map.get(query_chain_id)
    elif query_seq_load_logic == "structure":
        file_path = query_structures_directory / Path(
            f"{query_structure_filename}.{query_file_format}"
        )
        if query_file_format in ["cif", "bcif"]:
            # _, atom_array = parse_structure(
            # file_path, query_pdb_id, query_file_format)
            raise NotImplementedError
        elif query_file_format == "pdb":
            _, atom_array = parse_protein_monomer_pdb_tmp(
                file_path, s3_profile=s3_profile
            )
        else:
            raise ValueError(
                "Invalid query file format. Must be one of 'cif', 'bcif', or 'pdb'."
            )

        # Attempt to find the chain ID from the query TemplateHit name
        atom_array = atom_array[atom_array.chain_id == query_chain_id]

        # Fetch sequence from structure
        residue_starts = struc.get_residue_starts(atom_array)
        residue_names = atom_array.res_name[residue_starts]
        query_seq_structure = "".join(list(get_with_unknown_3_to_1(residue_names)))
    else:
        raise ValueError("Invalid load logic. Must be one of 'fasta' or 'structure'.")

    # Get the query sequence from the template alignment
    query_seq_hmm = query.hit_sequence.replace("-", "")

    # Compare
    if query_seq_structure is None:
        TEMPLATE_PROCESS_LOGGER.get().info(
            f"Query {query_pdb_id} chain {query_chain_id} not found in CIF file."
        )
    elif (query_seq_structure is not None) & (query_seq_hmm not in query_seq_structure):
        TEMPLATE_PROCESS_LOGGER.get().info(
            f"Query {query_pdb_id} chain {query_chain_id} sequence does not match CIF"
            " sequence."
        )
    else:
        is_query_invalid = False

    return is_query_invalid


def remap_chain_id(
    hit_pdb_id: str,
    hit_chain_id: str,
    hit_seq_hmm: str,
    chain_id_seq_map: dict[str, str],
) -> str | None:
    """Remaps the chain ID of a hit if the HMM sequence is found in another chain.

    Args:
        hit_pdb_id (str):
            The PDB ID of the hit.
        hit_chain_id (str):
            The chain ID of the hit.
        hit_seq_hmm (str):
            The HMM sequence of the hit.
        chain_id_seq_map (dict[str, str]):
            A mapping of chain IDs to their sequences parsed from the CIF file.

    Returns:
        str | None:
            The remapped chain ID, or None if no matching sequence is found.
    """
    hit_chain_id_matched = None
    for k, v in chain_id_seq_map.items():
        # Skip the original hit chain
        if k == hit_chain_id:
            continue
        # Remap hit chain ID if found in another chain
        if hit_seq_hmm in v:
            hit_chain_id_matched = k
            TEMPLATE_PROCESS_LOGGER.get().info(
                f"Found HMM sequence of template {hit_pdb_id} chain {hit_chain_id}"
                f" in new chain {k}. Remapping."
            )
            break

    return hit_chain_id_matched


def match_template_chain_and_sequence(
    chain_id_seq_map: dict[str, str],
    hit: TemplateHit,
) -> str | None:
    """Attempts to locate the chain of a template hit in its CIF file.

    Args:
        chain_id_seq_map (dict[str, str]):
            A mapping of chain IDs to their sequences parsed from the CIF file.
        hit (TemplateHit):
            The template hit.

    Returns:
        str | None:
            The chain ID of the hit, or None if the chain is not found in the CIF file.
    """
    hit_pdb_id, hit_chain_id = hit.name.split("_")

    # Attempt to get the sequence of the hit chain
    hit_seq_cif = chain_id_seq_map.get(hit_chain_id)
    hit_seq_hmm = hit.hit_sequence.replace("-", "")

    # A) If chain ID not in CIF file, attempt to find sequence in other chains
    if hit_seq_cif is None:
        TEMPLATE_PROCESS_LOGGER.get().info(
            f"Template {hit_pdb_id} chain {hit_chain_id} not found in CIF file."
            " Attempting to remap to other chains."
        )
        hit_chain_id_matched = remap_chain_id(
            hit_pdb_id, None, hit_seq_hmm, chain_id_seq_map
        )
    # B) If chain ID is in CIF file but HMM sequence does not match CIF sequence,
    # attempt to find in other chains
    elif (hit_seq_cif is not None) & (hit_seq_hmm not in hit_seq_cif):
        TEMPLATE_PROCESS_LOGGER.get().info(
            f"Template {hit_pdb_id} chain {hit_chain_id} found but mismatches "
            "sequence in CIF file. Attempting to remap to other chains."
        )
        hit_chain_id_matched = remap_chain_id(
            hit_pdb_id, hit_chain_id, hit_seq_hmm, chain_id_seq_map
        )
    # C) If HMM sequence matches CIF sequence, use original chain ID
    else:
        hit_chain_id_matched = hit_chain_id
        TEMPLATE_PROCESS_LOGGER.get().info(
            f"Template {hit_pdb_id} chain {hit_chain_id} HMM sequence matches "
            "CIF sequence."
        )

    return hit_chain_id_matched


def create_residue_idx_map(query: TemplateHit, hit: TemplateHit) -> np.ndarray[int]:
    """Create a mapping from the query to the hit residue indices.

    Args:
        query (TemplateHit):
            The query TemplateHit from the alignment.
        hit (TemplateHit):
            The filtered template TemplateHit from the alignment.

    Returns:
        np.ndarray[int]:
            A n_aligned_column-by-2 numpy array containing >global< residue indices of
            the query (1st col) and aligned hit (2nd col) for columns where the template
            sequence is not a gap.
    """
    return np.asarray(
        [
            (q_i, h_i)
            for q_i, h_i in zip(query.indices_hit, hit.indices_hit)
            if h_i != -1
        ],
        dtype=int,
    )


# Template cache filtering
@dataclasses.dataclass(frozen=False)
class TemplateHitCollection:
    """A dict-wrapper dataclass for storing filtered template hits."""

    data: dict[tuple[str, str], list[str]] = dataclasses.field(default_factory=dict)

    def __getitem__(self, key: tuple[str, str]) -> list[str]:
        return self.data[key]

    def __setitem__(self, key: tuple[str, str], value: list[str]) -> None:
        self.data[key] = value

    def __delitem__(self, key: tuple[str, str]) -> None:
        del self.data[key]

    def get(self, key: tuple[str, str], default=None) -> list[str]:
        return self.data.get(key, default)

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __repr__(self):
        return f"{self.data}"


def check_release_date_diff(
    query_release_date: datetime,
    template_release_date: datetime,
    min_release_date_diff: int,
) -> bool:
    """Calculates if the release date difference is less than the minimum required.

    As per AF3 SI Section 2.4. Used for the core training set.

    Args:
        query_release_date (datetime):
            The release date of the query structure.
        template_release_date (datetime):
            The release date of the template structure.
        min_release_date_diff (int):
            The minimum number of days required for the template to be released before a
            query structure.

    Returns:
        bool:
            Whether the release date difference is less than the minimum
            required.
    """
    return (query_release_date - template_release_date).days < min_release_date_diff


def check_release_date_max(
    template_release_date: datetime,
    max_release_date: datetime,
) -> bool:
    """Calc if the release date is before the maximum allowed release date.

    As per AF3 SI Section 2.4. Used for distillation and inference sets.

    Args:
        template_release_date (datetime):
            The release date of the template structure.
        max_release_date (datetime):
            The maximum allowed release date.

    Returns:
        bool:
            Whether the release date is before the maximum allowed release date.
    """
    return template_release_date < max_release_date
