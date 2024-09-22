"""Primitives for processing templates alignments."""

import dataclasses
import logging
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile

from openfold3.core.data.io.sequence.fasta import read_multichain_fasta
from openfold3.core.data.io.sequence.template import TemplateHit
from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.structure.labels import (
    get_chain_to_pdb_chain_dict,
)
from openfold3.core.data.primitives.structure.metadata import (
    get_chain_to_canonical_seq_dict,
    get_cif_block,
    get_release_date,
)

logger = logging.getLogger(__name__)


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
    dataset_cache: dict,
    is_core_train: bool,
) -> _TemplateQueryIterator:
    """Extracts the chain ID mappings and release dates from the dataset cache.

    Note: behaves differently for core training sets and distillation/inference sets
    to reduce the runtimes for the latter.

    Args:
        dataset_cache (dict):
            The precomputed dataset cache.
        is_core_train (bool):
            Parser mode. One of "construct", "filter_core_train", "filter_distillation".

    Returns:
        _TemplateDataIterator:
            The list of tuples of chain - representative ID pairs for core training or
            a dict mapping representative chain IDs to the list of corresponding PDB IDs
            for distillation training sets and inference sets
    """
    if is_core_train:
        return _TemplateQueryIterator(
            entries=[
                _TemplateQueryEntry(
                    chain_data["alignment_representative_id"],
                    _DatedQueryEntry(
                        f"{pdb_id}_{chain_id}", entry_data["release_date"]
                    ),
                )
                for pdb_id, entry_data in dataset_cache["structure_data"].items()
                for chain_id, chain_data in entry_data["chains"].items()
                if (chain_data["molecule_type"] == "PROTEIN")
            ],
        )
    else:
        entries = {}
        for pdb_id, entry_data in dataset_cache["structure_data"].items():
            for chain_id, chain_data in entry_data["chains"].items():
                if chain_data["molecule_type"] == "PROTEIN":
                    rep_id = chain_data["alignment_representative_id"]
                    if rep_id not in entries:
                        entries[rep_id] = []
                        entries[rep_id].append(
                            _DatedQueryEntry(
                                f"{pdb_id}_{chain_id}", entry_data["release_date"]
                            )
                        )
                    else:
                        entries[rep_id].append(
                            _DatedQueryEntry(
                                f"{pdb_id}_{chain_id}", entry_data["release_date"]
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
) -> bool:
    """Checks if the query sequences in the CIF and template alignment file match.

    Note: The chain-ID to sequence mapping is extracted from the preprocessed fasta file
    for the sanitized assembly and NOT directly from the CIF file.

    Args:
        query_structures_directory (Path):
            Path to the directory containing query structures in mmCIF format. The
            PDB IDs of the query CIF files need to match the PDB IDs for the query (1st
            row) in the alignment file for it to have any templates.
        query (TemplateHit):
            The query TemplateHit from the alignment.
        query_pdb_id (str):
            The PDB ID of the query.
        query_chain_id (str):
            The chain ID of the query.

    Returns:
        bool:
            Whether the query sequence-structure pair is invalid.
    """
    is_query_invalid = True
    # Attempt to locate template first template hit i.e. query in its CIF file
    chain_id_seq_map = read_multichain_fasta(
        query_structures_directory / Path(f"{query_pdb_id}.fasta")
    )
    query_seq_cif = chain_id_seq_map.get(query_chain_id)
    query_seq_hmm = query.hit_sequence.replace("-", "")

    if query_seq_cif is None:
        logging.info(
            f"Query {query_pdb_id} chain {query_chain_id} not found in CIF file."
        )
    elif (query_seq_cif is not None) & (query_seq_hmm not in query_seq_cif):
        logging.info(
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
            logging.info(
                f"Found HMM sequence of template {hit_pdb_id} chain {hit_chain_id}"
                f" in new chain {k}. Remapping."
            )
            break

    return hit_chain_id_matched


def match_template_chain_and_sequence(
    cif_file: CIFFile,
    atom_array: AtomArray,
    hit: TemplateHit,
) -> str | None:
    """Attempts to locate the chain of a template hit in its CIF file.

    Args:
        cif_file (CIFFile):
            Parsed mmCIF file.
        atom_array (AtomArray):
            Atom array of the structure.
        hit (TemplateHit):
            The template hit.

    Returns:
        str | None:
            The chain ID of the hit, or None if the chain is not found in the CIF file.
    """
    hit_pdb_id, hit_chain_id = hit.name.split("_")

    # Attempt to get the sequence of the hit chain
    # atom_array._annot["chain_id"] = atom_array.chain_id.astype(str)
    chain_id_map = get_chain_to_pdb_chain_dict(atom_array)
    cif_data = get_cif_block(cif_file)
    chain_id_seq_map_temp = get_chain_to_canonical_seq_dict(atom_array, cif_data)
    chain_id_seq_map = {
        chain_id_map[chain_id]: seq for chain_id, seq in chain_id_seq_map_temp.items()
    }
    hit_seq_cif = chain_id_seq_map.get(hit_chain_id)
    hit_seq_hmm = hit.hit_sequence.replace("-", "")

    # A) If chain ID not in CIF file, attempt to find sequence in other chains
    if hit_seq_cif is None:
        hit_chain_id_matched = remap_chain_id(
            hit_pdb_id, None, hit_seq_hmm, chain_id_seq_map
        )
    # B) If chain ID is in CIF file but HMM sequence does not match CIF sequence,
    # attempt to find in other chains
    elif (hit_seq_cif is not None) & (hit_seq_hmm not in hit_seq_cif):
        hit_chain_id_matched = remap_chain_id(
            hit_pdb_id, hit_chain_id, hit_seq_hmm, chain_id_seq_map
        )
    # C) If HMM sequence matches CIF sequence, use original chain ID
    else:
        hit_chain_id_matched = hit_chain_id

    return hit_chain_id_matched


def create_residue_idx_map(query: TemplateHit, hit: TemplateHit) -> dict[int, int]:
    """Create a mapping from the query to the hit residue indices.

    Args:
        query (TemplateHit):
            The query TemplateHit from the alignment.
        hit (TemplateHit):
            The filtered template TemplateHit from the alignment.

    Returns:
        dict[int, int]:
            A mapping from the query to the hit residue indices.
    """
    return {
        q_i: h_i for q_i, h_i in zip(query.indices_hit, hit.indices_hit) if h_i != -1
    }


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
    """Calc if the release date difference is at least as much as the minimum required.

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
            Whether the release date difference is at least as much as the minimum
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
