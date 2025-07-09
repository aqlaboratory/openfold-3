"""
Parsers for template alignments.
"""

from collections.abc import Iterable
import re
from typing import NamedTuple

import numpy as np
import pandas as pd

from openfold3.core.data.io.sequence.fasta import parse_fasta
from openfold3.core.data.resources.residues import MoleculeType

"""
Updates compared to the old OpenFold version:

The new template parsers expect the input template alignment to contain the query 
sequence as the first sequence in the alignment, globally aligned to the template hit
sequences. We achieve this by re-aligning the output sequences from hmmsearch to the 
query using hmmalign.

Other minor changes from old version:
    - dataclass -> NamedTuple
    - Removed skip_first argument from parse_hmmsearch_sto and parse_hmmsearch_a3m
    - Removed query_sequence and query_indices from parse_hmmsearch_sto, 
    parse_hmmsearch_a3m and TemplateHit
    - parse_hmmsearch_a3m and parse_hmmsearch_sto now return a dict[int, TemplateHit]
    - index in TemplateHit dict starts from 0 instead of 1
    - replaced e_value None with 0
"""


class HitMetadata(NamedTuple):
    """Tuple containing metadata for a hit in an HMM search.

    Attributes:
        pdb_id (str):
            The PDB ID of the hit.
        chain (str):
            The chain ID of the hit.
        start (int):
            The index of the first residue of the aligned hit substring in the full
            hit sequence.
    """

    pdb_id: str
    chain: str
    start: int


class TemplateHit(NamedTuple):
    """Tuple containing template hit information.

    Attributes:
        index (str):
            Row index of the hit in the alignment.
        name (str):
            PDB-chain ID of the hit.
        aligned_cols (int):
            Number of
        hit_sequence (str):
            The PDB ID of the hit.
        indices_hit (str):
            The PDB ID of the hit.
        e_value (str):
            The PDB ID of the hit.
    """

    index: int
    name: str
    aligned_cols: int
    hit_sequence: str
    indices_hit: list[int]
    e_value: float | None


def parse_entry_chain_id(entry_chain_id: str) -> tuple[str, str]:
    """Extracts the chain ID from a query entry.

    Assumes the format ENTRY_CHAIN or ENTRY. If ENTRY, the chain ID is assumed to be 1.

    Args:
        entry_chain_id (str):
            The entry-chain ID string.

    Returns:
        tuple[str, str]:
            The entry-chain ID tuple.
    """
    entry_chain_id_list = entry_chain_id.split("_")
    if len(entry_chain_id_list) == 1:
        return entry_chain_id_list[0], "1"
    elif len(entry_chain_id_list) == 2:
        return entry_chain_id_list[0], entry_chain_id_list[1]
    else:
        raise ValueError(
            "Invalid entry-chain ID format. Must be 'ENTRY' or 'ENTRY_CHAIN'."
        )


def _get_indices(sequence: str, start: int) -> list[int]:
    """Returns an index encoding of the aligned sequence starting at the given index.

    Indices for non-gap/insert residues are given a positive index 1 larger that the
    previous non-gap/insert residue, whereas gaps and deleted residues are given a
    -1 index.

    Args:
        sequence (str):
            Hit subsequence spanned by the global alignment to the query sequence.
        start (int):
            Starting index of the hit

    Returns:
        list[int]: _description_
    """
    indices = []
    index_runner = start
    for symbol in sequence:
        # Skip gaps but add a placeholder so that the alignment is preserved.
        if symbol == "-":
            indices.append(-1)
        # Skip deleted residues, but increase the counter.
        elif symbol.islower():
            index_runner += 1
        # Normal aligned residue. Increase the counter and append to indices.
        else:
            indices.append(index_runner)
            index_runner += 1
    return indices


def _parse_hmmsearch_description(description: str, index: int) -> HitMetadata:
    """Parses the hmmsearch + hmmalign A3M sequence description line.

    Example 1: >4pqx_A/2-217 [subseq from] mol:protein length:217  Free text
    Example 2: >5g3r_A/1-55 [subseq from] mol:protein length:352

    Args:
        description (str):
            STO sequence description line.
    Raises:
        ValueError:
            If the description cannot be parsed.

    Returns:
        HitMetadata:
            Metadata for the hit.
    """
    # Check if the description line contains a subsequence range
    desc_split = description.split("/")
    if len(desc_split) == 1:
        pdb_chain_id = desc_split[0]
        desc = None
    else:
        pdb_chain_id = desc_split[0]
        desc = " ".join(desc_split[1:])

    # Parse the PDB ID, chain ID and start index
    pdb_id, chain_id = parse_entry_chain_id(pdb_chain_id)
    if index == 0:
        start_index = 1
    else:
        start_index = int(desc.split(" ")[0].split("-")[0])

    return HitMetadata(
        pdb_id=pdb_id,
        chain=chain_id,
        start=start_index,
    )


def _convert_sto_seq_to_a3m(query_non_gaps: list[bool], sto_seq: str) -> Iterable[str]:
    """Convert stockholm sequence to a3m format.

    Args:
        query_non_gaps (list[bool]):
            List of booleans indicating whether the query sequence has a non-gap residue
            at each position.
        sto_seq (str):
            Stockholm sequence to convert to a3m format.

    Yields:
        Iterator[Iterable[str]]:
            Converted a3m sequence.
    """
    for is_query_res_non_gap, sequence_res in zip(query_non_gaps, sto_seq):
        if is_query_res_non_gap:
            yield sequence_res
        elif sequence_res != "-":
            yield sequence_res.lower()


def convert_stockholm_to_a3m(
    stockholm_string: str,
    remove_first_row_gaps: bool = False,
    max_sequences: int | None = None,
) -> str:
    """Converts MSA in Stockholm format to the A3M format.

    Args:
        stockholm_string (str):
            Stockholm formatted alignment string produced by hmmsearch + hmmalign.
        remove_first_row_gaps (bool, optional):
            Whether to remove gaps in the first row of the alignment. Defaults to False.
        max_sequences (Optional[int], optional):
            Maximum number of sequences to include in the output. Defaults to None.

    Returns:
        str:
            A3M formatted alignment string.
    """
    descriptions = {}
    sequences = {}
    reached_max_sequences = False

    for line in stockholm_string.splitlines():
        reached_max_sequences = max_sequences and len(sequences) >= max_sequences
        if line.strip() and not line.startswith(("#", "//")):
            # Ignore blank lines, markup and end symbols - remainder are alignment
            # sequence parts.
            seqname, aligned_seq = line.split(maxsplit=1)
            if seqname not in sequences:
                if reached_max_sequences:
                    continue
                sequences[seqname] = ""
            sequences[seqname] += aligned_seq

    for line in stockholm_string.splitlines():
        if line[:4] == "#=GS":
            # Description row - example format is:
            # #=GS UniRef90_Q9H5Z4/4-78            DE [subseq from] cDNA: FLJ22755 ...
            columns = line.split(maxsplit=3)
            seqname, feature = columns[1:3]
            value = columns[3] if len(columns) == 4 else ""
            if feature != "DE":
                continue
            if reached_max_sequences and seqname not in sequences:
                continue
            descriptions[seqname] = value
            if len(descriptions) == len(sequences):
                break

    # Convert sto format to a3m line by line
    a3m_sequences = {}
    if remove_first_row_gaps:
        # query_sequence is assumed to be the first sequence
        query_sequence = next(iter(sequences.values()))
        query_non_gaps = [res != "-" for res in query_sequence]
    for seqname, sto_sequence in sequences.items():
        # Dots are optional in a3m format and are commonly removed.
        out_sequence = sto_sequence.replace(".", "")
        if remove_first_row_gaps:
            out_sequence = "".join(
                _convert_sto_seq_to_a3m(query_non_gaps, out_sequence)
            )
        a3m_sequences[seqname] = out_sequence

    fasta_chunks = (
        f">{k} {descriptions.get(k, '')}\n{a3m_sequences[k]}" for k in a3m_sequences
    )
    return "\n".join(fasta_chunks) + "\n"  # Include terminating newline.


def parse_hmmsearch_a3m(a3m_string: str) -> dict[int, TemplateHit]:
    """Parses an a3m string produced by hmmsearch + hhalign.

    Expects the query sequence to be the first sequence in the alignment
    and all other sequences to be globally aligned to it.

    Args:
        a3m_string (str):
            A3M formatted alignment string produced by hmmsearch + hhalign.

    Returns:
        dict[int, TemplateHit]:
            Dictionary mapping the index of the hit in the alignment to the parsed
            template hit.
    """
    # Zip the descriptions and MSAs together
    parsed_a3m = list(zip(*parse_fasta(a3m_string)))

    hits = {}
    for i, (hit_sequence, hit_description) in enumerate(parsed_a3m):
        # Never skip first entry (query) but skip non-protein chains
        if (i != 0) & ("mol:protein" not in hit_description):
            continue

        # Parse the hit description line
        metadata = _parse_hmmsearch_description(hit_description, i)

        # Aligned columns are only the match states
        aligned_cols = sum([r.isupper() and r != "-" for r in hit_sequence])
        indices_hit = _get_indices(hit_sequence, start=metadata.start)

        # Embed in TempateHit dataclass
        hits[i] = TemplateHit(
            index=i,
            name=f"{metadata.pdb_id}_{metadata.chain}",
            aligned_cols=aligned_cols,
            e_value=0,
            hit_sequence=hit_sequence.upper(),
            indices_hit=indices_hit,
        )

    return hits


def parse_hmmsearch_sto(stockholm_string: str) -> dict[int, TemplateHit]:
    """Parses an stockholm string produced by hmmsearch + hmmalign.

    The returned dictionary maps the index of the hit in the alignment to the parsed
    template hit.

    Args:
        stockholm_string (str):
            Stockholm formatted alignment string produced by hmmsearch + hmmalign.

    Returns:
        dict[int, TemplateHit]:
            Dictionary mapping the index of the hit in the alignment to the parsed
            template hit.
    """
    a3m_string = convert_stockholm_to_a3m(stockholm_string)
    template_hits = parse_hmmsearch_a3m(a3m_string=a3m_string)
    return template_hits

# New template alignment parsers for inference
# TODO: update old parsers and pipelines for training with these
class TemplateData(NamedTuple):
    """Tuple storing information about a template hit in an alignment.

    Attributes:
        index (int): 
            The index of the template hit in the alignment.
        entry_id (str):
            The entry ID of the template.
        chain_id (str):
            The chain ID of the template.
        query_ids_hit (np.ndarray):
            The indices of the query residues that hit the template.
        template_ids_hit (np.ndarray):
            The indices of the template residues that hit the query.
        template_sequence (str):
            The sequence of the template aligned to the query.
        e_value (float | None):
            The e-value of the template hit, if available. Defaults to None.
    """

    index: int
    entry_id: str
    chain_id: str
    query_ids_hit: np.ndarray
    template_ids_hit: np.ndarray
    template_sequence: str
    e_value: float | None

def parse_hmmer_headers(hmmalign_string: str, max_sequences: int) -> tuple[pd.DataFrame, list[str]]:
    """Parses the headers from an hmmalign MSA string in Stockholm format.

    Expects the headers to be in the format:
        #=GS <entry id>_<chain id>/<start res id>-<end res id> mol:<moltype>
    where 
        <entry id>: an arbitrary ID for the first row and PDB ID for all other rows
        <chain id>: chain ID
        <start res id>: the residue ID of the first residue of the aligned sequence segment in the full sequence
        <end res id>: the residue ID of the last residue of the aligned sequence segment in the full sequence
        <moltype>: the molecule type, one of "protein", "dna", "rna"

    Args:
        hmmalign_string (str):
            The string containing the hmmalign MSA in Stockholm format.
        max_sequences (int):
            The maximum number of sequences to parse from the alignment.

    Returns:
        tuple[pd.DataFrame, list[str]]:
            A tuple containing:
            - A DataFrame with columns "id", "start", "end", and "moltype" for each sequence.
            - A list of sequence IDs in the order they appear in the alignment.
    """
    regex = re.compile(r"^#=GS\s+([^/]+)/(\d+)-(\d+).*?mol:(\w+)", re.MULTILINE)
    headers = pd.DataFrame([list(match) for match in regex.findall(hmmalign_string)][:max_sequences + 1], 
                        columns=["id", "start", "end", "moltype"])
    regex = re.compile(r"^#=GS\s+(\S+)", re.MULTILINE)
    ordered_ids = regex.findall(hmmalign_string)

    return headers, ordered_ids

def parse_hmmer_sequences(hmmalign_string: str) -> dict[str, str]:
    """Parses an hmmalign MSA str in sto format into a mapping of sequence IDs to sequences.

    Args:
        hmmalign_string (str):
            The string containing the hmmalign MSA in Stockholm format.

    Returns:
        dict[str, str]:
            A dictionary mapping sequence IDs to their corresponding sequences.
    """
    sequence_map = {}
    for line in hmmalign_string.splitlines():
        # Ignore annotation lines and blank lines
        if not line.strip() or line.startswith("#") or line.startswith("//"):
            continue

        # Split the line into the full ID and the sequence
        full_id, chunk = line.split(maxsplit=1)

        sequence_map[full_id] = sequence_map.get(full_id, "") + chunk.strip()

    return sequence_map
    
def calculate_ids_hit(q: np.ndarray, t: np.ndarray, query_start: int, template_start: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the residue correspondences between the full query and template sequences.

    Args:
        q (np.ndarray):
            The aligned query sequence as a numpy array of characters.
        t (np.ndarray):
            The aligned template sequence as a numpy array of characters.
        query_start (int):
            The starting index of the aligned query sequence segment in the full query sequence.
        template_start (int):
            The starting index of the aligned template sequence segment in the full template sequence.

    Returns:
        tuple[np.ndarray, np.ndarray]: _description_
    """

    # 1. Create boolean masks to identify non-gaps
    q_is_residue = (q != '-')
    t_is_residue = (t != '-')

    # 2. Create a mask to identify columns that should be kept
    columns_to_keep = q_is_residue | t_is_residue

    # 3. Calculate the running count of residues for each sequence
    q_cumsum = np.cumsum(q_is_residue)
    t_cumsum = np.cumsum(t_is_residue)

    # 4.  Apply the start offset and set gap positions to 0
    query_map = np.where(q_is_residue, q_cumsum + query_start - 1, -1)
    template_map = np.where(t_is_residue, t_cumsum + template_start - 1, -1)

    # 5. Filter out the columns where both sequences had a gap
    return query_map[columns_to_keep], template_map[columns_to_keep]

def create_template_hits(sequence_map: dict[str, str], headers: pd.DataFrame, ordered_ids: list[str], max_sequences: int) -> dict[int, TemplateData]:
    """Parses the sequence map and headers to create TemplateHit objects.

    Args:
        sequence_map (dict[str, str]):
            A mapping of sequence IDs to their corresponding sequences.
        headers (pd.DataFrame):
            A DataFrame containing the headers with columns "id", "start", "end", and "moltype".
        ordered_ids (list[str]):
            A list of sequence IDs in the order they appear in the alignment.
        max_sequences (int):
            The maximum number of sequences to consider for template hits.

    Returns:
        dict[int, TemplateHit]:
            A dictionary mapping row indices to TemplateHit objects, containing
            the aligned sequences and their indices in the ungapped query sequence.
    """

    # query_id should be the full ID like "1a0a_A/1-63" but is misformatted in the
    # snakemake pipeline and only has the base ID like "1a0a_A" so trying both here
    query_id = ordered_ids[0].split("/")[0]
    if query_id not in sequence_map:
        query_id = ordered_ids[0]

    # Get the ungapped query sequence and its mask 
    query_sequence_str = sequence_map[query_id]
    query_sequence = np.fromiter(query_sequence_str, dtype='<U1', count=len(query_sequence_str))
    query_nongap_mask = ~np.isin(query_sequence, ['-', '.'])
    query_sequence_nongap = query_sequence[query_nongap_mask]
    
    query_moltype = MoleculeType[headers.iloc[0]["moltype"].upper()]
    query_start = int(headers.iloc[0]["start"])

    # Iterate over the remaining sequences up to max_sequences
    templates = {}
    for seq_id, row in zip(ordered_ids[1:], headers.iterrows()):
        if seq_id in sequence_map:
            template_sequence_str = sequence_map[seq_id]
            template_sequence = np.fromiter(template_sequence_str, dtype='<U1', count=len(template_sequence_str))
            # np array of the template aligned to the ungapped query sequence
            template_sequence_nongap = template_sequence[query_nongap_mask]
            # str of the ungapped template sequence
            template_sequence_str_nongap = "".join(template_sequence[~np.isin(template_sequence, ['-', '.'])]).upper()

            index = row[0]
            header = row[1]

            # Skip the first sequence (query) and check molecule type
            if (index == 0) or (MoleculeType[header["moltype"].upper()] != query_moltype):
                continue

            # Get residue correspondences wrt. the full-length sequences
            query_ids_hit, template_ids_hit = calculate_ids_hit(
                q=query_sequence_nongap,
                t=template_sequence_nongap,
                query_start=query_start,
                template_start=int(header["start"]),
            )
            
            entry_id, chain_id = header["id"].split("_")
            template_hit = TemplateData(
                index=index,
                entry_id=entry_id,
                chain_id=chain_id,
                query_ids_hit=query_ids_hit,
                template_ids_hit=template_ids_hit,
                template_sequence=template_sequence_str_nongap,
                e_value=None,
            )
            templates[index] = template_hit

        if len(templates) == max_sequences:
            break

    return templates

def parse_hmmer_sto(hmm_string: str, max_sequences: int) -> dict[int, TemplateData]:
    """Parses an hmmalign MSA in Stockholm format into a dict of query-template mappings.

    Args:
        hmm_string (str):
            The string containing the hmmalign/hmmsearch MSA in Stockholm format.
        max_sequences (int):
            The maximum number of sequences to parse from the alignment.

    Returns:
        dict[int, TemplateData]:
            A dictionary mapping row indices to TemplateData objects, containing
            the aligned sequences and their indices in the ungapped query sequence.
    """
    
    # 1. Parse headers from format "#=GS id/start-end moltype"
    headers, ordered_ids = parse_hmmer_headers(hmm_string, max_sequences)

    # 2. Create id -> sequence mapping
    sequence_map = parse_hmmer_sequences(hmm_string)

    # 3. Get subset of sequence in order, aligned to the ungapped query sequence
    return create_template_hits(sequence_map, headers, ordered_ids, max_sequences)



