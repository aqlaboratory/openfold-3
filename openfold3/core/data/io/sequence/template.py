"""
Parsers for template alignments.
"""

import re
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np
import pandas as pd

from openfold3.core.data.io.sequence.fasta import parse_fasta
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.core.data.tools.kalign import run_kalign

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
            Row index of the template hit in the alignment.
        entry_id (str):
            Query ID for the query or PDB entry ID for the other template rows.
        chain_id (str):
            Chain ID of the chain. Uses label_asym_id for templates.
        query_ids_hit (np.ndarray):
            Residue indices of the query sequence aligned to the template sequence.
            1-based.
        template_ids_hit (np.ndarray):
            Residue indices of the template sequence aligned to the query sequence.
            1-based. -1 for template positions aligned to gaps in the query.
        template_sequence (str | None):
            The ungapped template sequence.
        e_value (float | None):
            The e-value of the template hit, if available. Defaults to None.
    """

    index: int
    entry_id: str
    chain_id: str
    query_ids_hit: np.ndarray
    template_ids_hit: np.ndarray
    template_sequence: str | None
    e_value: float | None


def parse_hmmer_headers(hmmer_string: str, max_sequences: int) -> pd.DataFrame:
    """Parses the headers from an hmmalign MSA string in Stockholm format.

    Expects the headers to be in the format:
        #=GS <entry id>_<chain id>/<start res id>-<end res id> mol:<moltype>
    where
        <entry id>:
            an arbitrary ID for the first row and PDB ID for all other rows
        <chain id>:
            chain ID
        <start res id>:
            the residue ID of the first residue of the aligned sequence segment in the
            full sequence
        <end res id>:
            the residue ID of the last residue of the aligned sequence segment in the
            full sequence
        <moltype>:
            the molecule type, one of "protein", "dna", "rna"

    Args:
        hmmer_string (str):
            The string containing the HMMER MSA in Stockholm format.
        max_sequences (int):
            The maximum number of sequences to parse from the alignment.

    Returns:
        pd.DataFrame:
             A DataFrame with columns "id", "start", "end", and
            "moltype" for each sequence.
    """
    regex = re.compile(r"^#=GS\s+([^/]+)/(\d+)-(\d+).*?mol:(\w+)", re.MULTILINE)
    matches = [list(match) for match in regex.findall(hmmer_string)]
    headers = pd.DataFrame(
        matches[: min(max_sequences + 1, len(matches))],
        columns=["id", "start", "end", "moltype"],
    )

    return headers


def parse_hmmer_aln_rows(hmmer_string: str) -> dict[str, str]:
    """Parses an hmmalign MSA str in sto format into a row ID-alignment map.

    Args:
        hmmer_string (str):
            The string containing the HMMER MSA in Stockholm format.

    Returns:
        dict[str, str]:
            A dictionary mapping alignment row IDs to their corresponding alignments.
    """
    aln_row_map = {}
    for line in hmmer_string.splitlines():
        # Ignore annotation lines and blank lines
        if not line.strip() or line.startswith("#") or line.startswith("//"):
            continue

        # Split the line into the full ID and the sequence
        full_id, chunk = line.split(maxsplit=1)

        aln_row_map[full_id] = aln_row_map.get(full_id, "") + chunk.strip()

    return aln_row_map


def parse_first_aln_data(
    headers: pd.DataFrame, aln_row_map: dict[str, str], query_seq_str: str
) -> tuple[pd.DataFrame, bool, np.ndarray, np.ndarray]:
    first_header = headers.iloc[0]
    first_id = first_header["id"]

    # query_id should be the full ID like "1a0a_A/1-63" but is misformatted in the
    # snakemake pipeline and only has the base ID like "1a0a_A" so trying both here
    first_id = first_header["id"]
    if first_id not in aln_row_map:
        first_id = (
            first_header["id"] + f"/{first_header['start']}-{first_header['end']}"
        )

    # Get the ungapped query sequence and its mask
    first_aln_str = aln_row_map[first_id]
    first_aln_arr = np.fromiter(first_aln_str, dtype="<U1", count=len(first_aln_str))
    first_aln_nongap_mask = ~np.isin(first_aln_arr, ["-", "."])
    first_seq_arr = first_aln_arr[first_aln_nongap_mask]

    is_first_query = "".join(first_seq_arr) in query_seq_str

    # Check if need to reindex wrt full query sequence if the first row is an exact
    # subsequence of a sequence different from the query sequence
    if is_first_query & (
        (first_header["start"] != 1) | (first_header["end"] != len(query_seq_str))
    ):
        query_seq_arr = np.fromiter(
            query_seq_str, dtype="<U1", count=len(query_seq_str)
        )

        n = query_seq_arr.size
        m = first_seq_arr.size

        # Create a 2D rolling window view of the query array
        shape = (n - m + 1, m)
        strides = (query_seq_arr.strides[0], query_seq_arr.strides[0])
        windows = np.lib.stride_tricks.as_strided(
            query_seq_arr, shape=shape, strides=strides
        )

        # Compare all windows to the subsequence array simultaneously
        matches = windows == first_seq_arr

        # Find the row where all elements match and update indices
        is_match = np.all(matches, axis=1)
        start_index = np.argmax(is_match)
        end_index = start_index + m - 1

        # Prepend correctly indexed query header if not a subsequence of the query
        # sequence itself
        if (first_header["start"] != start_index) | (first_header["end"] != end_index):
            headers = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "id": ["query"],
                            "start": [start_index + 1],
                            "end": [end_index + 1],
                            "moltype": [first_header["moltype"]],
                        }
                    ),
                    headers,
                ],
                ignore_index=True,
            )

    return headers, is_first_query, first_seq_arr, first_aln_nongap_mask


def calculate_ids_hit(
    q: np.ndarray, t: np.ndarray, query_start: int, template_start: int
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the residue correspondences between the full query and template
    sequences.

    Args:
        q (np.ndarray):
            The aligned query sequence as a numpy array of characters.
        t (np.ndarray):
            The aligned template sequence as a numpy array of characters.
        query_start (int):
            The starting index of the aligned query sequence segment in the full query
            sequence.
        template_start (int):
            The starting index of the aligned template sequence segment in the full
            template sequence.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Indices of the query and template residues wrt. the full sequences.
            The indices are 1-based and gaps are represented by -1.
    """

    # 1. Create boolean masks to identify non-gaps
    q_is_residue = q != "-"
    t_is_residue = t != "-"

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


def parse_template_hits_hmmalign(
    query_seq_arr: np.ndarray,
    query_aln_nongap_mask: np.ndarray,
    aln_row_map: dict[str, str],
    headers: pd.DataFrame,
) -> dict[int, TemplateData]:
    """Parses the sequence map and headers to create TemplateHit objects.

    Args:
        query_seq_arr (np.ndarray):
            The aligned query sequence as a numpy array of characters.
        query_aln_nongap_mask (np.ndarray):
            A boolean mask indicating the positions of non-gap residues in the aligned
            query sequence.
        aln_row_map (dict[str, str]):
            A mapping of sequence IDs to their corresponding sequences.
        headers (pd.DataFrame):
            A DataFrame containing the headers with columns "id", "start", "end", and
            "moltype".

    Returns:
        dict[int, TemplateHit]:
            A dictionary mapping row indices to TemplateHit objects, containing the
            aligned sequences and their indices in the ungapped query sequence.
    """
    query_moltype = MoleculeType[headers.iloc[0]["moltype"].upper()]
    query_start = int(headers.iloc[0]["start"])

    # Iterate over the remaining sequences up to max_sequences
    templates = {}
    for idx, row in headers.iterrows():
        # Skip the first sequence (query) and check molecule type
        if (idx == 0) | (MoleculeType[row["moltype"].upper()] != query_moltype):
            continue

        seq_id = row["id"] + f"/{row['start']}-{row['end']}"
        if seq_id in aln_row_map:
            template_aln_str = aln_row_map[seq_id]
            template_aln_arr = np.fromiter(
                template_aln_str, dtype="<U1", count=len(template_aln_str)
            )
            # np array of the template aligned to the ungapped query sequence
            template_aln_nongap_arr = template_aln_arr[query_aln_nongap_mask]
            # str of the ungapped template sequence
            template_seq_str = "".join(
                template_aln_arr[~np.isin(template_aln_arr, ["-", "."])]
            ).upper()

            # Get residue correspondences wrt. the full-length sequences
            query_ids_hit, template_ids_hit = calculate_ids_hit(
                q=query_seq_arr,
                t=template_aln_nongap_arr,
                query_start=query_start,
                template_start=int(row["start"]),
            )

            entry_id, chain_id = row["id"].split("_")
            template_hit = TemplateData(
                index=idx,
                entry_id=entry_id,
                chain_id=chain_id,
                query_ids_hit=query_ids_hit,
                template_ids_hit=template_ids_hit,
                template_sequence=template_seq_str,
                e_value=None,
            )
            templates[idx] = template_hit

    return templates


def parse_template_hits_hmmsearch(
    query_seq_str: np.ndarray,
    aln_row_map: dict[str, str],
    headers: pd.DataFrame,
) -> dict[int, TemplateData]:
    # Collect sequences
    all_sequences = f">query\n{query_seq_str}\n"

    for _, row in headers.iterrows():
        full_id = row["id"] + f"/{row['start']}-{row['end']}"
        if full_id in aln_row_map:
            all_sequences += ">{}\n{}\n".format(
                full_id, aln_row_map[full_id].replace(".", "").replace("-", "")
            )

    # Realign with kalign to the query sequence
    alignments, _ = parse_fasta(run_kalign(all_sequences))

    # Process query
    query_aln_str = alignments[0]
    query_aln_arr = np.fromiter(query_aln_str, dtype="<U1", count=len(query_aln_str))
    query_aln_nongap_mask = ~np.isin(query_aln_arr, ["-", "."])
    query_seq_arr = query_aln_arr[query_aln_nongap_mask]

    templates = {}
    for template_aln_str, (idx, row) in zip(alignments[1:], headers.iterrows()):
        template_aln_arr = np.fromiter(
            template_aln_str, dtype="<U1", count=len(template_aln_str)
        )

        # np array of the template aligned to the ungapped query sequence
        template_aln_nongap_arr = template_aln_arr[query_aln_nongap_mask]
        # str of the ungapped template sequence
        template_seq_str = "".join(
            template_aln_arr[~np.isin(template_aln_arr, ["-", "."])]
        ).upper()

        # Get residue correspondences wrt. the full-length sequences
        query_ids_hit, template_ids_hit = calculate_ids_hit(
            q=query_seq_arr,
            t=template_aln_nongap_arr,
            query_start=1,  # kalign does global alignment
            template_start=int(row["start"]),
        )

        entry_id, chain_id = row["id"].split("_")
        template_hit = TemplateData(
            index=idx,
            entry_id=entry_id,
            chain_id=chain_id,
            query_ids_hit=query_ids_hit,
            template_ids_hit=template_ids_hit,
            template_sequence=template_seq_str,
            e_value=None,
        )
        templates[idx] = template_hit

    return templates


def parse_template_hits_hmmer_sto(
    hmmer_string: str, max_sequences: int, query_seq_str: str
) -> dict[int, TemplateData]:
    """Parses template data from an HMMER Stockholm formatted string.

    It can handle both hmmalign and hmmsearch sto outputs depending on whether the first
    alignment row is a sequence segment from query sequence. In the latter, hmmsearch
    case (first alignment row in not from the query), it realigns each template hit to
    the query using kalign and hence results in slower runtime.

    Args:
        hmmer_string (str):
            The string containing the HMMER alignment in Stockholm format.
        max_sequences (int):
            The maximum number of sequences to parse from the alignment.
        query_seq_str (str):
            The query sequence string to check against the first alignment row.

    Returns:
        dict[int, TemplateData]: _description_
    """

    # 1. Parse headers from format "#=GS id/start-end moltype"
    headers = parse_hmmer_headers(hmmer_string, max_sequences)

    # 2. Create id -> sequence mapping
    aln_row_map = parse_hmmer_aln_rows(hmmer_string)

    # 3. Get data about the first alignment row
    headers, is_first_query, first_seq_arr, first_aln_nongap_mask = (
        parse_first_aln_data(headers, aln_row_map, query_seq_str)
    )

    if is_first_query:
        return parse_template_hits_hmmalign(
            query_seq_arr=first_seq_arr,
            query_aln_nongap_mask=first_aln_nongap_mask,
            aln_row_map=aln_row_map,
            headers=headers,
        )

    else:
        return parse_template_hits_hmmsearch(
            query_seq_str=query_seq_str,
            aln_row_map=aln_row_map,
            headers=headers,
        )


def parse_template_hits_a3m(
    a3m_string: str, max_sequences: int, query_seq_str: str
) -> dict[int, TemplateData]:
    # Parse a3m string
    alignments, headers = parse_fasta(a3m_string)

    # Subset assuming the first sequence is the query sequence
    n_sequences = min(max_sequences + 1, len(alignments))
    alignments = alignments[:n_sequences]
    headers = headers[:n_sequences]

    # Parse headers into dataframe from format ID/start-end
    for idx, i in enumerate(headers):
        entry_id, start_end = i.split("/")
        chain_start, chain_end = start_end.split("-")
        headers[idx] = (entry_id, chain_start, chain_end, MoleculeType.PROTEIN.name)
    headers = pd.DataFrame(headers, columns=["id", "start", "end", "moltype"])

    # Check if the first sequence is the query sequence
    first_aln_str = alignments[0]
    first_aln_arr = np.fromiter(first_aln_str, dtype="<U1", count=len(first_aln_str))
    first_aln_nongap_mask = ~np.isin(first_aln_arr, ["-", "."])
    first_seq_arr = first_aln_arr[first_aln_nongap_mask]
    is_first_query = "".join(first_seq_arr) in query_seq_str

    # Realign with kalign if not
    if not is_first_query:
        # Realign with kalign to the query sequence
        all_sequences = f">query\n{query_seq_str}\n"
        for seq_id, seq in zip(headers["id"], alignments[1:]):
            all_sequences += f">{seq_id}\n{seq.replace('.', '').replace('-', '')}\n"

        alignments, _ = parse_fasta(run_kalign(all_sequences))
    else:
        # Drop extra row
        headers = headers.iloc[:-1]
        alignments = alignments[:-1]

    # Process query
    query_aln_str = alignments[0]
    query_aln_arr = np.fromiter(query_aln_str, dtype="<U1", count=len(query_aln_str))
    query_aln_nongap_mask = ~np.isin(query_aln_arr, ["-", "."])
    query_seq_arr = query_aln_arr[query_aln_nongap_mask]

    templates = {}
    for template_aln_str, (idx, row) in zip(
        alignments[1:], headers.iloc[:-1].iterrows()
    ):
        template_aln_arr = np.fromiter(
            template_aln_str, dtype="<U1", count=len(template_aln_str)
        )

        # np array of the template aligned to the ungapped query sequence
        template_aln_nongap_arr = template_aln_arr[query_aln_nongap_mask]
        # str of the ungapped template sequence
        template_seq_str = "".join(
            template_aln_arr[~np.isin(template_aln_arr, ["-", "."])]
        ).upper()

        # Get residue correspondences wrt. the full-length sequences
        query_ids_hit, template_ids_hit = calculate_ids_hit(
            q=query_seq_arr,
            t=template_aln_nongap_arr,
            query_start=1,
            template_start=int(row["start"]),
        )

        entry_id, chain_id = row["id"].split("_")
        template_hit = TemplateData(
            index=idx,
            entry_id=entry_id,
            chain_id=chain_id,
            query_ids_hit=query_ids_hit,
            template_ids_hit=template_ids_hit,
            template_sequence=template_seq_str,
            e_value=None,
        )
        templates[idx] = template_hit

    return templates


def parse_template_hits_m8(
    m8_string: str, max_sequences: int
) -> dict[int, TemplateData]:
    # has cigar string -> use for residue correspondences
    # does not have cigar string -> use kalign for residue correspondences
    pass
