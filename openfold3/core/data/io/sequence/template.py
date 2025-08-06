"""
Parsers for template alignments.
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
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
        sequence_identity (float):
            Sequence identity of the template hit with respect to the query sequence.
            Calculated as the number of identical residues in the alignment divided by
            the number of query residues in the alignment.
        q_cov (float | None):
            Coverage of the full query sequence in the template alignment.
        template_sequence (str | None):
            The ungapped template sequence.
    """

    index: int
    entry_id: str
    chain_id: str
    query_aln_pos: np.ndarray | None
    aln_pos: np.ndarray | None
    seq_id: float
    q_cov: float | None
    seq: str | None


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


def calculate_ids_hit_cigar(
    cigar_string: str, query_start: int, template_start: int, gap_char: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts a CIGAR string into 1D numpy arrays mapping alignment positions
    to full-sequence coordinates for the query and template.

    Args:
        cigar_string (str):
            The CIGAR string representing the alignment.
        query_start (int):
            The 0-indexed start position of the alignment
                           in the full query sequence.
        template_start (int):
            The 0-indexed start position of the alignment
                              in the full template sequence.
        gap_char (int):
            The integer value used to represent gaps in the
                        output index arrays. Defaults to -1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two 1D NumPy arrays:
        - query_indices: Indices of aligned query residues.
        - template_indices: Indices of aligned template residues.
    """
    cigar_ops = re.findall(r"(\d+)([MIDNSHP=X])", cigar_string)

    if not cigar_ops:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Filter for ops that create alignment columns
    op_details = [(int(length), op) for length, op in cigar_ops if op in "MIDN=X"]

    if not op_details:
        return np.array([], dtype=int), np.array([], dtype=int)

    lengths, ops = zip(*op_details)
    lengths = np.array(lengths, dtype=int)

    def ops_to_idx(ops, lengths, start, gap_char, aln_ops):
        # 1: consuming a base, 0: gap
        op_deltas = np.array([int(op in aln_ops) for op in ops], dtype=np.int8)
        # Expand deltas for each aln position
        deltas = np.repeat(op_deltas, lengths)
        # cumsum -> idxs
        indices = np.cumsum(deltas) + start - 1
        # Mark gaps
        indices[deltas == 0] = gap_char
        return indices

    q_indices = ops_to_idx(ops, lengths, query_start, gap_char, "M=XI")
    t_indices = ops_to_idx(ops, lengths, template_start, gap_char, "M=XDN")

    return q_indices, t_indices


class TemplateParser(ABC):
    def __init__(self, max_sequences: int):
        self.max_sequences = max_sequences

    @abstractmethod
    def __call__(
        self,
        alignment_source: str | pd.DataFrame,
        query_seq_str: str,
        query_entry_id: str | None = None,
        query_chain_id: str | None = None,
    ) -> dict[int, TemplateData]:
        """Main entry point to parse an alignment source."""
        raise NotImplementedError

    def _process_alignment_hits(
        self,
        query_seq_str: str,
        query_aln_str: str,
        template_alignments: list[str],
        headers: pd.DataFrame,
        query_start_idx: int = 1,
    ) -> dict[int, TemplateData]:
        templates = {}
        query_aln_arr = np.fromiter(
            query_aln_str, dtype="<U1", count=len(query_aln_str)
        )

        for template_aln_str, (_, row) in zip(template_alignments, headers.iterrows()):
            template_aln_arr = np.fromiter(
                template_aln_str, dtype="<U1", count=len(template_aln_str)
            )
            template_gap_mask = ~np.isin(template_aln_arr, ["-", "."])
            template_seq_str = "".join(template_aln_arr[template_gap_mask]).upper()

            query_ids_hit, template_ids_hit = calculate_ids_hit(
                q=query_aln_arr,
                t=template_aln_arr,
                query_start=query_start_idx,
                template_start=int(row["start"]),
            )
            query_gap_mask = ~np.isin(query_aln_arr, ["-", "."])
            seq_id = sum((template_aln_arr == query_aln_arr)[query_gap_mask]) / sum(
                query_gap_mask
            )
            q_cov = sum(query_gap_mask & template_gap_mask) / len(query_seq_str)

            entry_id, chain_id = row["id"].split("_")
            templates[row.name] = TemplateData(
                index=row.name,
                entry_id=entry_id,
                chain_id=chain_id,
                query_aln_pos=query_ids_hit,
                aln_pos=template_ids_hit,
                seq_id=seq_id,
                q_cov=q_cov,
                seq=template_seq_str,
            )
        return templates


class StoParser(TemplateParser):
    """Parses HMMER Stockholm format (.sto) files."""

    def _parse_headers(self, hmmer_string: str) -> pd.DataFrame:
        regex = re.compile(r"^#=GS\s+([^/]+)/(\d+)-(\d+).*?mol:(\w+)", re.MULTILINE)
        matches = [list(match) for match in regex.findall(hmmer_string)]
        if self.max_sequences is None:
            max_sequences = len(matches)
        else:
            max_sequences = min(self.max_sequences + 1, len(matches))
        return pd.DataFrame(
            matches[:max_sequences],
            columns=["id", "start", "end", "moltype"],
        )

    def _parse_aln_rows(self, hmmer_string: str) -> dict[str, str]:
        aln_row_map = {}
        for line in hmmer_string.splitlines():
            if not line.strip() or line.startswith(("#", "//")):
                continue
            full_id, chunk = line.split(maxsplit=1)
            aln_row_map[full_id] = aln_row_map.get(full_id, "") + chunk.strip()
        return aln_row_map

    def __call__(
        self, alignment_source: str, query_seq_str: str
    ) -> dict[int, TemplateData]:
        headers = self._parse_headers(alignment_source)
        aln_row_map = self._parse_aln_rows(alignment_source)

        first_header = headers.iloc[0]
        first_id_base = first_header["id"]
        first_id_full = f"{first_id_base}/{first_header['start']}-{first_header['end']}"
        first_aln_str = aln_row_map.get(first_id_full) or aln_row_map.get(first_id_base)

        first_seq_str = first_aln_str.replace("-", "").replace(".", "").upper()
        is_first_query = first_seq_str in query_seq_str

        if is_first_query:
            query_start_idx = int(first_header["start"])
            # It is possible that the aligned subsequence in the first row is identical
            # to a subsequence in the query sequence, but is from a different full
            # sequence, in which case we need to reindex the start position
            if query_start_idx != 1:
                query_start_idx = query_seq_str.find(first_seq_str) + 1
            template_alignments = [
                aln_row_map.get(f"{row['id']}/{row['start']}-{row['end']}")
                or aln_row_map.get(first_id_base)
                for _, row in headers.iterrows()
            ]
            return self._process_alignment_hits(
                query_seq_str=query_seq_str,
                query_aln_str=first_aln_str,
                template_alignments=template_alignments,
                headers=headers,
                query_start_idx=query_start_idx,
            )
        else:
            all_sequences = f">query\n{query_seq_str}\n"
            for _, row in headers.iterrows():
                full_id = f"{row['id']}/{row['start']}-{row['end']}"
                ungapped_seq = aln_row_map[full_id].replace(".", "").replace("-", "")
                all_sequences += f">{full_id}\n{ungapped_seq}\n"

            realigned_str = run_kalign(all_sequences)
            alignments, _ = parse_fasta(realigned_str)

            return self._process_alignment_hits(
                query_seq_str=query_seq_str,
                query_aln_str=alignments[0],
                template_alignments=alignments[1:],
                headers=headers,
            )


class A3mParser(TemplateParser):
    """Parses A3M format files."""

    def __call__(
        self,
        alignment_source: str,
        query_seq_str: str,
        realign: bool = False,
    ) -> dict[int, TemplateData]:
        # 1. Parse the A3M file as a FASTA file
        alignments, headers_raw = parse_fasta(alignment_source)

        # 2. Subset to max_sequences
        if self.max_sequences is None:
            max_sequences = len(alignments)
        else:
            max_sequences = min(self.max_sequences + 1, len(alignments))
        alignments = alignments[:max_sequences]
        headers_raw = headers_raw[:max_sequences]

        # 3. Process A3M-specific headers into a DataFrame
        header_data = []
        for h in headers_raw:
            entry_id, start_end = h.split("/")
            start, end = start_end.split("-")
            header_data.append((entry_id, start, end, MoleculeType.PROTEIN.name))

        headers = pd.DataFrame(header_data, columns=["id", "start", "end", "moltype"])

        # 4. Check if the first sequence is the query
        first_aln_str = alignments[0]
        first_seq_str = "".join(c for c in first_aln_str if c not in ["-", "."]).upper()
        is_first_query = first_seq_str in query_seq_str

        # 5. Process alignments
        if is_first_query and not realign:
            # Use existing alignment if query is first
            return self._process_alignment_hits(
                query_seq_str=query_seq_str,
                query_aln_str=alignments[0],
                template_alignments=alignments,
                headers=headers,
            )
        else:
            # Realign with kalign if query is not first or realign is requested
            all_sequences = f">query\n{query_seq_str}\n"
            for header, seq in zip(headers_raw, alignments):
                ungapped_seq = "".join(c for c in seq if c.isupper())
                all_sequences += f">{header}\n{ungapped_seq}\n"

            realigned_str = run_kalign(all_sequences)
            realigned_alignments, _ = parse_fasta(realigned_str)

            return self._process_alignment_hits(
                query_seq_str=query_seq_str,
                query_aln_str=realigned_alignments[0],
                template_alignments=realigned_alignments[1:],
                headers=headers,  # Use original headers for metadata
            )


class M8Parser(TemplateParser):
    """Parses tabular .m8 file format.

    See the BLAST m8 section here for details of the expected format:
    https://linsalrob.github.io/ComputationalGenomicsManual/SequenceFileFormats/
    """

    def __call__(
        self, alignment_source: pd.DataFrame, query_seq_str: str
    ) -> dict[int, TemplateData]:
        columns = [
            "query_id",
            "template_id",
            "seq_identity",
            "aln_len",
            "n_gaps",
            "n_mismatches",
            "query_start",
            "query_end",
            "template_start",
            "template_end",
            "e_value",
            "bit_score",
        ]
        if len(alignment_source.columns) == 12:
            alignment_source.columns = columns
        elif len(alignment_source.columns) == 13:
            alignment_source.columns = columns + ["cigar"]

        df = alignment_source.sort_values("e_value", ignore_index=True)
        if self.max_sequences is None:
            max_sequences = len(df)
        else:
            max_sequences = min(self.max_sequences + 1, len(df))
        df = df.iloc[:max_sequences]
        df[["entry_id", "chain_id"]] = df["template_id"].str.split("_", expand=True)

        templates = {}
        for idx, row in df.iterrows():
            query_ids_hit, template_ids_hit, q_cov = None, None, None
            # For now, the cigar string is not going to be used, it is way too annoying
            # to do the validity checks with it.

            # if "cigar" in row and pd.notna(row["cigar"]):
            #     query_ids_hit, template_ids_hit = calculate_ids_hit_cigar(
            #         cigar_string=row["cigar"],
            #         query_start=row["query_start"],
            #         template_start=row["template_start"],
            #     )
            #     q_cov = sum((query_ids_hit != -1) & (template_ids_hit != -1)) /
            # len(query_seq_str)

            templates[idx] = TemplateData(
                index=idx,
                entry_id=row["entry_id"],
                chain_id=row["chain_id"],
                query_aln_pos=query_ids_hit,
                aln_pos=template_ids_hit,
                seq_id=row["seq_identity"],
                seq=None,
                q_cov=q_cov,
            )
        return templates


TEMPLATE_PARSER_REGISTRY = {".a3m": A3mParser, ".sto": StoParser, ".m8": M8Parser}


def parse_template_alignment(
    aln_path: Path, query_seq_str: str, max_sequences: int
) -> dict[int, TemplateData]:
    """Parses a template alignment file.

    Args:
        aln_path (Path):
            Path to the template alignment file.
        query_seq_str (str):
            The query sequence string to align the templates to.
        max_sequences (int):
            The maximum number of sequences to include in the output.

    Raises:
        ValueError:
            If the template alignment file format is not supported.

    Returns:
        dict[int, TemplateData]:
            Dictionary mapping the index of the hit in the alignment to the parsed
            template hit.
    """
    _, ext = aln_path.stem, aln_path.suffix
    if ext not in TEMPLATE_PARSER_REGISTRY:
        raise ValueError(f"Unsupported template alignment file format: {ext}")
    aln_parser = TEMPLATE_PARSER_REGISTRY[ext](max_sequences=max_sequences)
    if ext == ".m8":
        parser_input = {
            "alignment_source": pd.read_csv(aln_path, sep="\t", header=None),
            "query_seq_str": query_seq_str,
        }
    else:
        with open(aln_path.absolute()) as f:
            parser_input = {
                "alignment_source": f.read(),
                "query_seq_str": query_seq_str,
            }
    return aln_parser(**parser_input)
