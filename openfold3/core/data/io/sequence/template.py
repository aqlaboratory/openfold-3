"""
Parsers for template alignments.
"""

from collections.abc import Iterable, Sequence
from typing import NamedTuple

from openfold3.core.data.io.sequence.fasta import parse_fasta

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
    pdb_id, chain_id = pdb_chain_id.split("_")
    if index == 0:
        start_index = 1
    else:
        start_index = int(desc.split(" ")[0].split("-")[0])

    return HitMetadata(
        pdb_id=pdb_id,
        chain=chain_id,
        start=start_index,
    )


def _convert_sto_seq_to_a3m(
    query_non_gaps: Sequence[bool], sto_seq: str
) -> Iterable[str]:
    """_summary_

    Args:
        query_non_gaps (Sequence[bool]): _description_
        sto_seq (str): _description_

    Returns:
        Iterable[str]: _description_

    Yields:
        Iterator[Iterable[str]]: _description_
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
