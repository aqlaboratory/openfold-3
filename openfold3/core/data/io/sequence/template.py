"""
This module contains parsers for template alignments.
"""

import re
from typing import Iterable, NamedTuple, Optional, Sequence

from openfold3.core.data.io.sequence.fasta import parse_fasta

"""
Changes from old version:
    - dataclass -> NamedTuple
    - Removed skip_first argument from parse_hmmsearch_sto and parse_hmmsearch_a3m
    - Removed query_sequence and query_indices from parse_hmmsearch_sto, 
    parse_hmmsearch_a3m and TemplateHit
    - set remove_first_row_gaps argument in parse_hmmsearch_sto to True
    - parse_hmmsearch_a3m and parse_hmmsearch_sto now return a dict[int, TemplateHit]
    - index in TemplateHit dict starts from 0 instead of 1
    - replaced e_value None with 0
"""


class HitMetadata(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """

    pdb_id: str
    chain: str
    start: int
    end: int
    length: int
    text: str


class TemplateHit(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """

    index: int
    name: str
    aligned_cols: int
    hit_sequence: str
    indices_hit: list[int]
    e_value: Optional[float]


def _get_indices(sequence: str, start: int) -> list[int]:
    """Returns indices for non-gap/insert residues starting at the given index.

    Args:
        sequence (str): _description_
        start (int): _description_

    Returns:
        list[int]: _description_
    """
    indices = []
    counter = start
    for symbol in sequence:
        # Skip gaps but add a placeholder so that the alignment is preserved.
        if symbol == "-":
            indices.append(-1)
        # Skip deleted residues, but increase the counter.
        elif symbol.islower():
            counter += 1
        # Normal aligned residue. Increase the counter and append to indices.
        else:
            indices.append(counter)
            counter += 1
    return indices


def _parse_hmmsearch_description(description: str) -> HitMetadata:
    """Parses the hmmsearch A3M sequence description line.

    Args:
        description (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        HitMetadata: _description_
    """
    # Example 1: >4pqx_A/2-217 [subseq from] mol:protein length:217  Free text
    # Example 2: >5g3r_A/1-55 [subseq from] mol:protein length:352
    match = re.match(
        r"^>?([a-z0-9]+)_(\w+)/([0-9]+)-([0-9]+).*protein length:([0-9]+) *(.*)$",
        description.strip(),
    )

    if not match:
        raise ValueError(f'Could not parse description: "{description}".')

    return HitMetadata(
        pdb_id=match[1],
        chain=match[2],
        start=int(match[3]),
        end=int(match[4]),
        length=int(match[5]),
        text=match[6],
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
    stockholm_format: str,
    max_sequences: Optional[int] = None,
    remove_first_row_gaps: bool = True,
) -> str:
    """Converts MSA in Stockholm format to the A3M format.

    Args:
        stockholm_format (str): _description_
        max_sequences (Optional[int], optional): _description_. Defaults to None.
        remove_first_row_gaps (bool, optional): _description_. Defaults to True.

    Returns:
        str: _description_
    """
    descriptions = {}
    sequences = {}
    reached_max_sequences = False

    for line in stockholm_format.splitlines():
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

    for line in stockholm_format.splitlines():
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
    """Parses an a3m string produced by hmmsearch.

    Args:
        a3m_string (str): _description_

    Returns:
        dict[int, TemplateHit]: _description_
    """
    # Zip the descriptions and MSAs together, skip the first query sequence.
    parsed_a3m = list(zip(*parse_fasta(a3m_string)))

    hits = {}
    for i, (hit_sequence, hit_description) in enumerate(parsed_a3m):
        if "mol:protein" not in hit_description:
            continue  # Skip non-protein chains.
        metadata = _parse_hmmsearch_description(hit_description)
        # Aligned columns are only the match states.
        aligned_cols = sum([r.isupper() and r != "-" for r in hit_sequence])
        indices_hit = _get_indices(hit_sequence, start=metadata.start - 1)

        hit = TemplateHit(
            index=i,
            name=f"{metadata.pdb_id}_{metadata.chain}",
            aligned_cols=aligned_cols,
            e_value=0,
            hit_sequence=hit_sequence.upper(),
            indices_hit=indices_hit,
        )
        hits[i] = hit

    return hits


def parse_hmmsearch_sto(output_string: str) -> dict[int, TemplateHit]:
    """Gets parsed template hits from the raw string output by the tool.

    Args:
        output_string (str): _description_

    Returns:
        dict[int, TemplateHit]: _description_
    """
    a3m_string = convert_stockholm_to_a3m(output_string)
    template_hits = parse_hmmsearch_a3m(a3m_string=a3m_string)
    return template_hits
