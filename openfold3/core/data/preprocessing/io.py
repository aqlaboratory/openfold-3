import os
from collections import OrderedDict

from openfold3.core.data.preprocessing.msa_primitives import Msa


def parse_stockholm(stockholm_string: str) -> Msa:
    """Parses sequences and deletion matrix from stockholm format alignment.

    Args:
        stockholm_string: The string contents of a stockholm file. The first
            sequence in the file should be the query sequence.

    Returns:
        A tuple of:
            * A list of sequences that have been aligned to the query. These
                might contain duplicates.
            * The deletion matrix for the alignment as a list of lists. The element
                at `deletion_matrix[i][j]` is the number of residues deleted from
                the aligned sequence i at residue position j.
            * The names of the targets matched, including the jackhmmer subsequence
                suffix.
    """
    name_to_sequence = OrderedDict()
    for line in stockholm_string.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        print(line)
        print(line.split())
        name, sequence = line.split()
        if name not in name_to_sequence:
            name_to_sequence[name] = ""
        name_to_sequence[name] += sequence

    msa = []
    deletion_matrix = []

    query = ""
    keep_columns = []
    for seq_index, sequence in enumerate(name_to_sequence.values()):
        if seq_index == 0:
            # Gather the columns with gaps from the query
            query = sequence
            keep_columns = [i for i, res in enumerate(query) if res != "-"]

        # Remove the columns with gaps in the query from all sequences.
        aligned_sequence = "".join([sequence[c] for c in keep_columns])

        msa.append(aligned_sequence)

        # Count the number of deletions w.r.t. query.
        deletion_vec = []
        deletion_count = 0
        for seq_res, query_res in zip(sequence, query):
            if seq_res != "-" or query_res != "-":
                if query_res == "-":
                    deletion_count += 1
                else:
                    deletion_vec.append(deletion_count)
                    deletion_count = 0
        deletion_matrix.append(deletion_vec)

    return Msa(
        sequences=msa,
        deletion_matrix=deletion_matrix,
        descriptions=list(name_to_sequence.keys()),
    )


def parse_stockholm_file(alignment_dir: str, stockholm_file: str):
    path = os.path.join(alignment_dir, stockholm_file)
    file_name,_ = os.path.splitext(stockholm_file)
    with open(path, "r") as infile:
        msa = parse_stockholm(infile.read())
        infile.close()
    return {file_name: msa}


def _keep_line(line: str, seqnames: set[str]) -> bool:
    """Function to decide which lines to keep."""
    if not line.strip():
        return True
    if line.strip() == "//":  # End tag
        return True
    if line.startswith("# STOCKHOLM"):  # Start tag
        return True
    if line.startswith("#=GC RF"):  # Reference Annotation Line
        return True
    if line[:4] == "#=GS":  # Description lines - keep if sequence in list.
        _, seqname, _ = line.split(maxsplit=2)
        return seqname in seqnames
    elif line.startswith("#"):  # Other markup - filter out
        return False
    else:  # Alignment data - keep if sequence in list.
        seqname = line.partition(" ")[0]
        return seqname in seqnames


def truncate_stockholm_msa(stockholm_msa_path: str, max_sequences: int) -> str:
    """Reads + truncates a Stockholm file while preventing excessive RAM usage."""
    seqnames = set()
    filtered_lines = []

    with open(stockholm_msa_path) as f:
        for line in f:
            if line.strip() and not line.startswith(("#", "//")):
                # Ignore blank lines, markup and end symbols - remainder are alignment
                # sequence parts.
                seqname = line.partition(" ")[0]
                seqnames.add(seqname)
                if len(seqnames) >= max_sequences:
                    break

        f.seek(0)
        for line in f:
            if _keep_line(line, seqnames):
                filtered_lines.append(line)

    return "".join(filtered_lines)
