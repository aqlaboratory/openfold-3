#!/usr/bin/env python3
"""
Pipeline:

1) mmseqs createdb Rfam.fa rfamDB
2) mmseqs cluster rfamDB rfamDB_clu tmp --min-seq-id 0.9 -c 0.8 --cov-mode 1
3) mmseqs createtsv rfamDB rfamDB rfamDB_clu rfamDB_clu.tsv
4) awk '{n[$1]++} END {for (c in n) if (n[c] >= 3) print c}' rfamDB_clu.tsv 
    > reps_ge3.ids
5) mmseqs createsubdb rfamDB_clu rfamDB rfamDB_clu_rep
6) mmseqs convert2fasta rfamDB_clu_rep rfamDB_clu_rep.fasta
7) awk 'NR==FNR {ids[$1]; next} /^>/ {id=substr($1,2); keep=(id in ids)} keep' \
       reps_ge3.ids rfamDB_clu_rep.fasta > Rfam_reps_ge3.fa
8) awk (len > 10) Rfam_reps_ge3.fa > Rfam_reps_ge3_lenGT10.fa

This script replicates that logic in Python, with mmseqs calls + Python filters.
"""

import subprocess  # run external mmseqs commands
import sys  # command-line args, stderr printing
from collections import Counter  # count cluster sizes
from pathlib import Path  # nicer path handling


def run(cmd: list[str]) -> None:
    """Run a shell command, echoing it first, and stop on error."""
    print("[RUN]", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def count_clusters(tsv_path: Path, min_size: int, ids_out_path: Path) -> int:
    """
    Parse rfamDB_clu.tsv and write representative IDs for clusters
    with size >= min_size to ids_out_path.

    Shell equivalent:
      awk '{n[$1]++} END {for (c in n) if (n[c] >= 3) print c}' rfamDB_clu.tsv
      > reps_ge3.ids
    """
    counts = Counter()

    # Read the cluster TSV line by line
    with tsv_path.open() as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines
            rep_id = line.split("\t", 1)[0]  # column 1 (cluster representative)
            counts[rep_id] += 1  # increment cluster size

    kept = 0
    # Write representatives of clusters with size >= min_size
    with ids_out_path.open("w") as out:
        for rep_id, size in counts.items():
            if size >= min_size:
                out.write(rep_id + "\n")
                kept += 1

    print(f"[INFO] clusters with size >= {min_size}: {kept}", file=sys.stderr)
    return kept


def filter_fasta_by_ids(ids_path: Path, fasta_in: Path, fasta_out: Path) -> None:
    """
    Keep only records whose header ID is in ids_path.

    Shell equivalent:
      awk 'NR==FNR {ids[$1]; next} /^>/ {id=substr($1,2); keep=(id in ids)} keep' \
          reps_ge3.ids rfamDB_clu_rep.fasta > Rfam_reps_ge3.fa
    """
    # Load the allowed IDs (one per line)
    keep_ids = {line.strip() for line in ids_path.open() if line.strip()}
    print(f"[INFO] loaded {len(keep_ids):,} IDs to keep", file=sys.stderr)

    with fasta_in.open() as fin, fasta_out.open("w") as fout:
        keep = False  # whether we are currently inside a record to keep

        for line in fin:
            if line.startswith(">"):
                # Extract ID = first token in header (after '>')
                header_id = line[1:].split()[0]
                keep = header_id in keep_ids
            if keep:
                # Write header + all subsequent sequence lines until next header
                fout.write(line)


def filter_fasta_by_length(
    fasta_in: Path,
    fasta_out: Path,
    min_len: int = 10,
) -> None:
    """
    Keep only sequences with total length > min_len (across all lines).

    Shell equivalent: the long awk block that accumulates seqlen per record.
    """
    with fasta_in.open() as fin, fasta_out.open("w") as fout:
        header = None  # current header line
        seq_lines: list[str] = []  # all sequence lines for this record

        def flush_record() -> None:
            """Write the current record if its length > min_len."""
            if header is None:
                return
            # Compute sequence length ignoring whitespace/newlines
            seqlen = sum(len(s.strip()) for s in seq_lines)
            if seqlen > min_len:
                fout.write(header)
                for s in seq_lines:
                    fout.write(s)

        for line in fin:
            if line.startswith(">"):
                # We hit a new header: flush the previous record
                flush_record()
                header = line  # start new record
                seq_lines = []  # reset sequence buffer
            else:
                # Sequence line: keep raw to preserve formatting
                seq_lines.append(line)

        # Flush the last record at EOF
        flush_record()


def main() -> None:
    # Require exactly one argument: the input FASTA file
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} Rfam.fa", file=sys.stderr)
        sys.exit(1)

    # Resolve the FASTA path
    fasta_path = Path(sys.argv[1]).resolve()

    # Base name (e.g. "Rfam" from "Rfam.fa")
    base = fasta_path.stem

    # MMseqs DB names derived from the base
    db_prefix = base + "DB"  # rfamDB
    db = Path(db_prefix)  # rfamDB
    clu = Path(db_prefix + "_clu")  # rfamDB_clu

    # Temp directory for mmseqs cluster
    tmp_dir = Path("tmp")
    tmp_dir.mkdir(exist_ok=True)

    # File names for various intermediate and final outputs
    clu_tsv = Path(str(clu) + ".tsv")  # rfamDB_clu.tsv
    reps_ids = Path("reps_ge3.ids")  # reps_ge3.ids
    subdb = Path(db_prefix + "_clu_rep")  # rfamDB_clu_rep (DB)
    subdb_fa = Path(str(subdb) + ".fasta")  # rfamDB_clu_rep.fasta

    reps_ge3_fa = Path(f"{base}_reps_ge3.fa")  # Rfam_reps_ge3.fa
    reps_ge3_len_fa = Path(f"{base}_reps_ge3_lenGT10.fa")  # Rfam_reps_ge3_lenGT10.fa

    # 1) mmseqs createdb Rfam.fa rfamDB
    run(["mmseqs", "createdb", str(fasta_path), str(db)])

    # 2) mmseqs cluster rfamDB rfamDB_clu tmp --min-seq-id 0.9 -c 0.8 --cov-mode 1
    run(
        [
            "mmseqs",
            "cluster",
            str(db),
            str(clu),
            str(tmp_dir),
            "--min-seq-id",
            "0.9",
            "-c",
            "0.8",
            "--cov-mode",
            "1",
        ]
    )

    # 3) mmseqs createtsv rfamDB rfamDB rfamDB_clu rfamDB_clu.tsv
    run(["mmseqs", "createtsv", str(db), str(db), str(clu), str(clu_tsv)])

    # 4) Compute cluster sizes and write reps_ge3.ids (clusters with >=3 members)
    count_clusters(clu_tsv, min_size=3, ids_out_path=reps_ids)

    # 5) mmseqs createsubdb rfamDB_clu rfamDB rfamDB_clu_rep
    run(["mmseqs", "createsubdb", str(clu), str(db), str(subdb)])

    # 6) mmseqs convert2fasta rfamDB_clu_rep rfamDB_clu_rep.fasta
    run(["mmseqs", "convert2fasta", str(subdb), str(subdb_fa)])

    # 7) Filter FASTA by representatives from clusters >=3 (reps_ge3.ids)
    filter_fasta_by_ids(reps_ids, subdb_fa, reps_ge3_fa)

    # 8) Filter again by sequence length > 10 nucleotides
    filter_fasta_by_length(reps_ge3_fa, reps_ge3_len_fa, min_len=10)

    # Final status messages
    print(f"[DONE] reps from clusters >=3 written to: {reps_ge3_fa}", file=sys.stderr)
    print(f"[DONE] length>10 subset written to: {reps_ge3_len_fa}", file=sys.stderr)


if __name__ == "__main__":
    main()
