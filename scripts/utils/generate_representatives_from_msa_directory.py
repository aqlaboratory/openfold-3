# %%
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import tqdm
from Bio import AlignIO, SeqIO

parser = ArgumentParser(
    usage="""
    Script for generating a representatives fasta from
    an output MSA directory from the snakemake pipeline
    usage:
    python generate_representatives_from_msa_directory.py \
        --msa-directory <path/to/aln/dir> \
        --out-fasta <path/to/aln/dir> \
        --protein-dbs "db1,db2,...,dbn" \
        --rna-dbs "db1,db2...dbn"
    
    Only one of --protein-dbs or --rna-dbs is needed
    
    This does not produce a de-duplicated fasta, 
    to avoid reparsing some redundant sequences
    """
)
parser.add_argument(
    "--msa-directory",
    help="Path to the directory containing MSA files.",
    type=str,
    required=True,
)

parser.add_argument(
    "--out-fasta",
    help="output fasta to write representatives to",
    type=str,
    required=True,
)
parser.add_argument(
    "--existing-fasta",
    help="Path to an existing fasta to exclude sequences from. "
    "Useful for reprocessing an existing dir",
    type=str,
    required=False,
)

parser.add_argument(
    "--protein-dbs",
    help="""what protein databases to check exist and pull query sequences from
    Must be a string formatted as a comma separated list""",
    type=str,
    default="uniref90,uniprot,cfdb,mgnify,hmm_output",
)

parser.add_argument(
    "--rna-dbs",
    help="""what RNA databases to check exist and pull query sequences from.
    string formatted as a comma separated list """,
    type=str,
    default="nucleotide_collection,rfam,rnacentral",
)

parser.add_argument(
    "--ncores",
    help="number of cores to use for parallel processing",
    type=int,
    default=1,
)

## common files that we can always ignore
BLACKLISTED_FILES = set(["VALID_DIR", "query.hmm", "query.sto", "EMPTY_TEMPLATE_OK"])


def repair_sto(file):
    with open(file) as infl:
        lines = infl.readlines()
    for i, line in enumerate(lines):  # noqa: B007
        if line == "# STOCKHOLM 1.0\n":
            break
    repaired_lines = lines[i:]
    with open(file, "w") as outfl:
        outfl.writelines(repaired_lines)
    return


def get_queryseq_from_file(file: Path):
    try:
        ext = file.suffix
        if ext == ".a3m":
            with open(file) as infl:
                lines = [line.strip() for line in infl]
            return lines[1]
        elif ext == ".sto":
            ## biopython does not like stockholm files that
            ## don't start with the # STOCKHOLM 1.0 header
            ## so wrap an attempt to repair the file
            try:
                alignment = AlignIO.read(str(file), "stockholm")
            except Exception:
                repair_sto(file)
                alignment = AlignIO.read(str(file), "stockholm")
            query = alignment[0]
            return str(query.seq).replace("-", "").upper()
        else:
            raise ValueError(
                f"Only alignment outputs in .sto,.a3m accepted, found {file}"
            )
    except Exception:
        return None


def wrap_extract(aln_dir, protein_dbs, rna_dbs):
    status = "PASS"
    reason = None
    alignments = list(aln_dir.iterdir())
    if len(alignments) == 0:
        status = "FAIL"
        reason = "No alignment files found"
        return None, (aln_dir, status, reason)
    alignments = [aln for aln in alignments if aln.name not in BLACKLISTED_FILES]
    detected_aln_dbs = set(
        [
            f.stem.replace("_hits", "")
            for f in alignments
            if f.suffix in [".a3m", ".sto"]
        ]
    )
    matched_protein_dbs = protein_dbs.intersection(detected_aln_dbs)
    matched_rna_dbs = rna_dbs.intersection(detected_aln_dbs)
    if (len(matched_protein_dbs) == 0) and (len(matched_rna_dbs) == 0):
        status = "FAIL"
        reason = "No expected alignment databases found"
        return None, (aln_dir, status, reason)

    ## too many DBs
    if (len(matched_protein_dbs) > len(protein_dbs)) or (
        len(matched_rna_dbs) > len(rna_dbs)
    ):
        status = "FAIL"
        reason = "Too many DBs detected"
        return None, (aln_dir, status, reason)

    ## missing DBs
    if (len(matched_protein_dbs) > 0) and (
        len(matched_protein_dbs) != len(protein_dbs)
    ):
        missing_dbs = protein_dbs - detected_aln_dbs
        mising_dbs = ",".join(list(missing_dbs))
        status = "WARN"
        reason = f"Missing DBs detected: {mising_dbs}"
    elif (len(matched_rna_dbs) > 0) and (len(matched_rna_dbs) != len(rna_dbs)):
        missing_dbs = rna_dbs - detected_aln_dbs
        mising_dbs = ",".join(list(missing_dbs))
        status = "WARN"
        reason = f"Missing DBs detected: {mising_dbs}"

    query_seq = None
    detected_qseqs = []
    for aln in alignments:
        query_seq = get_queryseq_from_file(aln)
        if aln.stem == "hmm_output" and query_seq is None:
            # this is the case where the hmm output is empty -
            # or malformed - this only happens when there are
            # no hits
            continue
        detected_qseqs.append(query_seq)
    detected_qseqs = set(detected_qseqs)
    ## check for multiple different query sequences
    if len(detected_qseqs) > 1:
        status = "FAIL"
        reason = "Multiple different query sequences detected"
        return None, (aln_dir, status, reason)

    query_seq = list(detected_qseqs)[0]

    if query_seq is None:
        status = "FAIL"
        reason = "Query sequence not found"
        return None, (aln_dir, status, reason)
    seqid = str(aln_dir).split("/")[-1]
    query_seq = (seqid, query_seq)
    return query_seq, (aln_dir, status, reason)


def main(args):
    msa_directory = Path(args.msa_directory)
    if not msa_directory.exists():
        raise FileNotFoundError(f"SA directory {msa_directory} does not exist.")

    if (args.protein_dbs is None) and (args.rna_dbs is None):
        raise ValueError("Must provide at least one set of dbs for protein or RNA")

    protein_dbs = set()
    rna_dbs = set()
    if args.protein_dbs:
        protein_dbs = set(args.protein_dbs.split(","))

    if args.rna_dbs:
        rna_dbs = set(args.rna_dbs.split(","))

    existing_seqids = set()
    if args.existing_fasta:
        with open(args.existing_fasta) as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                existing_seqids.add(record.id)

    ## get alignment folders
    aln_folders = list(msa_directory.iterdir())
    aln_folders = [aln for aln in aln_folders if aln.name not in existing_seqids]

    pfn = partial(wrap_extract, protein_dbs=protein_dbs, rna_dbs=rna_dbs)
    all_query_seqs = []
    logging_info = []
    with Pool(args.ncores) as p:
        for query_seq, log in tqdm.tqdm(
            p.imap_unordered(pfn, aln_folders, chunksize=10), total=len(aln_folders)
        ):
            if query_seq is not None:
                all_query_seqs.append(query_seq)
            if log is not None:
                logging_info.append(log)

    all_query_seqs = pd.DataFrame(all_query_seqs, columns=["seqid", "seq"])

    with open(args.out_fasta, "w+") as ofl:
        for _, row in all_query_seqs.iterrows():
            ofl.write(f">{row.seqid}\n{row.seq}\n")

    logging_info_df = pd.DataFrame(
        logging_info, columns=["aln_dir", "status", "reason"]
    )
    logging_info_df.to_csv(args.out_fasta + ".failed_dirs.csv", index=False)
    return


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
# %%
