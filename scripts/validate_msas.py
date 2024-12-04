import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Callable
from Bio import SeqIO
from openfold3.core.data.io.sequence.msa import parse_a3m, parse_stockholm


def check_msa(file: str, ref_seq: str, parser: Callable[str], db: str, pdb_id: str):
    """make sure MSA exists, and query sequence matches ground truth

    Args:
        file (str): abs path to MSA file
        ref_seq (str): ground truth amino acid sequence
        parser (Callable[str]): parsing function for MSA file. one of parse_a3m 
                                or parse_stockholm
        db (str): which database MSA searched
        pdb_id (str): pdb_identifier for sequence. propagated from 
                      input fasta

    Returns:
        str: logging info
    """
    if not Path(file).exists():
        return f"{pdb_id},{db},missing\n"
    with open(file) as f:
        msa_array = parser(f.read())
    q_obs = "".join(msa_array.msa[0])
    if q_obs == ref_seq:
        return ""
    else:
        return f"{pdb_id},{db},corrupt\n"

def validate(pdb_id: str, ref_seq: str, msa_dir: str, outfile: str):
    """
    Args:
        pdb_id (str): pdb_identifier for sequence. propagated from input fasta
        ref_seq (str): ground truth amino acid sequence
        msa_dir (str): folder that contains alignments. folder 
                    vnames need to match to ids in input fasta
        outfile (str): path to logfile that tracks alignments that fail validation
    """

    if Path(f"{msa_dir}/{pdb_id}/VALID_DIR").exists():
        return
    outstr = ""
    outstr += check_msa(
        f"{msa_dir}/{pdb_id}/uniref90_hits.sto",
        ref_seq,
        parse_stockholm,
        "uniref90",
        pdb_id,
    )
    outstr += check_msa(
        f"{msa_dir}/{pdb_id}/uniprot_hits.sto",
        ref_seq,
        parse_stockholm,
        "uniprot",
        pdb_id,
    )
    outstr += check_msa(
        f"{msa_dir}/{pdb_id}/mgnify_hits.sto",
        ref_seq,
        parse_stockholm,
        "mgnify",
        pdb_id,
    )
    outstr += check_msa(
        f"{msa_dir}/{pdb_id}/cfdb_uniref30.a3m", ref_seq, parse_a3m, "cfdb", pdb_id
    )
    with open(outfile, "a") as ofl:
        ofl.write(outstr)
    if outstr == "":
        Path(f"{msa_dir}/{pdb_id}/VALID_DIR").touch()
    return


def wrap_validate(arg_l):
    for args in arg_l:
        validate(*args)
    return


def main():
    fasta = sys.argv[1]  ## path to fasta of ground truth sequences
    msa_dir = sys.argv[
        2
    ]  ## folder that contains alignments. 
       ## folder names need to match to ids in input fasta
    outfile = sys.argv[
        3
    ]  ## path to logfile that tracks alignments that fail validation
    nworkers = int(sys.argv[4])  ## number of parallel jobs to use
    seq_l = [
        (seq.id, str(seq.seq), msa_dir, outfile) for seq in SeqIO.parse(fasta, "fasta")
    ]
    ### batch input for multiprocessing
    input_per_worker = (len(seq_l) // nworkers) + 1
    batched_args = [
        seq_l[i * input_per_worker : (i + 1) * input_per_worker]
        for i in range(nworkers)
    ]
    with Pool(nworkers) as p:
        p.map(wrap_validate, batched_args)

if __name__ == "__main__":
    main()
