"""
Contains functions around clustering sequences or molecules or finding MSA
representatives.
"""


def cluster_sequences(
    sequences: dict[str, str],
    min_seq_identity: float = 0.4,
    coverage: float = 0.8,
    coverage_mode: str = "0",
    mmseq_binary: str = "/Users/vss2134/miniforge3/bin/mmseqs",
    output_dir: str = "./",
) -> list[str]:
    """Run MMseqs2 clustering and return a representative sequence from each cluster

    Args:
        sequences (Dict[str, str]):
            Mapping of sequence id to sequence
        min_seq_identity (float, optional):
            Sequence similarity threshold to cluster at.  Defaults to 0.4.
        coverage (float, optional):
            Minimum sequence coverage of query/subject/both (depends on cov_mode).
            Defaults to 0.8.
        coverage_mode (str, optional):
            Coverage definition to use (see
            https://github.com/soedinglab/MMseqs2/wiki#how-to-set-the-right-alignment-coverage-to-cluster).
            Defaults to "0".
        mmseq_binary (str, optional):
            Full path to mmseqs2 binary. Defaults to
            "/Users/vss2134/miniforge3/bin/mmseqs".
        output_dir (str, optional): where to write out temporary fasta file. Defaults to
        "./".

    Returns:
        List[str]: list of representative sequences from each cluster
    """

    with open(f"{output_dir}/seqs.fa", "w+") as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n{seq}\n")
        f.flush()
    cmd = f"{mmseq_binary} easy-cluster {output_dir}/seqs.fa clusterRes tmp --min-seq-id {min_seq_identity} -c {coverage} --cov-mode {coverage_mode}"
    ## run and check for errors
    try:
        sp.run(cmd, shell=True, check=True)
    except sp.CalledProcessError as e:
        print(f"mmseqs failed with exit code {e.returncode}")
        raise e
    cluster_res = pd.read_csv(
        f"{output_dir}/clusterRes_cluster.tsv",
        sep="\t",
        names=["cluster_id", "uniprot_id"],
    )
    ## select an observation from each cluster
    cluster_sampled = cluster_res.groupby("cluster_id").sample(1, random_state=42)
    ## TODO: implement interface clustering

    ## clean up( this could be refactoered to be platform independent)
    sp.run(f"rm {output_dir}/seqs.fa", shell=True)
    sp.run(f"rm {output_dir}/clusterRes*", shell=True)
    return cluster_sampled
