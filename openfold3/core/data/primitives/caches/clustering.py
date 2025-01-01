import itertools
import subprocess as sp
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Literal

import pandas as pd

from openfold3.core.data.io.sequence.fasta import write_multichain_fasta
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
)
from openfold3.core.data.primitives.caches.metadata import (
    get_all_cache_chains,
    logger,
)
from openfold3.core.data.resources.residues import MoleculeType


def get_sequence_to_cluster_id(
    id_to_sequence: dict[str, str],
    min_seq_identity: float = 0.4,
    coverage: float = 0.9,
    coverage_mode: str = "0",
    sensitivity: float = 8,
    max_seqs: int = 1000,
    cluster_mode: Literal[0, 1, 2, 3] = 0,
    verbosity_level: int = 3,
    mmseq_binary: str = "mmseqs",
    split_mem: str = "5G",  # TODO: pass these through to high-level arguments
    threads: int = 4,
) -> dict[str, int]:
    """Runs MMseqs2 clustering and returns a mapping of sequence id to cluster id.

    Default settings are mostly derived from what is internally used at RCSB PDB (see
    https://github.com/soedinglab/MMseqs2/issues/452), although we set the cluster_mode
    to the greedy set cover default in order to avoid getting too-large clusters, and
    the default min_seq_identity to 40% which is used in AF3 SI 2.5.3.

    Args:
        id_to_sequence (dict[str, str]):
            Mapping of sequence id to sequence
        min_seq_identity (float, optional):
            Sequence similarity threshold to cluster at. Defaults to 0.4.
        coverage (float, optional):
            Minimum sequence coverage of query/subject/both (depends on cov_mode).
            Defaults to 0.9.
        coverage_mode (str, optional):
            Coverage definition to use (see
            https://github.com/soedinglab/MMseqs2/wiki#how-to-set-the-right-alignment-coverage-to-cluster).
            Defaults to "0".
        sensitivity (float, optional):
            Sensitivity of the clustering algorithm. Defaults to 8.
        max_seqs (int, optional):
            Maximum number of sequences allowed to pass the prefilter. Defaults to 1000.
        cluster_mode (Literal[0, 1, 2, 3], optional):
            Clustering mode to use (see
            https://github.com/soedinglab/MMseqs2/wiki#clustering-modes). Defaults to 0.
        mmseq_binary (str, optional):
            Full path to mmseqs2 binary. Defaults to "mmseqs".
    Returns:
        dict[str, int]: Mapping of sequence id to cluster id
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        temp_fasta = write_multichain_fasta(temp_dir / "seqs.fa", id_to_sequence)
        output_prefix = temp_dir / "clusterRes"

        cmd = (
            f"{mmseq_binary} easy-cluster {temp_fasta} {output_prefix} {temp_dir} "
            f"--min-seq-id {min_seq_identity} -c {coverage} --cov-mode {coverage_mode} "
            f"-s {sensitivity} --max-seqs {max_seqs} --cluster-mode {cluster_mode} "
            f"-v {verbosity_level} --split-memory-limit {split_mem} --threads {threads}"
        )

        # Run and read required cluster information, then delete tmp_dir
        logger.info("Clustering protein sequences with MMSeqs2.")
        try:
            sp.run(cmd, shell=True, check=True)
        except sp.CalledProcessError as e:
            print(f"mmseqs failed with exit code {e.returncode}")
            raise e
        logger.info("Done clustering protein sequences.")

        cluster_data = pd.read_csv(
            f"{temp_dir}/clusterRes_cluster.tsv",
            sep="\t",
            names=["cluster_id", "seq_id"],
        )

    # Remap cluster IDs to numerical ones
    cluster_data["cluster_id_numeric"] = pd.factorize(cluster_data["cluster_id"])[0]

    id_to_cluster_id = cluster_data.set_index("seq_id")["cluster_id_numeric"].to_dict()

    return id_to_cluster_id


def make_interface_cluster_id(chain_1_cluster_id: str, chain_2_cluster_id: str) -> str:
    """Get the cluster ID for an interface from the cluster IDs of its chains.

    Following AF3 SI 2.5.3, the interface cluster ID is the sorted concatenation of the
    cluster IDs of the two chains that form the interface.

    Args:
        chain_1_cluster_id:
            Cluster ID of the first chain in the interface.
        chain_2_cluster_id:
            Cluster ID of the second chain in the interface.
    Returns:
        Cluster ID of the interface.
    """
    return "_".join(sorted([chain_1_cluster_id, chain_2_cluster_id]))


def add_cluster_data(
    dataset_cache: ClusteredDatasetCache,
    id_to_sequence: dict[str, str],
    add_sizes: bool = True,
) -> None:
    """Add cluster IDs to the structure metadata cache.

    Adds cluster IDs and cluster sizes for all chains in the structure metadata cache,
    following 2.5.3 of the AF3 SI. Note that we don't cluster ligands on CCD ID but
    canonical SMILES instead, which more appropriately deals with multi-residue ligands
    such as glycans.

    Args:
        dataset_cache:
            The dataset cache to update.
        id_to_sequence:
            Dictionary mapping sequence IDs to sequences.
        add_sizes:
            Whether to add cluster sizes to the cluster_size field of each chain and
            interface. Defaults to True.
    """
    structure_cache = dataset_cache.structure_data
    reference_mol_cache = dataset_cache.reference_molecule_data

    # Subset sequences to only the ones explicitly in cache for correct clustering
    all_protein_chains = get_all_cache_chains(
        structure_cache, restrict_to_molecule_types=[MoleculeType.PROTEIN]
    )
    id_to_sequence = {
        k: v for k, v in id_to_sequence.items() if k in all_protein_chains
    }

    # Get sequences to cluster IDs with MMSeqs2-based clustering
    id_to_cluster_id = get_sequence_to_cluster_id(id_to_sequence)

    # Make a generator for new cluster IDs
    cluster_id_gen = itertools.count(start=max(id_to_cluster_id.values()) + 1)

    # Map unique identifiers for entities that are not strictly proteins to cluster IDs,
    # following AF3 SI 2.5.3. Unique identifiers are sequences for peptides and NAs, and
    # canonical SMILES for ligands. On accession, this dict either returns an existing
    # cluster ID or generates a new one.
    non_protein_ident_to_cluster_id = defaultdict(lambda: str(next(cluster_id_gen)))

    cluster_id_to_size = defaultdict(lambda: 0)

    # Get all cluster IDs and track sizes
    for pdb_id, metadata in structure_cache.items():
        # Get cluster IDs for each chain
        for chain_id, chain_metadata in metadata.chains.items():
            molecule_type = chain_metadata.molecule_type

            # Standard polymers
            if molecule_type in (
                MoleculeType.PROTEIN,
                MoleculeType.DNA,
                MoleculeType.RNA,
            ):
                pdb_chain_id = f"{pdb_id}_{chain_id}"

                sequence = id_to_sequence[pdb_chain_id]

                # Get cluster IDs for standard proteins
                if molecule_type == MoleculeType.PROTEIN and len(sequence) >= 10:
                    cluster_id = str(id_to_cluster_id[pdb_chain_id])

                # Cluster based on 100% sequence identity for peptides and NAs
                else:
                    cluster_id = non_protein_ident_to_cluster_id[sequence]

            # Ligands
            elif molecule_type == MoleculeType.LIGAND:
                reference_mol_id = chain_metadata.reference_mol_id

                # TODO: remove this logic after debugging the preprocessing
                try:
                    smiles = reference_mol_cache[reference_mol_id].canonical_smiles
                except KeyError:
                    logger.warning(
                        f"No reference molecule found for {reference_mol_id}"
                    )
                    cluster_id = "UNKNOWN"

                cluster_id = non_protein_ident_to_cluster_id[smiles]
            else:
                raise ValueError(f"Unexpected molecule type: {molecule_type}")

            # Add cluster_id and increment size tracker
            chain_metadata.cluster_id = cluster_id
            cluster_id_to_size[cluster_id] += 1

        # Get cluster IDs for each interface by joining the cluster IDs of individual
        # chains
        for interface_id in metadata.interfaces:
            chain_1, chain_2 = interface_id.split("_")

            chain_1_cluster_id = structure_cache[pdb_id].chains[chain_1].cluster_id
            chain_2_cluster_id = structure_cache[pdb_id].chains[chain_2].cluster_id

            interface_cluster_id = make_interface_cluster_id(
                chain_1_cluster_id=chain_1_cluster_id,
                chain_2_cluster_id=chain_2_cluster_id,
            )

            metadata.interfaces[interface_id].cluster_id = interface_cluster_id

            # Increment cluster size
            cluster_id_to_size[interface_cluster_id] += 1

    # Add cluster sizes
    if add_sizes:
        for metadata in structure_cache.values():
            for chain_data in metadata.chains.values():
                cluster_id = chain_data.cluster_id

                # TODO: remove this after debugging preprocessing
                if cluster_id == "UNKNOWN":
                    chain_data.cluster_size = 1
                else:
                    chain_data.cluster_size = cluster_id_to_size[cluster_id]

            for interface_data in metadata.interfaces.values():
                cluster_id = interface_data.cluster_id

                # TODO: remove this after debugging preprocessing
                if "UNKNOWN" in cluster_id:
                    interface_data.cluster_size = 1
                else:
                    interface_data.cluster_size = cluster_id_to_size[cluster_id]

    return None
