import subprocess as sp
import tempfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem

from openfold3.core.data.io.sequence.fasta import write_multichain_fasta
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    ValClusteredDatasetCache,
)
from openfold3.core.data.primitives.caches.metadata import (
    get_all_cache_chains,
    get_mol_id_to_smiles,
    logger,
)
from openfold3.core.data.resources.residues import MoleculeType


def precompute_fingerprints(
    smiles_list: list[str], mfpgen: Chem.rdFingerprintGenerator
) -> list[Chem.DataStructs.ExplicitBitVect | None]:
    """Precompute fingerprints for a list of SMILES strings.

    Args:
        smiles_list (list[str]): List of SMILES strings.
        mfpgen (Chem.rdFingerprintGenerator): RDKit fingerprint generator.

    Returns:
        list[Chem.DataStructs.ExplicitBitVect | None]:
            List of fingerprints. If a fingerprint could not be generated, None is
            returned.

    """
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

    fingerprints = []

    for index, mol in enumerate(mols):
        if mol is None:
            logger.warning(
                f"Error in generating fingerprint for molecule: {smiles_list[index]}"
            )
            fingerprints.append(None)
        else:
            fingerprints.append(mfpgen.GetFingerprint(mol))

    return fingerprints


def get_mol_id_to_tanimoto_ligands(
    val_dataset_cache: ClusteredDatasetCache,
    train_dataset_cache: ClusteredDatasetCache,
    similarity_threshold: float = 0.85,
) -> dict[str, set[str]]:
    """
    Identify ligands in the validation dataset that have high Tanimoto similarity (>=
    similarity_threshold) with any ligand in the training dataset.

    Args:
        val_dataset_cache (ClusteredDatasetCache): Validation dataset cache.
        train_dataset_cache (ClusteredDatasetCache): Training dataset cache.
        similarity_threshold (float): Threshold for high homology.

    Returns:
        Dict[str, set[str]]:
            Mapping of ligand reference IDs in the validation set to homologous ligand
            reference IDs in the training set.
    """
    # Initialize fingerprint generator once
    mfpgen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)

    # Extract SMILES and reference IDs from validation dataset
    val_refid_to_smiles = get_mol_id_to_smiles(val_dataset_cache)

    # Extract SMILES and reference IDs from training dataset
    train_refid_to_smiles = get_mol_id_to_smiles(train_dataset_cache)

    # Precompute fingerprints for validation SMILES
    val_refids = list(val_refid_to_smiles.keys())
    val_smiles = list(val_refid_to_smiles.values())
    val_fps = precompute_fingerprints(val_smiles, mfpgen)

    assert len(val_refids) == len(val_fps)
    val_refid_to_fp = dict(zip(val_refids, val_fps))

    # Precompute fingerprints for training SMILES
    train_refids = list(train_refid_to_smiles.keys())
    train_smiles = list(train_refid_to_smiles.values())
    train_fps = precompute_fingerprints(train_smiles, mfpgen)

    assert len(train_refids) == len(train_fps)

    # Make map of ligands to high-homology-related ligands
    val_refid_to_homologs = defaultdict(set)

    # Iterate over validation set ligands
    for val_refid, val_fp in val_refid_to_fp.items():
        # If fingerprint is not available, conservatively set homologous ligands
        # to everything
        if val_fp is None:
            val_refid_to_homologs[val_refid] = set(train_refids)
            continue

        # Compute similarities in bulk
        similarities = Chem.DataStructs.BulkTanimotoSimilarity(val_fp, train_fps)

        # Get all similar ligands in training set
        homolog_train_refids = {
            train_refids[i]
            for i, score in enumerate(similarities)
            if score >= similarity_threshold
        }

        # Set homologous ligands for this ligand
        val_refid_to_homologs[val_refid] = homolog_train_refids

    assert set(val_refid_to_homologs.keys()) == set(val_refids)

    return val_refid_to_homologs


def run_mmseqs_search(
    query_id_to_sequence: dict[str, str],
    target_id_to_sequence: dict[str, str],
    min_sequence_identity: float = 0.4,
    sensitivity: float = 8.0,
    mmseqs_binary: str = "mmseqs",
    threads: int = 4,
    split_mem: str = "5G",
) -> dict[str, set[str]]:
    """
    Perform MMseqs2 search between query and target sequences and return
    pairs above a certain sequence identity.

    Args:
        query_id_to_sequence (Dict[str, str]):
            Mapping of query sequence IDs to sequences.
        target_id_to_sequence (Dict[str, str]):
            Mapping of target sequence IDs to sequences.
        min_sequence_identity (float):
            Minimum sequence identity for a hit to be considered homologous.
        sensitivity (float):
            Sensitivity of the search.
        mmseqs_binary (str):
            Path to the MMseqs2 binary.
        threads (int):
            Number of threads to use.
        split_mem (str):
            Limits the system RAM for prefiltering data structures. See MMSeqs
            --split-memory-limit option for more information.
    Returns:
        Dict[str, set[str]]:
            Mapping of query sequence IDs to homologous target sequence IDs.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        query_fasta = temp_dir / "query.fasta"
        target_fasta = temp_dir / "target.fasta"
        db_target = temp_dir / "target_db"
        db_query = temp_dir / "query_db"
        db_result = temp_dir / "result_db"
        result_tsv = temp_dir / "search_result.tsv"

        # Write query and target sequences to FASTA files
        write_multichain_fasta(query_fasta, query_id_to_sequence)
        write_multichain_fasta(target_fasta, target_id_to_sequence)

        # Create MMseqs2 database for target
        cmd_make_db = f"{mmseqs_binary} createdb {target_fasta} {db_target}"
        logger.info("Creating MMseqs2 database for target sequences.")
        sp.run(cmd_make_db, shell=True, check=True)

        cmd_make_db = f"{mmseqs_binary} createdb {query_fasta} {db_query}"
        logger.info("Creating MMseqs2 database for query sequences.")
        sp.run(cmd_make_db, shell=True, check=True)

        # Run MMseqs2 search with high sensitivity and ensuring that all target
        # sequences can be included in the hits
        cmd_search = (
            f"{mmseqs_binary} search {db_query} {db_target} {db_result} "
            f"{temp_dir}/tmp -s {sensitivity} --split-memory-limit {split_mem} "
            f"--threads {threads} --max-seqs {len(target_id_to_sequence)}"
        )
        logger.info("Running MMseqs2 search.")
        sp.run(cmd_search, shell=True, check=True)

        # Convert results to tabular format
        cmd_convert = (
            f"{mmseqs_binary} convertalis "
            f"{db_query} "
            f"{db_target} "
            f"{db_result} "
            f"{result_tsv}"
        )
        logger.info("Converting MMseqs2 search results to TSV.")
        sp.run(cmd_convert, shell=True, check=True)

        # Parse the search results
        query_seq_id_to_homologs = defaultdict(set)
        logger.info("Parsing MMseqs2 search results.")
        df = pd.read_csv(result_tsv, sep="\t", header=None)

        assert df[2].min() >= 0.0 and df[2].max() <= 1.0

        high_identity = df[df[2] > min_sequence_identity]
        for _, row in high_identity.iterrows():
            query_seq_id_to_homologs[row[0]].add(row[1])

    logger.info(f"Found hits for {len(query_seq_id_to_homologs)} sequences.")

    return query_seq_id_to_homologs


def get_polymer_chain_to_homolog_chains(
    val_dataset_cache: ClusteredDatasetCache,
    train_dataset_cache: ClusteredDatasetCache,
    id_to_sequence: dict[str, str],
    min_sequence_identity: float = 0.4,
) -> dict[str, set[str]]:
    """Maps protein/nucleic-acid validation chains to homologous training chains.

    This uses MMSeqs template search to find homologous chains between the validation
    and training datasets. The search is run separately for protein and nucleotide
    chains to prevent any sequence ambiguity.

    Args:
        val_dataset_cache (ClusteredDatasetCache):
            Validation dataset cache.
        train_dataset_cache (ClusteredDatasetCache):
            Training dataset cache.
        id_to_sequence (Dict[str, str]):
            Mapping of all preprocessed PDB-chain-IDs to sequences (should include both
            training and validation).
        min_sequence_identity (float):
            Minimum sequence identity for a hit to be considered homologous.

    Returns:
        Dict[str, set[str]]:
            Mapping of validation {pdb_id}_{chain_id} to homologous training
            {pdb_id}_{chain_id}s.
    """
    val_structure_cache = val_dataset_cache.structure_cache
    train_structure_cache = train_dataset_cache.structure_cache

    # Dictionary keyed on validation {pdb_id}_{chain_id} mapping to the homologous
    # training {pdb_id}_{chain_id}s
    chain_to_homologs = {}

    # Run MMSeqs search separately for protein and nucleotide chains to prevent any
    # sequence ambiguity
    for molecule_types in (
        (MoleculeType.PROTEIN,),
        (MoleculeType.DNA, MoleculeType.RNA),
    ):
        val_chains = get_all_cache_chains(
            val_structure_cache, restrict_to_molecule_types=molecule_types
        )
        train_chains = get_all_cache_chains(
            train_structure_cache, restrict_to_molecule_types=molecule_types
        )

        val_id_to_sequence = {k: id_to_sequence[k] for k in val_chains}
        train_id_to_sequence = {k: id_to_sequence[k] for k in train_chains}

        if molecule_types == (MoleculeType.PROTEIN,):
            logger.info("Running MMseqs2 search for protein chains.")
        else:
            logger.info("Running MMseqs2 search for nucleotide chains.")

        chain_to_homologs.update(
            run_mmseqs_search(
                query_id_to_sequence=val_id_to_sequence,
                target_id_to_sequence=train_id_to_sequence,
                min_sequence_identity=min_sequence_identity,
            )
        )

    return chain_to_homologs


def assign_homology_labels(
    val_dataset_cache: ValClusteredDatasetCache,
    train_dataset_cache: ClusteredDatasetCache,
    id_to_sequence: dict[str, str],
    seq_identity_threshold: float = 0.4,
    tanimoto_threshold: float = 0.85,
) -> ValClusteredDatasetCache:
    """Detects if chains/interfaces are low-homology to the training dataset.

    Following AF3 SI 5.8, this function labels chains as low homology if there is no
    chain in the training set with a sequence identity above a certain threshold for
    polymer chains, or a Tanimoto similarity above a certain threshold for ligands.

    Interfaces in the validation dataset are labeled as low homology if there is no PDB
    in the training set that contains homologous chains to both chains in the interface.

    Args:
        val_dataset_cache (ValClusteredDatasetCache):
            Validation dataset cache.
        train_dataset_cache (ClusteredDatasetCache):
            Training dataset cache.
        id_to_sequence (Dict[str, str]):
            Mapping of all preprocessed PDB-chain-IDs to sequences (should include both
            training and validation).
        seq_identity_threshold (float):
            Minimum sequence identity for a hit to be considered homologous.
        tanimoto_threshold (float):
            Minimum Tanimoto similarity for a hit to be considered homologous.
    """
    val_structure_cache = val_dataset_cache.structure_cache
    train_structure_cache = train_dataset_cache.structure_cache

    # PREPARE LIGAND SIMILARITY
    # Get ligands with high Tanimoto similarity
    val_mol_id_to_homologs = get_mol_id_to_tanimoto_ligands(
        val_dataset_cache, train_dataset_cache, similarity_threshold=tanimoto_threshold
    )

    # Get mapping of every training set reference molecule ID to all PDB-IDs it occurs
    # in.
    train_mol_id_to_pdb_ids = defaultdict(set)
    for pdb_id, metadata in train_structure_cache.items():
        for chain_metadata in metadata.chains.values():
            if chain_metadata.molecule_type == MoleculeType.LIGAND:
                train_mol_id_to_pdb_ids[chain_metadata.reference_mol_id].add(pdb_id)

    # Map validation molecule IDs to training PDB IDs containing a homologous ligand
    val_mol_id_to_homolog_pdbs = {
        val_mol_id: set.union(
            *(train_mol_id_to_pdb_ids[train_mol_id] for train_mol_id in train_mol_ids)
        )
        if train_mol_ids
        else set()
        for val_mol_id, train_mol_ids in val_mol_id_to_homologs.items()
    }

    # PREPARE SEQUENCE SIMILARITY
    val_chain_to_homologs = get_polymer_chain_to_homolog_chains(
        val_dataset_cache=val_dataset_cache,
        train_dataset_cache=train_dataset_cache,
        id_to_sequence=id_to_sequence,
        min_sequence_identity=seq_identity_threshold,
    )

    # Similarly to earlier, map validation chain IDs to training PDB IDs containing a
    # homologous chain
    val_chain_to_homolog_pdbs = {
        val_chain: set(pdb_chain_id[:4] for pdb_chain_id in train_chain_ids)
        for val_chain, train_chain_ids in val_chain_to_homologs.items()
    }

    def get_homolog_pdbs(chain_id: str, chain_data):
        """Small helper function to retrieve the set of homolog PDBs for a chain."""
        # Get homologous PDBs for ligand chains
        if chain_data.molecule_type == MoleculeType.LIGAND:
            return val_mol_id_to_homolog_pdbs.get(chain_data.reference_mol_id, set())
        # Get homologous PDBs for polymer chains
        else:
            return val_chain_to_homolog_pdbs.get(chain_id, set())

    # SET HOMOLOGY LABELS
    # Assign homology labels to validation dataset
    for structure_data in val_structure_cache.values():
        # Start by setting chain-wise homology
        for chain_id, chain_data in structure_data.chains.items():
            # Set homology to low if there is no training PDB with a homologous chain
            homolog_pdbs = get_homolog_pdbs(chain_id, chain_data)
            chain_data.low_homology = len(homolog_pdbs) == 0

        # Continue with interface-wise homology
        for interface_id, interface_data in structure_data.interfaces.items():
            chain_id_1, chain_id_2 = interface_id.split("_")
            chain_data_1 = structure_data.chains[chain_id_1]
            chain_data_2 = structure_data.chains[chain_id_2]

            pdbs_with_homolog_chain_1 = get_homolog_pdbs(chain_id_1, chain_data_1)
            pdbs_with_homolog_chain_2 = get_homolog_pdbs(chain_id_2, chain_data_2)

            # Set to low homology if there is no single PDB-ID containing chains
            # homologous to both interface chains
            interface_data.low_homology = pdbs_with_homolog_chain_1.isdisjoint(
                pdbs_with_homolog_chain_2
            )

    return val_dataset_cache
