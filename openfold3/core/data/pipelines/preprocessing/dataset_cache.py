import datetime
import json
import logging
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path

from tqdm import tqdm

from openfold3.core.data.io.dataset_cache import (
    format_nested_dict_for_json,
    write_datacache_to_json,
)
from openfold3.core.data.io.s3 import list_bucket_entries
from openfold3.core.data.io.sequence.fasta import (
    consolidate_preprocessed_fastas,
)
from openfold3.core.data.primitives.caches.clustering import (
    add_cluster_data,
)
from openfold3.core.data.primitives.caches.filtering import (
    ChainDataPoint,
    InterfaceDataPoint,
    add_and_filter_alignment_representatives,
    add_ligand_data_to_monomer_cache,
    assign_interface_metric_eligibility_labels,
    assign_ligand_model_fits,
    build_provisional_clustered_dataset_cache,
    build_provisional_clustered_val_dataset_cache,
    filter_by_max_polymer_chains,
    filter_by_release_date,
    filter_by_resolution,
    filter_by_skipped_structures,
    filter_by_token_count,
    filter_cache_by_specified_interfaces,
    filter_cache_to_specified_chains,
    filter_only_ligand_ligand_metrics,
    func_with_n_filtered_chain_log,
    get_validation_summary_stats,
    select_final_validation_data,
    set_nan_fallback_conformer_flag,
    subsample_chains_by_type,
    subsample_interfaces_by_type,
    subsample_interfaces_per_cluster,
)
from openfold3.core.data.primitives.caches.format import (
    ClusteredDatasetCache,
    DatasetReferenceMoleculeData,
    PreprocessingDataCache,
    PreprocessingStructureDataCache,
    ProteinMonomerChainData,
    ProteinMonomerDatasetCache,
    ProteinMonomerStructureData,
    ValidationDatasetCache,
)
from openfold3.core.data.primitives.caches.homology import assign_homology_labels
from openfold3.core.data.resources.residues import MoleculeType

logger = logging.getLogger(__name__)


# TODO: Make docstring more complete for new args
def filter_structure_metadata_af3(
    structure_cache: PreprocessingStructureDataCache,
    max_release_date: datetime.date | str,
    min_release_date: datetime.date | str = None,
    max_resolution: float = 9.0,
    max_polymer_chains: int = 300,
    max_tokens: int | None = None,
) -> PreprocessingStructureDataCache:
    """Filter the structure metadata cache for training or validation.

    TODO: Update docstring

    Applies the following filters from the AF3 SI 2.5.4 that have not yet been applied
    in preprocessing:
    - release date <= max_release_date
    - number of polymer chains <= 300
    - resolution <= 9.0

    Args:
        structure_cache:
            Structure metadata cache to filter.

    Returns:
        Filtered structure metadata cache.
    """
    if not isinstance(max_release_date, datetime.date):
        max_release_date = datetime.datetime.strptime(
            max_release_date, "%Y-%m-%d"
        ).date()
    if min_release_date and not isinstance(min_release_date, datetime.date):
        min_release_date = datetime.datetime.strptime(
            min_release_date, "%Y-%m-%d"
        ).date()

    # Convenience wrapper that logs the number of structures filtered out
    with_log = partial(func_with_n_filtered_chain_log, logger=logger)

    # Removes structures that were skipped in preprocessing (skip logging here because
    # it does not work with skipped/failed structures)
    filtered_cache = filter_by_skipped_structures(structure_cache)
    # TODO: we need to catch NMR etc. here which don't have a defined resolution and
    # probably set resolution for them to zero
    filtered_cache = with_log(filter_by_resolution)(
        filtered_cache, max_resolution, ignore_nmr=True
    )

    filtered_cache = with_log(filter_by_release_date)(
        filtered_cache,
        min_date=min_release_date,
        max_date=max_release_date,
    )

    filtered_cache = with_log(filter_by_max_polymer_chains)(
        filtered_cache, max_polymer_chains
    )
    if max_tokens:
        filtered_cache = with_log(filter_by_token_count)(filtered_cache, max_tokens)

    return filtered_cache


# TODO: Add docstring!
def create_pdb_training_dataset_cache_af3(
    metadata_cache_path: Path,
    preprocessed_dir: Path,
    alignment_representatives_fasta: Path,
    output_path: Path,
    dataset_name: str,
    max_release_date: datetime.date | str,
    max_resolution: float = 9.0,
    max_polymer_chains: int = 300,
    filter_missing_alignment: bool = True,
    missing_alignment_log: Path = None,
) -> None:
    """Create a training cache from a metadata cache.

    Args:
        metadata_cache_path:
            Path to the preprocessed metadata cache.
        preprocessed_dir:
            Preprocessing output directory with preprocessed structure and fasta files.
        alignment_representatives_fasta:
            A FASTA file containing the identifier of each alignment as a header and the
            query sequence as the sequence. Used to map every sequence in the
            preprocessed directory to a corresponding MSA. Every protein or RNA sequence
            without an alignment will be filtered out.
        output_path:
            Path to write the training dataset cache to.
        dataset_name:
            Name of the dataset, e.g. 'PDB-weighted'.
        max_release_date:
            Maximum release date for included structures, formatted as 'YYYY-MM-DD'.
        max_resolution:
            Maximum resolution for structures in the dataset in Å.
        max_polymer_chains:
            Maximum number of polymer chains for included structures.
        filter_missing_alignment:
            Whether to filter out chains (and their corresponding structures) whose
            sequences do not match to a representative in the
            alignment_representatives_fasta.
        missing_alignment_log:
            Path to write a JSON file containing all chains that were filtered out
            because they do not have a corresponding alignment.
    """
    metadata_cache = PreprocessingDataCache.from_json(metadata_cache_path)

    # Read in FASTAs of all sequences in the training set
    logger.info("Scanning FASTA directories...")
    id_to_sequence = consolidate_preprocessed_fastas(preprocessed_dir)

    # Get a mapping of PDB IDs to release dates before any filtering is done
    pdb_id_to_release_date = {}
    for pdb_id, metadata in metadata_cache.structure_data.items():
        pdb_id_to_release_date[pdb_id] = metadata.release_date

    # Subset the structures in the preprocessed metadata to only the desired ones
    metadata_cache.structure_data = filter_structure_metadata_af3(
        metadata_cache.structure_data,
        max_release_date=max_release_date,
        max_resolution=max_resolution,
        max_polymer_chains=max_polymer_chains,
    )

    # Create a provisional dataset training cache with extra fields for cluster and NaN
    # conformer information that will be filled in later
    dataset_cache = build_provisional_clustered_dataset_cache(
        preprocessing_cache=metadata_cache,
        dataset_name=dataset_name,
    )

    # Convenience wrapper that logs the number of structures filtered out
    with_log = partial(func_with_n_filtered_chain_log, logger=logger)

    # Map each target chain to an alignment representative, then filter all structures
    # without alignment representatives
    if filter_missing_alignment:
        if missing_alignment_log:
            structure_data, unmatched_entries = with_log(
                add_and_filter_alignment_representatives
            )(
                structure_cache=dataset_cache.structure_data,
                query_chain_to_seq=id_to_sequence,
                alignment_representatives_fasta=alignment_representatives_fasta,
                return_no_repr=True,
            )

            # Write all chains without alignment representatives to a JSON file. These
            # are excluded from training.
            with open(missing_alignment_log, "w") as f:
                # Convert the internal dataclasses to dict
                unmatched_entries = {
                    pdb_id: {chain_id: asdict(chain_data)}
                    for pdb_id, chain_data in unmatched_entries.items()
                    for chain_id, chain_data in chain_data.items()
                }

                # Format datacache-types appropriately
                unmatched_entries = format_nested_dict_for_json(unmatched_entries)

                json.dump(unmatched_entries, f, indent=4)
        else:
            structure_data = with_log(add_and_filter_alignment_representatives)(
                dataset_cache.structure_data,
                query_chain_to_seq=id_to_sequence,
                alignment_representatives_fasta=alignment_representatives_fasta,
                preprocessed_dir=preprocessed_dir,
            )

        dataset_cache.structure_data = structure_data

    # Add cluster IDs and cluster sizes for all chains
    logger.info("Adding cluster information...")
    add_cluster_data(
        dataset_cache=dataset_cache,
        id_to_sequence=id_to_sequence,
        add_sizes=True,
    )
    logger.info("Done clustering.")

    # Block usage of reference conformer coordinates from PDB-IDs that are outside the
    # training split. Needs to be run before the filtering to use the full release date
    # information in structure_data.
    set_nan_fallback_conformer_flag(
        pdb_id_to_release_date=pdb_id_to_release_date,
        reference_mol_cache=dataset_cache.reference_molecule_data,
        max_model_pdb_release_date=max_release_date,
    )

    # Write the final dataset cache to disk
    write_datacache_to_json(dataset_cache, output_path)

    logger.info("DONE.")


# TODO: implement a more general way to interact with both local and S3 data
def create_protein_monomer_dataset_cache_af3(
    data_directory: Path,
    protein_reference_molecule_data_file: Path,
    dataset_name: str,
    output_dir: Path,
    s3_client_config: dict | None = None,
    check_filename_exists: str | None = None,
    num_workers: int = 1,
) -> ProteinMonomerDatasetCache:
    """Creates a protein monomer dataset cache.

    Args:
        data_directory (Path):
            Directory containing subdirectories for each protein monomer. Names of the
            per-chain directories will be used as chain IDs and representative ids for
            the corresponding chain in the dataset cache. If the directory lives in an
            S3 bucket, the path should be "s3:/<bucket>/<prefix>".
        protein_reference_molecule_data_file (Path):
            Path to a JSON file containing reference molecule data for each canonical
            protein monomer.
        dataset_name (str):
            Name of the dataset.
        output_dir (Path):
            Directory to write the dataset cache to.
        s3_client_config (dict, optional):
            Configuration for the S3 client. If None, the client is started without a
            profile. Supports profile and max_keys keys. Defaults to None.,
        check_filename_exists (str, optional):
            If provided, only adds proteins to the dataset cache if the given filename
            exists within the chain directory. Defaults to None, and if None all
            directories are added.
        num_workers (int, optional):
            Number of workers to use for parallel processing. Defaults to 1. Only used
            if check_filename_exists is specified.

    Returns:
        ProteinMonomerDatasetCache: The created dataset cache.
    """
    # Get all chain directories
    # S3
    if str(data_directory).startswith("s3:/"):
        if s3_client_config is None:
            s3_client_config = {}
        logger.info("1/4: Fetching chain directories from S3 bucket.")
        chain_directories = list_bucket_entries(
            bucket_name=data_directory.parts[1],
            prefix="/".join(data_directory.parts[2:]) + "/",
            profile=s3_client_config.get("profile", None),
            max_keys=s3_client_config.get("max_keys", 1000),
            check_filename_exists=check_filename_exists,
            num_workers=num_workers,
        )
    # Local
    else:
        print("1/4: Fetching chain directories locally.")
        chain_directories = [
            entry for entry in data_directory.iterdir() if entry.is_dir()
        ]

    # Load reference molecule data
    with open(protein_reference_molecule_data_file) as f:
        reference_molecule_data_dict = json.load(f)

    # Populate structure data field
    structure_data = {}
    for chain_directory in tqdm(
        chain_directories,
        total=len(chain_directories),
        desc="2/4: Populating structure data",
    ):
        chain_id = chain_directory.stem
        structure_data[chain_id] = ProteinMonomerStructureData(
            {
                "1": ProteinMonomerChainData(
                    alignment_representative_id=chain_id, template_ids=[]
                )
            }
        )

    # Reference molecule data
    print("3/4: Populating reference molecule data.")
    reference_molecule_data = {}
    for ref_mol_id, ref_mol_data in reference_molecule_data_dict.items():
        reference_molecule_data[ref_mol_id] = DatasetReferenceMoleculeData(
            conformer_gen_strategy=ref_mol_data["conformer_gen_strategy"],
            fallback_conformer_pdb_id=ref_mol_data["fallback_conformer_pdb_id"],
            canonical_smiles=ref_mol_data["canonical_smiles"],
            set_fallback_to_nan=ref_mol_data["set_fallback_to_nan"],
        )

    # Create dataset cache
    dataset_cache = ProteinMonomerDatasetCache(
        name=dataset_name,
        structure_data=structure_data,
        reference_molecule_data=reference_molecule_data,
    )

    # Write the final dataset cache to disk
    print("4/4: Writing dataset cache to disk.")
    write_datacache_to_json(
        dataset_cache, output_dir / "training_cache_protein_monomer.json"
    )

    return dataset_cache


def select_multimer_cache(
    val_dataset_cache: ValidationDatasetCache,
    id_to_sequence: dict[str, str],
    max_token_count: int = 2048,
    n_protein_protein: int = 600,
    n_protein_dna: int = 100,
    n_dna_dna: int = 100,
    n_protein_ligand: int = 600,
    n_dna_ligand: int = 50,
    n_ligand_ligand: int = 200,
    n_protein_rna: int | None = None,
    n_rna_rna: int | None = None,
    n_dna_rna: int | None = None,
    n_rna_ligand: int | None = None,
    random_seed: int | None = None,
) -> ValidationDatasetCache:
    """Filters out chains/interfaces following AF3 SI 5.8 Multimer Selection Step 2-4.

    Filters the cache to only interfaces passing the metrics inclusion criteria
    (low-homology, and good ranking model fit and single-residue for ligands). Then
    subsamples the remaining interfaces as specified in the SI and only keeps the chains
    corresponding to those interfaces.

    Args:
        val_dataset_cache (ValClusteredDatasetCache):
            The cache to filter.
        id_to_sequence (dict[str, str]):
            A dictionary mapping PDB-chain IDs to sequences. Required for clustering.
        max_token_count (int):
            The maximum token count for structures to be included in the cache. Default
            is 2048.
        n_protein_protein (int):
            How many interfaces to sample from protein-protein interfaces. Default is
            600.
        n_protein_dna (int):
            How many interfaces to sample from protein-DNA interfaces. Default is 100.
        n_dna_dna (int):
            How many interfaces to sample from DNA-DNA interfaces. Default is 100.
        n_protein_ligand (int):
            How many interfaces to sample from protein-ligand interfaces. Default is
            600.
        n_dna_ligand (int):
            How many interfaces to sample from DNA-ligand interfaces. Default is 50.
        n_ligand_ligand (int):
            How many interfaces to sample from ligand-ligand interfaces. Default is 200.
        n_protein_rna (int | None):
            How many interfaces to sample from protein-RNA interfaces. Default is all.
        n_rna_rna (int | None):
            How many interfaces to sample from RNA-RNA interfaces. Default is all.
        n_dna_rna (int | None):
            How many interfaces to sample from DNA-RNA interfaces. Default is all.
        n_rna_ligand (int | None):
            How many interfaces to sample from RNA-ligand interfaces. Default is all.
        random_seed (int | None):
            The random seed to use for subsampling. Default is None.

    Returns:
        A copy of the original cache filtered to only contain the subsampled interfaces
        created by the filtering steps, as well as their corresponding chains.
    """
    logger.info("Selecting multimer set...")
    filtered_cache = deepcopy(val_dataset_cache)

    keep_interface_datapoints = set()

    for pdb_id, structure_data in val_dataset_cache.structure_data.items():
        # Mark interfaces and chains to keep
        for interface_id, interface_data in structure_data.interfaces.items():
            # Check if interface satisfies both low-homology and ligand criteria
            if interface_data.metric_eligible:
                keep_interface_datapoints.add(InterfaceDataPoint(pdb_id, interface_id))

    # Remove any chains/interfaces that should not be kept from the cache
    logger.info("Filtering cache by specified interfaces.")
    filter_cache_by_specified_interfaces(filtered_cache, keep_interface_datapoints)

    # Assign cluster ids
    logger.info("Assigning cluster IDs.")
    add_cluster_data(filtered_cache, id_to_sequence=id_to_sequence, add_sizes=False)

    # Subsample one interface per cluster
    logger.info("Subsampling interfaces.")
    subsample_interfaces_per_cluster(filtered_cache, random_seed=random_seed)

    # Subsample interfaces by prespecified counts for certain types
    subsample_interfaces_by_type(
        filtered_cache,
        n_protein_protein=n_protein_protein,
        n_protein_dna=n_protein_dna,
        n_dna_dna=n_dna_dna,
        n_protein_ligand=n_protein_ligand,
        n_dna_ligand=n_dna_ligand,
        n_ligand_ligand=n_ligand_ligand,
        n_protein_rna=n_protein_rna,
        n_rna_rna=n_rna_rna,
        n_dna_rna=n_dna_rna,
        n_rna_ligand=n_rna_ligand,
        random_seed=random_seed,
    )

    # Filter by token count
    filtered_cache.structure_data = filter_by_token_count(
        filtered_cache.structure_data, max_token_count
    )

    return filtered_cache


def select_monomer_cache(
    val_dataset_cache: ValidationDatasetCache,
    id_to_sequence: dict[str, str],
    max_token_count: int = 2048,
    n_protein: int = 40,
    n_dna: int | None = None,
    n_rna: int | None = None,
    random_seed: int | None = None,
) -> ValidationDatasetCache:
    """Filters out chains/interfaces following AF3 SI 5.8 Monomer Selection Step 2-4.

    Filters the cache down to only low-homology polymeric chains with optional ligands,
    and subsamples the remaining chains according to the SI.

    Note that proteins are sampled as unique cluster representatives, which is not
    directly stated in the SI but seems logical given that chains are preclustered.

    Args:
        val_dataset_cache (ValClusteredDatasetCache):
            The cache to filter.
        id_to_sequence (dict[str, str]):
            A dictionary mapping PDB-chain IDs to sequences. Required for clustering.
        max_token_count (int):
            The maximum token count for structures to be included in the cache. Defaults
            to 2048.
        n_protein (int):
            The number of protein chains to sample. Default is 40.
        n_dna (int | None):
            The number of DNA chains to sample. Default is None, which means that all
            DNA chains will be selected.
        n_rna (int | None):
            The number of RNA chains to sample. Default is None, which means that all
            RNA chains will be selected.
        random_seed (int | None):
            The random seed to use for subsampling. Default is None.

    Returns:
        A copy of the original cache filtered to only contain desired chains selected by
        the filtering steps. The returned cache's structure_data contains the subsampled
        monomer chains as well as all low-homology ligand chains and ligand-containing
        interfaces that pass the metric-eligibility criteria (low-homology, good ranking
        model fit, and single-residue for ligands).
    """
    logger.info("Selecting monomer set...")
    filtered_cache = deepcopy(val_dataset_cache)

    # Filter to only low-homology polymers
    keep_chain_datapoints = set()

    for pdb_id, structure_data in val_dataset_cache.structure_data.items():
        polymer_chains = [
            chain_id
            for chain_id, chain_data in structure_data.chains.items()
            if chain_data.molecule_type
            in (MoleculeType.PROTEIN, MoleculeType.RNA, MoleculeType.DNA)
        ]

        if len(polymer_chains) > 1 or len(polymer_chains) == 0:
            # Skip if there are multiple or no polymeric chains
            continue
        else:
            # Otherwise check if the one polymer chain is low-homology
            chain_id = polymer_chains[0]

            if structure_data.chains[chain_id].low_homology:
                keep_chain_datapoints.add(ChainDataPoint(pdb_id, chain_id))

    # Filter the cache to only contain the specified chains
    logger.info("Filtering cache by specified chains.")
    filter_cache_to_specified_chains(filtered_cache, keep_chain_datapoints)

    # Assign cluster IDs
    logger.info("Assigning cluster IDs.")
    add_cluster_data(filtered_cache, id_to_sequence=id_to_sequence, add_sizes=False)

    # Subsample chains by molecule type
    logger.info("Subsampling chains.")
    subsample_chains_by_type(
        filtered_cache,
        n_protein=n_protein,
        n_dna=n_dna,
        n_rna=n_rna,
        random_seed=random_seed,
    )

    # Filter by token count
    filtered_cache.structure_data = filter_by_token_count(
        filtered_cache.structure_data, max_tokens=max_token_count
    )

    # Add back valid ligand chains and interfaces
    add_ligand_data_to_monomer_cache(
        val_dataset_cache.structure_data, filtered_cache.structure_data
    )

    return filtered_cache


# TODO: Could expose more arguments?
# TODO: Add docstring!
def create_pdb_val_dataset_cache_af3(
    train_cache_path: Path,
    metadata_cache_path: Path,
    preprocessed_dir: Path,
    alignment_representatives_fasta: Path,
    output_path: Path,
    dataset_name: str,
    max_release_date: datetime.date | str = "2023-01-13",
    min_release_date: datetime.date | str = "2021-09-30",
    max_resolution: float = 4.5,
    max_polymer_chains: int = 1000,
    filter_missing_alignment: bool = True,
    missing_alignment_log: Path = None,
    max_tokens_initial: int = 2560,
    max_tokens_final: int = 2048,
    ranking_fit_threshold: float = 0.5,
    seq_identity_threshold: float = 0.4,
    tanimoto_threshold: float = 0.85,
    random_seed: int = 12345,
) -> None:
    metadata_cache = PreprocessingDataCache.from_json(metadata_cache_path)

    # TODO: Following code has quite a bit of redundancy with training code, consider
    # refactoring later
    # Read in FASTAs of all sequences in the training set
    logger.info("Scanning FASTA directories...")
    id_to_sequence = consolidate_preprocessed_fastas(preprocessed_dir)

    # Get a mapping of PDB IDs to release dates before any filtering is done
    pdb_id_to_release_date = {}
    for pdb_id, metadata in metadata_cache.structure_data.items():
        pdb_id_to_release_date[pdb_id] = metadata.release_date

    # Subset the structures in the preprocessed metadata to only the desired ones
    metadata_cache.structure_data = filter_structure_metadata_af3(
        metadata_cache.structure_data,
        max_release_date=max_release_date,
        min_release_date=min_release_date,
        max_resolution=max_resolution,
        max_polymer_chains=max_polymer_chains,
        max_tokens=max_tokens_initial,
    )

    # Create a provisional dataset training cache with extra fields
    val_dataset_cache = build_provisional_clustered_val_dataset_cache(
        preprocessing_cache=metadata_cache,
        dataset_name=dataset_name,
    )

    # Convenience wrapper that logs the number of structures filtered out
    with_log = partial(func_with_n_filtered_chain_log, logger=logger)

    # Map each target chain to an alignment representative, then filter all structures
    # without alignment representatives
    if filter_missing_alignment:
        if missing_alignment_log:
            structure_data, unmatched_entries = with_log(
                add_and_filter_alignment_representatives
            )(
                structure_cache=val_dataset_cache.structure_data,
                query_chain_to_seq=id_to_sequence,
                alignment_representatives_fasta=alignment_representatives_fasta,
                return_no_repr=True,
            )

            # Write all chains without alignment representatives to a JSON file. These
            # are excluded from training.
            with open(missing_alignment_log, "w") as f:
                # Convert the internal dataclasses to dict
                unmatched_entries = {
                    pdb_id: {chain_id: asdict(chain_data)}
                    for pdb_id, chain_data in unmatched_entries.items()
                    for chain_id, chain_data in chain_data.items()
                }

                # Format datacache-types appropriately
                unmatched_entries = format_nested_dict_for_json(unmatched_entries)

                json.dump(unmatched_entries, f, indent=4)
        else:
            structure_data = with_log(add_and_filter_alignment_representatives)(
                structure_cache=val_dataset_cache.structure_data,
                query_chain_to_seq=id_to_sequence,
                alignment_representatives_fasta=alignment_representatives_fasta,
                return_no_repr=False,
            )

        val_dataset_cache.structure_data = structure_data

    # Load the training cache which we need for homology comparisons
    train_dataset_cache = ClusteredDatasetCache.from_json(train_cache_path)

    # TODO: This is a temporary solution, MoleculeType should be parsed as a class
    # natively
    for structure_data in train_dataset_cache.structure_data.values():
        for chain_data in structure_data.chains.values():
            chain_data.molecule_type = MoleculeType[chain_data.molecule_type]

    # Get model_ranking_fit for all ligand chains
    logger.info("Fetching ligand model fit from RCSB PDB.")
    assign_ligand_model_fits(val_dataset_cache.structure_data)

    # Set low_homology attributes for all chains and interfaces
    assign_homology_labels(
        val_dataset_cache=val_dataset_cache,
        train_dataset_cache=train_dataset_cache,
        id_to_sequence=id_to_sequence,
        seq_identity_threshold=seq_identity_threshold,
        tanimoto_threshold=tanimoto_threshold,
    )

    # Set metric_eligible attributes for all interfaces
    assign_interface_metric_eligibility_labels(
        val_dataset_cache=val_dataset_cache,
        min_ranking_model_fit=ranking_fit_threshold,
    )

    # Build a validation dataset cache corresponding to the multimer set in SI 5.8
    multimer_cache = select_multimer_cache(
        val_dataset_cache=val_dataset_cache,
        id_to_sequence=id_to_sequence,
        max_token_count=max_tokens_final,
        random_seed=random_seed,
    )

    # Build a validation dataset cache corresponding to the monomer set in SI 5.8
    monomer_cache = select_monomer_cache(
        val_dataset_cache=val_dataset_cache,
        id_to_sequence=id_to_sequence,
        max_token_count=max_tokens_final,
        random_seed=random_seed,
    )

    # Subset the original cache to only the PDB-IDs in the multimer and monomer sets
    # and turn metrics on for only select chains and interfaces in those two sets
    select_final_validation_data(
        unfiltered_cache=val_dataset_cache,
        monomer_structure_data=monomer_cache.structure_data,
        multimer_structure_data=multimer_cache.structure_data,
    )

    final_stats = get_validation_summary_stats(val_dataset_cache.structure_data)

    logger.info("Final cache statistics:")
    logger.info("=" * 40)
    logger.info(f"Number of PDB-IDs: {final_stats.n_pdb_ids}")
    logger.info(f"Number of chains: {final_stats.n_chains}")
    logger.info(f"Number of low-homology chains: {final_stats.n_low_homology_chains}")
    logger.info(f"Number of scored chains: {final_stats.n_scored_chains}")
    logger.info(f"Number of interfaces: {final_stats.n_interfaces}")
    logger.info(
        f"Number of low-homology interfaces: {final_stats.n_low_homology_interfaces}"
    )
    logger.info(f"Number of scored interfaces: {final_stats.n_scored_interfaces}")

    # Write out final dataset cache
    write_datacache_to_json(val_dataset_cache, output_path)
