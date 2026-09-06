# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict
from functools import partial
from pathlib import Path

from openfold3.core.data.io.dataset_cache import (
    format_nested_dict_for_json,
    read_datacache,
    write_datacache_to_json,
)
from openfold3.core.data.io.sequence.fasta import (
    consolidate_preprocessed_fastas,
)
from openfold3.core.data.pipelines.preprocessing.caches.pdb_weighted import (
    filter_structure_metadata_of3,
)
from openfold3.core.data.primitives.caches.clustering import (
    add_cluster_data,
)
from openfold3.core.data.primitives.caches.filtering import (
    JOINT_LIGAND_EXCLUSION_SET,
    ChainDataPoint,
    InterfaceDataPoint,
    add_and_filter_alignment_representatives,
    assign_ligand_model_fits,
    assign_metric_eligibility_labels,
    build_provisional_clustered_val_dataset_cache,
    filter_id_to_seq_by_cache,
    filter_only_ligand_ligand_metrics,
    func_with_n_filtered_chain_log,
    get_validation_summary_stats,
    select_final_validation_data,
    select_one_per_cluster,
)
from openfold3.core.data.primitives.caches.format import (
    PreprocessingDataCache,
    ValidationDatasetCache,
)
from openfold3.core.data.primitives.caches.homology import assign_homology_labels
from openfold3.core.data.resources.residues import MoleculeType

logger = logging.getLogger(__name__)


def _get_interface_type(
    interface_id: str,
    structure_data,
) -> str | None:
    """Get the interface type string from an interface ID and structure data."""
    chain_1, chain_2 = interface_id.split("_")
    molecule_types = [
        structure_data.chains[chain_1].molecule_type,
        structure_data.chains[chain_2].molecule_type,
    ]

    n_protein = molecule_types.count(MoleculeType.PROTEIN)
    n_dna = molecule_types.count(MoleculeType.DNA)
    n_rna = molecule_types.count(MoleculeType.RNA)
    n_ligand = molecule_types.count(MoleculeType.LIGAND)

    if n_protein == 2:
        return "protein_protein"
    elif n_protein == 1 and n_dna == 1:
        return "protein_dna"
    elif n_dna == 2:
        return "dna_dna"
    elif n_protein == 1 and n_ligand == 1:
        return "protein_ligand"
    elif n_dna == 1 and n_ligand == 1:
        return "dna_ligand"
    elif n_ligand == 2:
        return "ligand_ligand"
    elif n_protein == 1 and n_rna == 1:
        return "protein_rna"
    elif n_rna == 2:
        return "rna_rna"
    elif n_dna == 1 and n_rna == 1:
        return "dna_rna"
    elif n_rna == 1 and n_ligand == 1:
        return "rna_ligand"
    else:
        return None


def select_multimer_cache(
    val_dataset_cache: ValidationDatasetCache,
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
) -> list[InterfaceDataPoint]:
    """Selects multimer interfaces following AF3 SI 5.8.

    Collects metric-eligible interfaces, selects one representative per cluster, then
    subsamples by interface type.

    Args:
        val_dataset_cache:
            The validation dataset cache.
        max_token_count:
            Maximum token count for filtering structures at the very end. (The reason
            for doing this at the end is not clear but we keep it for consistency with
            the SI.)
        n_protein_protein:
            Number of protein-protein interfaces to sample.
        n_protein_dna:
            Number of protein-DNA interfaces to sample.
        n_dna_dna:
            Number of DNA-DNA interfaces to sample.
        n_protein_ligand:
            Number of protein-ligand interfaces to sample.
        n_dna_ligand:
            Number of DNA-ligand interfaces to sample.
        n_ligand_ligand:
            Number of ligand-ligand interfaces to sample.
        n_protein_rna:
            Number of protein-RNA interfaces (None = all).
        n_rna_rna:
            Number of RNA-RNA interfaces (None = all).
        n_dna_rna:
            Number of DNA-RNA interfaces (None = all).
        n_rna_ligand:
            Number of RNA-ligand interfaces (None = all).
        random_seed:
            Random seed for reproducibility.

    Returns:
        List of InterfaceDataPoint objects for the selected interfaces.
    """
    logger.info("Selecting multimer set...")

    if random_seed is not None:
        random.seed(random_seed)

    # Collect metric-eligible interfaces by type
    interface_type_to_datapoints: dict[str, list[InterfaceDataPoint]] = defaultdict(
        list
    )

    for pdb_id, structure_data in val_dataset_cache.structure_data.items():
        for interface_id, interface_data in structure_data.interfaces.items():
            if not interface_data.metric_eligible:
                continue

            interface_type = _get_interface_type(interface_id, structure_data)
            if interface_type is None:
                continue

            interface_type_to_datapoints[interface_type].append(
                InterfaceDataPoint(pdb_id, interface_id)
            )

    # Define sample counts by type
    n_samples_by_type = {
        "protein_protein": n_protein_protein,
        "protein_dna": n_protein_dna,
        "dna_dna": n_dna_dna,
        "protein_ligand": n_protein_ligand,
        "dna_ligand": n_dna_ligand,
        "ligand_ligand": n_ligand_ligand,
        "protein_rna": n_protein_rna,
        "rna_rna": n_rna_rna,
        "dna_rna": n_dna_rna,
        "rna_ligand": n_rna_ligand,
    }

    # For each type: select one per cluster, then subsample
    selected_interfaces: list[InterfaceDataPoint] = []

    for interface_type, datapoints in interface_type_to_datapoints.items():
        # Select one per cluster
        representatives = select_one_per_cluster(
            datapoints, val_dataset_cache, random_seed
        )

        # Subsample (None means keep all)
        n_samples = n_samples_by_type.get(interface_type)
        if n_samples is not None and len(representatives) > n_samples:
            representatives = random.sample(representatives, n_samples)

        selected_interfaces.extend(representatives)

    # Subsample by max token count
    selected_interfaces = [
        dp
        for dp in selected_interfaces
        if val_dataset_cache.structure_data[dp.pdb_id].token_count <= max_token_count
    ]

    logger.info(f"Selected {len(selected_interfaces)} interfaces for multimer set.")

    return selected_interfaces


def select_monomer_cache(
    val_dataset_cache: ValidationDatasetCache,
    max_token_count: int = 2048,
    n_protein: int = 40,
    n_dna: int | None = None,
    n_rna: int | None = None,
    random_seed: int | None = None,
) -> list[ChainDataPoint]:
    """Selects monomer chains following AF3 SI 5.8.

    Collects metric-eligible single-polymer chains, selects one representative per
    cluster, then subsamples by molecule type.

    Args:
        val_dataset_cache:
            The validation dataset cache.
        max_token_count:
            Maximum token count for filtering structures at the very end. (The reason
            for doing this at the end is not clear but we keep it for consistency with
            the SI.)
        n_protein:
            Number of protein chains to sample.
        n_dna:
            Number of DNA chains to sample (None = all representatives).
        n_rna:
            Number of RNA chains to sample (None = all representatives).
        random_seed:
            Random seed for reproducibility.

    Returns:
        List of ChainDataPoint objects for the selected monomer chains.
    """
    logger.info("Selecting monomer set...")

    if random_seed is not None:
        random.seed(random_seed)

    # Collect metric-eligible monomer chains by molecule type
    chain_type_to_datapoints: dict[MoleculeType, list[ChainDataPoint]] = defaultdict(
        list
    )

    for pdb_id, structure_data in val_dataset_cache.structure_data.items():
        # Get polymer chains
        polymer_chains = [
            chain_id
            for chain_id, chain_data in structure_data.chains.items()
            if chain_data.molecule_type
            in (MoleculeType.PROTEIN, MoleculeType.DNA, MoleculeType.RNA)
        ]

        # Must be single-polymer for monomer set
        if len(polymer_chains) != 1:
            continue

        chain_id = polymer_chains[0]
        chain_data = structure_data.chains[chain_id]

        if chain_data.metric_eligible:
            chain_type_to_datapoints[chain_data.molecule_type].append(
                ChainDataPoint(pdb_id, chain_id)
            )

    # Define sample counts by type
    n_samples_by_type = {
        MoleculeType.PROTEIN: n_protein,
        MoleculeType.DNA: n_dna,
        MoleculeType.RNA: n_rna,
    }

    # For each type: select one per cluster, then subsample
    selected_chains: list[ChainDataPoint] = []

    for chain_type, datapoints in chain_type_to_datapoints.items():
        # Select one per cluster
        representatives = select_one_per_cluster(
            datapoints, val_dataset_cache, random_seed
        )

        # Subsample (None means keep all)
        n_samples = n_samples_by_type.get(chain_type)
        if n_samples is not None and len(representatives) > n_samples:
            representatives = random.sample(representatives, n_samples)

        selected_chains.extend(representatives)

    # Subsample by max token count
    selected_chains = [
        dp
        for dp in selected_chains
        if val_dataset_cache.structure_data[dp.pdb_id].token_count <= max_token_count
    ]

    logger.info(f"Selected {len(selected_chains)} chains for monomer set.")

    return selected_chains


# TODO: Could expose more arguments?
# TODO: Add docstring!
def create_pdb_val_dataset_cache_of3(
    metadata_cache_path: Path,
    preprocessed_dir: Path,
    train_cache_path: Path,
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
    val_id_to_sequence = consolidate_preprocessed_fastas(preprocessed_dir)

    # Get a mapping of PDB IDs to release dates before any filtering is done
    pdb_id_to_release_date = {}
    for pdb_id, metadata in metadata_cache.structure_data.items():
        pdb_id_to_release_date[pdb_id] = metadata.release_date

    # Subset the structures in the preprocessed metadata to only the desired ones
    metadata_cache.structure_data = filter_structure_metadata_of3(
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
                query_chain_to_seq=val_id_to_sequence,
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
                query_chain_to_seq=val_id_to_sequence,
                alignment_representatives_fasta=alignment_representatives_fasta,
                return_no_repr=False,
            )

        val_dataset_cache.structure_data = structure_data

    # Subset dictionary to only chains included in the validation cache
    val_id_to_sequence = filter_id_to_seq_by_cache(
        val_dataset_cache.structure_data, val_id_to_sequence
    )

    # Load training cache for homology assignment
    logger.info(f"Loading training cache from {train_cache_path}...")
    train_dataset_cache = read_datacache(train_cache_path)

    # Read training set sequences from the same preprocessed directory
    train_id_to_sequence = consolidate_preprocessed_fastas(preprocessed_dir)
    train_id_to_sequence = filter_id_to_seq_by_cache(
        train_dataset_cache.structure_data, train_id_to_sequence
    )

    # TODO: This is a temporary solution, MoleculeType should be parsed as a class
    # natively
    for structure_data in train_dataset_cache.structure_data.values():
        for chain_data in structure_data.chains.values():
            chain_data.molecule_type = MoleculeType[chain_data.molecule_type]

    # Get model_ranking_fit for all ligand chains
    logger.info("Fetching ligand model fit from RCSB PDB.")
    assign_ligand_model_fits(val_dataset_cache.structure_data)

    logger.info("Assigning cluster IDs.")
    add_cluster_data(
        val_dataset_cache, id_to_sequence=val_id_to_sequence, add_sizes=False
    )

    # Set low_homology attributes for all chains and interfaces
    assign_homology_labels(
        val_dataset_cache=val_dataset_cache,
        train_dataset_cache=train_dataset_cache,
        val_id_to_sequence=val_id_to_sequence,
        train_id_to_sequence=train_id_to_sequence,
        seq_identity_threshold=seq_identity_threshold,
        tanimoto_threshold=tanimoto_threshold,
    )

    # Set metric_eligible attributes for all chains and interfaces
    assign_metric_eligibility_labels(
        val_dataset_cache=val_dataset_cache,
        min_ranking_model_fit=ranking_fit_threshold,
        lig_exclusion_list=JOINT_LIGAND_EXCLUSION_SET,
    )

    # Select multimer interfaces (one per cluster, then subsampled by type)
    selected_interfaces = select_multimer_cache(
        val_dataset_cache=val_dataset_cache,
        max_token_count=max_tokens_final,
        random_seed=random_seed,
    )

    # Select monomer chains (one per cluster, then subsampled by type)
    selected_chains = select_monomer_cache(
        val_dataset_cache=val_dataset_cache,
        max_token_count=max_tokens_final,
        random_seed=random_seed,
    )

    monomer_pdb_ids = {dp.pdb_id for dp in selected_chains}
    multimer_pdb_ids = {dp.pdb_id for dp in selected_interfaces}

    logger.info(
        "Overlap between monomer and multimer PDB IDs: "
        f"{len(monomer_pdb_ids & multimer_pdb_ids)}",
    )

    # Subset cache to selected PDB IDs and mark cluster representatives
    # (priority to selected chains/interfaces, then fill in additional representatives)
    select_final_validation_data(
        val_dataset_cache=val_dataset_cache,
        selected_chains=selected_chains,
        selected_interfaces=selected_interfaces,
        random_seed=random_seed,
    )

    val_dataset_cache.structure_data = with_log(filter_only_ligand_ligand_metrics)(
        val_dataset_cache.structure_data
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
