"""
Centralized module for pre-assembled workflows corresponding to structure cleanup
procedures of different models.
"""

# TODO: organize this file so that we separate components for creating the metadata
# cache for each dataset
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import biotite.structure as struc
import biotite.structure.io as strucio
import boto3
import numpy as np
import pandas as pd
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFBlock, CIFFile
from rdkit import Chem
from tqdm import tqdm

from openfold3.core.data.io.s3 import download_file_from_s3
from openfold3.core.data.io.sequence.fasta import write_multichain_fasta
from openfold3.core.data.io.structure.atom_array import write_atomarray_to_npz
from openfold3.core.data.io.structure.cif import (
    SkippedStructure,
    parse_mmcif,
    parse_target_structure,
    write_structure,
)
from openfold3.core.data.io.structure.mol import write_annotated_sdf
from openfold3.core.data.io.structure.pdb import (
    parse_pdb_af2,
    parse_protein_monomer_pdb_tmp,
)
from openfold3.core.data.io.utils import encode_numpy_types
from openfold3.core.data.pipelines.preprocessing.utils import SharedSet
from openfold3.core.data.primitives.caches.format import (
    DisorderedPreprocessingDataCache,
    DisorderedPreprocessingStructureData,
    PreprocessingChainData,
    PreprocessingDataCache,
    PreprocessingStructureData,
    ProteinMonomerDatasetCache,
)
from openfold3.core.data.primitives.permutation.mol_labels import (
    assign_mol_permutation_ids,
)
from openfold3.core.data.primitives.structure.alignment import (
    calculate_distance_clash_map,
    coalign_atom_arrays,
)
from openfold3.core.data.primitives.structure.cleanup import (
    canonicalize_atom_order,
    convert_MSE_to_MET,
    fix_arginine_naming,
    remove_chains_with_CA_gaps,
    remove_clashing_chains,
    remove_covalent_nonprotein_chains,
    remove_crystallization_aids,
    remove_fully_unknown_polymers,
    remove_hydrogens,
    remove_non_CCD_atoms,
    remove_small_polymers,
    remove_std_residue_terminal_atoms,
    remove_waters,
    subset_large_structure,
)
from openfold3.core.data.primitives.structure.component import (
    AnnotatedMol,
    assign_component_ids_from_metadata,
    get_component_info,
    mol_from_atomarray,
    mol_from_ccd_entry,
)
from openfold3.core.data.primitives.structure.conformer import (
    resolve_and_format_fallback_conformer,
)
from openfold3.core.data.primitives.structure.interface import (
    get_interface_chain_id_pairs,
)
from openfold3.core.data.primitives.structure.labels import (
    get_chain_to_author_chain_dict,
    get_chain_to_entity_dict,
    get_chain_to_molecule_type_dict,
    get_chain_to_pdb_chain_dict,
    remove_transfer_annotations,
)
from openfold3.core.data.primitives.structure.metadata import (
    get_chain_to_canonical_seq_dict,
    get_cif_block,
    get_experimental_method,
    get_pdb_id,
    get_release_date,
    get_resolution,
)
from openfold3.core.data.primitives.structure.tokenization import (
    get_token_count,
    tokenize_atom_array,
)
from openfold3.core.data.primitives.structure.unresolved import add_unresolved_atoms
from openfold3.core.data.resources.residues import MoleculeType

logger = logging.getLogger(__name__)

_worker_session = None


# ---- Core PDB metadata cache pipelines ----
def _init_worker(profile_name: str = "openfold") -> None:
    """Initialize the boto3 session in each worker."""
    global _worker_session
    _worker_session = boto3.Session(profile_name=profile_name)


def cleanup_structure_af3(
    atom_array: AtomArray,
    cif_data: CIFBlock,
    ccd: CIFFile,
    random_seed: int | None = None,
) -> AtomArray:
    """Cleans up a structure following the AlphaFold3 SI and formats it for training.

    This function first applies all cleaning steps outlined in the AlphaFold3 SI 2.5.4.
    The only non-applied filters are the number-of-chain filter, which is handled before
    passing to this function, as well as release date and resolution filters which are
    deferred to the training cache generation script for easier adjustment in the
    future.

    Second, this function also adds all unresolved atoms to the AtomArray as explicit
    atoms with NaN coordinates, and removes terminal atoms for standard residues to
    ensure a consistent token -> n_atom mapping that is expected by the model.

    Args:
        atom_array:
            AtomArray containing the structure to clean up
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see
            `metadata_extraction.get_cif_block`)
        ccd:
            CIFFile containing the parsed CCD (components.cif)
        random_seed:
            Random seed for reproducibility in large-assembly-subsetting.

    Returns:
        AtomArray with all cleaning steps applied
    """
    atom_array = atom_array.copy()

    ## Structure cleanup
    convert_MSE_to_MET(atom_array)
    fix_arginine_naming(atom_array)  #
    atom_array = remove_waters(atom_array)

    if get_experimental_method(cif_data) == "X-RAY DIFFRACTION":
        atom_array = remove_crystallization_aids(atom_array)

    atom_array = remove_hydrogens(atom_array)
    atom_array = remove_small_polymers(atom_array, max_residues=3)
    atom_array = remove_fully_unknown_polymers(atom_array)
    atom_array = remove_clashing_chains(
        atom_array, clash_distance=1.7, clash_percentage=0.3
    )
    atom_array = remove_non_CCD_atoms(atom_array, ccd)
    atom_array = canonicalize_atom_order(atom_array, ccd)
    atom_array = remove_chains_with_CA_gaps(atom_array, distance_threshold=10.0)

    # TODO: Could change this to use get_chain_count from Biotite which will be more
    # robust for non-renumbered chains
    # Subset bioassemblies larger than 20 chains
    if len(np.unique(atom_array.chain_id)) > 20:
        # Tokenization is required for large-structure subsetting
        tokenize_atom_array(atom_array)
        atom_array = subset_large_structure(
            atom_array=atom_array,
            n_chains=20,
            interface_distance_threshold=15.0,
            random_seed=random_seed,
        )

    ## Structure formatting
    # Add unresolved atoms explicitly with NaN coordinates
    atom_array = add_unresolved_atoms(atom_array, cif_data, ccd)

    # Remove terminal atoms to ensure consistent atom count for standard tokens in the
    # model
    atom_array = remove_std_residue_terminal_atoms(atom_array)

    return atom_array


# TODO: extend docstring
def extract_chain_and_interface_metadata_af3(
    atom_array: AtomArray, cif_data: CIFBlock
) -> dict:
    """Extracts basic, chain and interface metadata from a structure.

    This extracts general metadata from the structure, as well as chain-level metadata
    and interface-level metadata.

    Args:
        atom_array:
            AtomArray containing the structure to extract metadata from
        cif_data:
            Parsed mmCIF data of the structure. Note that this expects a CIFBlock which
            requires one prior level of indexing into the CIFFile, (see
            `metadata_extraction.get_cif_block`)

    Returns:
        dict containing the extracted metadata
    """

    metadata_dict = {}

    # Get basic metadata
    metadata_dict["release_date"] = get_release_date(cif_data).strftime("%Y-%m-%d")
    metadata_dict["resolution"] = get_resolution(cif_data)
    metadata_dict["experimental_method"] = get_experimental_method(cif_data)
    metadata_dict["token_count"] = get_token_count(atom_array)

    # NOTE: This could be reduced to only the critical information, currently some
    # chain IDs are put in for easier manual interpretability
    # |
    # V
    # Get chain-level metadata
    chain_to_pdb_chain = get_chain_to_pdb_chain_dict(atom_array)
    chain_to_author_chain = get_chain_to_author_chain_dict(atom_array)
    chain_to_entity = get_chain_to_entity_dict(atom_array)
    chain_to_molecule_type = get_chain_to_molecule_type_dict(atom_array)

    # Take any key set to get all chains
    all_chains = set(chain_to_pdb_chain.keys())

    metadata_dict["chains"] = {}
    for chain_id in sorted(all_chains):
        metadata_dict["chains"][chain_id] = {
            "label_asym_id": chain_to_pdb_chain[chain_id],
            "auth_asym_id": chain_to_author_chain[chain_id],
            "entity_id": chain_to_entity[chain_id],
            "molecule_type": chain_to_molecule_type[chain_id],
        }

    metadata_dict["interfaces"] = get_interface_chain_id_pairs(
        atom_array, distance_threshold=5.0
    )

    return metadata_dict


def extract_component_data_af3(
    atom_array: AtomArray,
    ccd: CIFFile,
    pdb_id: str,
    sdf_out_dir: Path,
    skip_components: set | SharedSet | None = None,
) -> tuple[dict, dict]:
    """Extracts component data from a structure.

    This extraxts all "components" from a structure, which are standard residues,
    standard ligands, and non-standard (multi-residue or any ligand that can not be
    represented by a single CCD code) ligands. For each unique component, an RDKit
    reference molecule is created alongside a fallback conformer that is either computed
    using RDKit's reference conformer generation (see AF3 SI 2.8), or taken from the
    "ideal" or "model" CCD coordinates.

    Args:
        atom_array:
            AtomArray containing the structure to extract components from
        ccd:
            CIFFile containing the parsed CCD (components.cif)
        pdb_id:
            PDB ID of the structure
        sdf_out_dir:
            Directory to write the reference molecule SDF files to
        skip_components:
            Set of components to skip, if any (useful to avoid repeated processing of
            components e.g. by using a SharedSet)

    Returns:
        Tuple containing:
            - A dictionary mapping chain IDs to the corresponding component IDs.
                Component IDs are either CCD codes or formatted as
                "{pdb_id}_{entity_id}" for non-standard ligands.
            - A dictionary containing metadata for each component:
                - "conformer_gen_strategy": The strategy used to generate the conformer
                - "fallback_conformer_pdb_id": The PDB ID of the fallback conformer
                - "canonical_smiles": The canonical SMILES of the component
    """

    def get_reference_molecule_metadata(
        mol: AnnotatedMol,
        conformer_strategy: Literal["default", "random_init", "use_fallback"],
        residue_count: int,
    ) -> dict:
        """Convenience function to return the metadata for a reference molecule."""
        conf_metadata = {
            "residue_count": residue_count,
            "conformer_gen_strategy": conformer_strategy,
        }

        if mol.HasProp("model_pdb_id"):
            fallback_conformer_pdb_id = mol.GetProp("model_pdb_id")
            if fallback_conformer_pdb_id == "?":
                fallback_conformer_pdb_id = None
        else:
            fallback_conformer_pdb_id = None

        conf_metadata["fallback_conformer_pdb_id"] = fallback_conformer_pdb_id
        conf_metadata["canonical_smiles"] = Chem.MolToSmiles(mol)

        return conf_metadata

    if skip_components is None:
        skip_components = set()

    # Instantiate output dicts
    chain_to_component_id = {}
    reference_mol_metadata = {}

    # Get all different types of components
    (
        residue_components,
        std_ligands_to_chains,
        non_std_ligands_to_chains,
        non_std_ligands_to_rescount,
    ) = get_component_info(atom_array)

    # Assign IDs to non-standard components based on the PDB ID and entity ID
    non_std_ligands_to_chains = {
        f"{pdb_id}_{entity}": chains
        for entity, chains in non_std_ligands_to_chains.items()
    }
    non_std_ligands_to_rescount = {
        f"{pdb_id}_{entity}": rescount
        for entity, rescount in non_std_ligands_to_rescount.items()
    }

    all_ligands_to_chains = {**std_ligands_to_chains, **non_std_ligands_to_chains}

    # Track which ligand chain corresponds to which ligand ID
    for mol_id, chains in all_ligands_to_chains.items():
        for chain in chains:
            chain_to_component_id[chain] = mol_id

    # Create a ccd_id: rdkit Mol mapping for all components, so that we can run
    # conformer generation jointly
    all_component_mols = {}

    # Start with standard components
    std_ligand_ccd_ids = list(std_ligands_to_chains.keys())
    std_component_ccd_ids = std_ligand_ccd_ids + residue_components
    std_component_ccd_ids = [
        c for c in std_component_ccd_ids if c not in skip_components
    ]

    for ccd_id in std_component_ccd_ids:
        mol = mol_from_ccd_entry(ccd_id, ccd)
        all_component_mols[ccd_id] = mol

    # Add non-standard ligands
    non_std_ligand_ids = list(non_std_ligands_to_chains.keys())
    # (NOTE: non-std ligands are not shared between structures so this should not be
    # strictly needed)
    non_std_ligand_ids = [c for c in non_std_ligand_ids if c not in skip_components]

    for mol_id in non_std_ligand_ids:
        # Compute molecule by arbitrarily taking the first chain (all should be the same
        # if entity ID is the same)
        entity_id = int(mol_id.split("_")[1])
        entity_atom_array = atom_array[atom_array.entity_id == entity_id]
        first_ligand = entity_atom_array[
            entity_atom_array.chain_id == all_ligands_to_chains[mol_id][0]
        ]
        mol = mol_from_atomarray(first_ligand)
        all_component_mols[mol_id] = mol

    # Generate conformer metadata for all components and write SDF files with reference
    # conformer coordinates
    for mol_id, mol in all_component_mols.items():
        residue_count = non_std_ligands_to_rescount.get(mol_id, 1)
        mol, conformer_strategy = resolve_and_format_fallback_conformer(mol)
        reference_mol_metadata[mol_id] = get_reference_molecule_metadata(
            mol, conformer_strategy, residue_count
        )

        # Write SDF file
        sdf_out_path = sdf_out_dir / f"{mol_id}.sdf"
        write_annotated_sdf(mol, sdf_out_path)

    return chain_to_component_id, reference_mol_metadata


def preprocess_structure_and_write_outputs_af3(
    input_cif: Path,
    ccd: CIFFile,
    out_dir: Path,
    reference_mol_out_dir: Path,
    output_formats: list[Literal["npz", "cif", "bcif", "pkl"]],
    max_polymer_chains: int | None = None,
    skip_components: set | SharedSet | None = None,
    random_seed: int | None = None,
) -> tuple[dict, dict]:
    """Wrapper function to preprocess a single structure for the AF3 data pipeline.

    This will parse the input CIF file, clean up the structure, extract metadata, and
    write out the cleaned-up structure to a binary CIF file, as well as all the sequence
    information to a FASTA file.

    Args:
        input_cif:
            Path to the input CIF file.
        ccd:
            CIFFile containing the parsed CCD (components.cif)
        out_dir:
            Path to the output directory.
        reference_mol_out_dir:
            Path to the output directory that reference molecule SDF files (specifying
            the molecular graph for each ligand as well as a fallback conformer for use
            in featurization) are written to.
        output_formats:
            What formats to write the output files to. Allowed values are "cif", "bcif",
            "npz", and "pkl".
        max_polymer_chains:
            The maximum number of polymer chains in the first bioassembly after which a
            structure is skipped by the parser.
        skip_components:
            A set of components to skip, if any. Useful to avoid repeated processing of
            components e.g. by using a SharedSet.
        random_seed:
            Random seed for reproducibility in large-assembly-subsetting.


    Returns:
        Tuple containing:
            - A dictionary containing the structure metadata, including chain-level
                metadata and interface metadata:
                pdb_id: {
                    "release_date": str,
                    "resolution": float,
                    "token_count": int,
                    "chains": {
                        chain_id: {
                            "label_asym_id": str,
                            "auth_asym_id": str,
                            "entity_id": int,
                            "molecule_type": str,
                            "reference_mol_id": str
                        },
                    "interfaces": [(chain_id1, chain_id2), ...]
                }
            - A dictionary containing metadata for each component:
                - "conformer_gen_strategy": The strategy used to generate the conformer
                - "fallback_conformer_pdb_id": The PDB ID of the fallback conformer
                - "canonical_smiles": The canonical SMILES of the component
    """
    # Log how long this is taking
    start_time = time.perf_counter()

    # Parse the input CIF file
    parsed_mmcif = parse_mmcif(
        input_cif,
        expand_bioassembly=True,
        include_bonds=True,
        renumber_chain_ids=True,
        max_polymer_chains=max_polymer_chains,
    )
    cif_file = parsed_mmcif.structure_file

    # Basic structure-level metadata
    cif_data = get_cif_block(cif_file)
    pdb_id = get_pdb_id(cif_file)
    release_date = get_release_date(cif_data).strftime("%Y-%m-%d")

    # Handle structures that got skipped due to max_polymer_chains
    if isinstance(parsed_mmcif, SkippedStructure):
        logger.info(f"Skipping structure {pdb_id}: {parsed_mmcif.reason}")

        return {
            pdb_id: {
                "release_date": release_date,
                "status": f"skipped: {parsed_mmcif.reason}",
            }
        }, {}
    else:
        atom_array = parsed_mmcif.atom_array

    # Cleanup structure and extract metadata
    atom_array = cleanup_structure_af3(
        atom_array=atom_array, cif_data=cif_data, ccd=ccd, random_seed=random_seed
    )

    chain_int_metadata_dict = extract_chain_and_interface_metadata_af3(
        atom_array, cif_data
    )

    chain_to_ligand_ids, ref_mol_metadata_dict = extract_component_data_af3(
        atom_array,
        ccd,
        pdb_id,
        reference_mol_out_dir,
        skip_components=skip_components,
    )

    # Add chain-to-ligand-ID mapping to metadata
    for chain_id, ligand_id in chain_to_ligand_ids.items():
        chain_int_metadata_dict["chains"][chain_id]["reference_mol_id"] = ligand_id

    structure_metadata_dict = {
        pdb_id: {
            "release_date": release_date,
            "status": "success",
            **chain_int_metadata_dict,
        }
    }

    # TODO: Clean this up
    # Set permutation labels
    per_chain_metadata = {
        chain_id: PreprocessingChainData(
            label_asym_id=chain_data["label_asym_id"],
            auth_asym_id=chain_data["auth_asym_id"],
            entity_id=chain_data["entity_id"],
            molecule_type=chain_data["molecule_type"],
            reference_mol_id=chain_data.get("reference_mol_id"),
        )
        for chain_id, chain_data in chain_int_metadata_dict["chains"].items()
    }
    assign_component_ids_from_metadata(
        atom_array, per_chain_metadata=per_chain_metadata
    )
    tokenize_atom_array(atom_array)
    atom_array = assign_mol_permutation_ids(atom_array)

    # Get canonicalized sequence for each chain
    chain_to_canonical_seq = get_chain_to_canonical_seq_dict(
        atom_array, cif_data, multi_letter_res_to_X=True, ccd=ccd
    )

    # Write CIF and FASTA outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    for output_format in output_formats:
        out_path = out_dir / f"{pdb_id}.{output_format}"
        write_structure(atom_array, out_path, data_block=pdb_id)

    out_fasta_path = out_dir / f"{pdb_id}.fasta"
    write_multichain_fasta(out_fasta_path, chain_to_canonical_seq)

    end_time = time.perf_counter()

    logger.debug(f"Processing {pdb_id} took {end_time - start_time:.2f} seconds.")

    return structure_metadata_dict, ref_mol_metadata_dict


class _AF3PreprocessingWrapper:
    """Wrapper class that fills in all the constant arguments and adds logging.

    This wrapper around `preprocess_structure_and_write_outputs_af3` is needed for
    multiprocessing, so that we can pass the constant arguments in a convenient way
    catch any errors that would crash the workers, and change the function call to
    accept a single Iterable. In addition, the wrapper updates the set passed to
    skip_components in-place after the function completion, so that this information is
    immediately available to other workers when passing a SharedSet.

    The wrapper is written as a class object because multiprocessing doesn't support
    decorator-like nested functions.

    Attributes:
        ccd:
            The CIFFile object.
        reference_mol_out_dir:
            The directory where reference molecules are stored.
        max_polymer_chains:
            The maximum number of polymer chains in the first bioassembly after which a
            structure is skipped by the parser.
        skip_components:
            A set of components to skip, if any.
        output_formats:
            What formats to write the output files to. Allowed values are "cif", "bcif",
            and "pkl".
        random_seed:
            Random seed for reproducibility in large-assembly-subsetting.
    """

    def __init__(
        self,
        ccd: CIFFile,
        reference_mol_out_dir: Path,
        max_polymer_chains: int | None,
        skip_components: set | SharedSet | None,
        output_formats: list[Literal["npz", "cif", "bcif", "pkl"]],
        random_seed: int | None = None,
    ):
        self.ccd = ccd
        self.reference_mol_out_dir = reference_mol_out_dir
        self.max_polymer_chains = max_polymer_chains
        self.skip_components = skip_components
        self.output_formats = output_formats
        self.random_seed = random_seed

    def __call__(self, paths: tuple[Path, Path]) -> tuple[dict, dict]:
        cif_file, out_dir = paths

        logger.debug(f"Processing {cif_file.stem}")
        try:
            structure_metadata_dict, ref_mol_metadata_dict = (
                preprocess_structure_and_write_outputs_af3(
                    input_cif=cif_file,
                    out_dir=out_dir,
                    ccd=self.ccd,
                    reference_mol_out_dir=self.reference_mol_out_dir,
                    max_polymer_chains=self.max_polymer_chains,
                    skip_components=self.skip_components,
                    output_formats=self.output_formats,
                    random_seed=self.random_seed,
                )
            )

            # Update the set of processed components in-place
            processed_mols = set(ref_mol_metadata_dict.keys())
            self.skip_components.update(processed_mols)

            logger.debug(f"Finished processing {cif_file.stem}")
            return structure_metadata_dict, ref_mol_metadata_dict

        except Exception as e:
            tb = traceback.format_exc()  # Get the full traceback
            logger.warning(
                "-" * 40
                + "\n"
                + f"Failed to process {cif_file.stem}: {str(e)}\n"
                + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                + "-" * 40
            )
            pdb_id = cif_file.stem

            output_dict = {pdb_id: {"status": "failed"}}
            empty_conformer_dict = {}

            return output_dict, empty_conformer_dict


def preprocess_cif_dir_af3(
    cif_dir: Path,
    ccd_path: Path,
    out_dir: Path,
    max_polymer_chains: int | None = None,
    num_workers: int | None = None,
    chunksize: int = 20,
    output_formats: list[Literal["npz", "cif", "bcif", "pkl"]] = False,
    random_seed: int | None = None,
    early_stop: int | None = None,
) -> None:
    """Preprocesses a directory of PDB files following the AlphaFold3 SI.

    This function applies the full AlphaFold3 structure cleanup pipeline to a directory
    of PDB files. The output is a set of cleaned-up structure files in the output
    directory, as well as a set of metadata files containing chain-level metadata and
    reference molecule metadata for all components.

    Args:
        cif_dir:
            Path to the directory containing the PDB files to preprocess.
        ccd_path:
            Path to the CCD file.
        out_dir:
            Path to the output directory.
        max_polymer_chains:
            The maximum number of polymer chains in the first bioassembly after which a
            structure is skipped by the parser.
        num_workers:
            Number of workers to use for parallel processing. Use None for all available
            CPUs, and 0 for a single process (not using the multiprocessing module).
        chunksize:
            Number of CIF files to process in each worker task.
        output_formats:
            What formats to write the output files to. Allowed values are "npz", "cif",
            "bcif", and "pkl".
        random_seed:
            Random seed for reproducibility in large-assembly-subsetting.
        early_stop:
            Stop after processing this many CIFs. Only used for debugging.
    """
    logger.debug("Reading CCD file")
    ccd = CIFFile.read(ccd_path)

    logger.debug("Reading CIF files")
    cif_files = [file for file in tqdm(cif_dir.glob("*.cif"))]

    if early_stop is not None:
        cif_files = cif_files[:early_stop]

    output_dict = {
        "structure_data": {},
        "reference_molecule_data": {},
    }

    # Set up output directories
    reference_mol_out_dir = out_dir / "reference_mols"
    reference_mol_out_dir.mkdir(parents=True, exist_ok=True)

    cif_out_dir = out_dir / "cif_files"
    cif_out_dir.mkdir(parents=True, exist_ok=True)

    cif_output_dirs = []

    for cif_file in tqdm(cif_files):
        pdb_id = cif_file.stem
        out_subdir = cif_out_dir / pdb_id
        cif_output_dirs.append(out_subdir)

    processed_mol_ids = SharedSet() if num_workers != 0 else set()

    # Load the preprocessing function with the constant arguments
    wrapped_preprocessing_func = _AF3PreprocessingWrapper(
        ccd=ccd,
        reference_mol_out_dir=reference_mol_out_dir,
        max_polymer_chains=max_polymer_chains,
        skip_components=processed_mol_ids,
        output_formats=output_formats,
        random_seed=random_seed,
    )

    def update_output_dicts(structure_metadata_dict: dict, ref_mol_metadata_dict: dict):
        """Convenience function to update the output dicts with the metadata."""
        output_dict["structure_data"].update(structure_metadata_dict)
        output_dict["reference_molecule_data"].update(ref_mol_metadata_dict)

        processed_mol_ids.update(ref_mol_metadata_dict.keys())

    ## Preprocess all CIF files, cleaning up structures and writing out metadata

    # Use a single process if num_workers is 0 (for debugging)
    logger.debug("Starting processing.")
    if num_workers == 0:
        for structure_metadata_dict, ref_mol_metadata_dict in tqdm(
            map(wrapped_preprocessing_func, zip(cif_files, cif_output_dirs)),
            total=len(cif_files),
        ):
            update_output_dicts(structure_metadata_dict, ref_mol_metadata_dict)

    else:
        with mp.Pool(num_workers) as pool:
            for i, (structure_metadata_dict, ref_mol_metadata_dict) in enumerate(
                tqdm(
                    pool.imap_unordered(
                        wrapped_preprocessing_func,
                        zip(cif_files, cif_output_dirs),
                        chunksize=chunksize,
                    ),
                    total=len(cif_files),
                )
            ):
                update_output_dicts(structure_metadata_dict, ref_mol_metadata_dict)

                # Periodically save the output dict to avoid losing data in case of a
                # crash
                if i % 1000 == 0:
                    with open(out_dir / "metadata.json", "w") as f:
                        json.dump(output_dict, f, indent=4, default=encode_numpy_types)

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(output_dict, f, indent=4, default=encode_numpy_types)


# ---- protein monomer preprocessing pipelines ----
# TODO: combine local and S3 preparsing
class _WrapProcessMonomerDistillStructure:
    def __init__(self, s3_config: dict, output_dir: Path):
        self.s3_config = s3_config
        self.output_dir = output_dir

    def __call__(self, pdb_id):
        try:
            with NamedTemporaryFile() as temp_file:
                prefix = self.s3_config["prefix"]
                prefix = f"{prefix}/{pdb_id}"
                global _worker_session
                download_file_from_s3(
                    bucket=self.s3_config["bucket"],
                    prefix=prefix,
                    filename="best_structure_relaxed.pdb",
                    outfile=temp_file.name,
                    session=_worker_session,
                )
                _, atom_array = parse_protein_monomer_pdb_tmp(temp_file.name)
                id_outdir = self.output_dir / pdb_id
                id_outdir.mkdir(parents=True, exist_ok=True)
                write_structure(atom_array, id_outdir / f"{pdb_id}.pkl")
        except Exception as e:
            tb = traceback.format_exc()  # Get the full traceback
            logger.warning(
                "-" * 40
                + "\n"
                + f"Failed to process {pdb_id}: {str(e)}\n"
                + f"Exception type: {type(e).__name__}\nTraceback: {tb}"
                + "-" * 40
            )
            return


def preprocess_pdb_monomer_distilation(
    output_dir: Path,
    dataset_cache: Path,
    s3_config: dict,
    num_workers: int = 1,
):
    """
    Args:
        structure_pred_dir (Path): _description_
        output_dir (Path): _description_
        dataset_cache (Path): _description_
        num_workers (int | None, optional): _description_. Defaults to None.
    """

    with open(dataset_cache) as f:
        dataset_cache = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_ids = list(dataset_cache["structure_data"].keys())

    wrapper = _WrapProcessMonomerDistillStructure(s3_config, output_dir)
    if num_workers > 1:
        with mp.Pool(
            num_workers, initializer=_init_worker, initargs=(s3_config["profile"],)
        ) as p:
            for _ in tqdm(p.imap_unordered(wrapper, pdb_ids), total=len(pdb_ids)):
                pass
    else:
        for pdb_id in tqdm(pdb_ids):
            wrapper(pdb_id)


# TODO: combine local and S3 monomer preparsing
def preparse_monomer(
    entry_id: str,
    data_directory: Path,
    structure_filename: str,
    structure_file_format: str,
    output_dir: Path,
):
    ### to reduce run times only parse if the file does not exist
    output_file = output_dir / f"{entry_id}/{entry_id}.pkl"
    if output_file.exists():
        return
    _, atom_array = parse_protein_monomer_pdb_tmp(
        data_directory / entry_id / f"{structure_filename}.{structure_file_format}"
    )
    write_structure(atom_array, output_dir / f"{entry_id}/{entry_id}.pkl")


class _ProteinMonomerPreprocessingWrapper:
    def __init__(
        self,
        data_directory: Path,
        structure_filename: str,
        structure_file_format: str,
        output_dir: Path,
    ) -> None:
        """Wrapper class for pre-parsing protein mononer files into .pkl."""
        self.data_directory = data_directory
        self.structure_filename = structure_filename
        self.structure_file_format = structure_file_format
        self.output_dir = output_dir

    def __call__(self, entry_id: str) -> None:
        try:
            preparse_monomer(
                entry_id,
                self.data_directory,
                self.structure_filename,
                self.structure_file_format,
                self.output_dir,
            )
        except Exception as e:
            print(f"Failed to preparse monomer {entry_id}:\n{e}\n")


def preparse_protein_monomer_structures(
    dataset_cache: ProteinMonomerDatasetCache,
    data_directory: Path,
    structure_filename: str,
    structure_file_format: str,
    output_dir: Path,
    num_workers: int,
    chunksize: int,
):
    # Create per-chain directories
    entry_ids = list(dataset_cache.structure_data.keys())
    output_dir = output_dir / "structure_files"
    output_dir.mkdir(parents=True, exist_ok=True)
    for entry_id in tqdm(
        entry_ids, total=len(entry_ids), desc="1/2: Creating output directories"
    ):
        entry_dir = output_dir / f"{entry_id}"
        if not entry_dir.exists():
            entry_dir.mkdir(parents=True, exist_ok=True)

    wrapped_monomer_preparser = _ProteinMonomerPreprocessingWrapper(
        data_directory, structure_filename, structure_file_format, output_dir
    )

    with mp.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                wrapped_monomer_preparser,
                entry_ids,
                chunksize=chunksize,
            ),
            total=len(entry_ids),
            desc="2/2: Pre-parsing monomer structures",
        ):
            pass


# ---- disordered PDB metadata cache pipelines ----
# TODO: improve docstrings
def find_parent_metadata_cache_subset(
    parent_metadata_cache: PreprocessingDataCache,
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    ost_aln_output_directory: Path,
    subset_file: Path | None,
    logger: logging.Logger,
) -> list[str]:
    """Finds the subset of PDB IDs that have all necessary data available.

    Args:
        parent_metadata_cache (PreprocessingDataCache): _description_
        gt_structures_directory (Path): _description_
        pred_structures_directory (Path): _description_
        ost_aln_output_directory (Path): _description_
        subset_file (Path | None): _description_
        logger (logging.Logger): _description_

    Returns:
        list[str]: _description_
    """

    logger.info(
        "Loaded parent metadata cache with "
        f"{len(parent_metadata_cache.structure_data)} entries."
    )

    logger.info("1/6: Finding shared PDB IDs with all necessary data available.")
    # - structures available
    pred_pdb_ids = [i.stem for i in list(pred_structures_directory.iterdir())]
    gt_pdb_ids = [i.stem for i in list(gt_structures_directory.iterdir())]

    shared_pdb_ids = sorted(
        set(pred_pdb_ids)
        & set(gt_pdb_ids)
        & set(parent_metadata_cache.structure_data.keys())
    )
    logger.info(
        f"{len(shared_pdb_ids)} metadata keys have both GT and predicted structures "
        f"in {gt_structures_directory} and {pred_structures_directory}, respectively."
    )

    # - alignment successful
    # TODO: set optional once OST is integrated into this script
    aln_pdb_ids = [i.stem for i in list(ost_aln_output_directory.iterdir())]

    shared_pdb_ids = sorted(set(shared_pdb_ids) & set(aln_pdb_ids))
    logger.info(
        f"Out of the above, {len(shared_pdb_ids)} have precomputed alignments available"
        f" in {ost_aln_output_directory}."
    )

    # - subset file
    if subset_file is not None:
        subset_pdb_ids = list(pd.read_csv(subset_file, header=None, sep="\t")[0])
        shared_pdb_ids = sorted(set(shared_pdb_ids) & set(subset_pdb_ids))
        logger.info(
            f"Out of the above, {len(shared_pdb_ids)} are in the subset "
            f"file {subset_file}."
        )
    else:
        logger.info(f"No subset file, using remaining {len(shared_pdb_ids)} entries.")

    return shared_pdb_ids


def create_provisional_disordered_structure_data_entry(
    pdb_id: str,
    structure_data_entry: PreprocessingStructureData,
    ost_aln_output_directory: Path,
) -> DisorderedPreprocessingStructureData:
    """Selects the model with the highest GDT to GT and creates its structure data field

    Args:
        pdb_id (str): _description_
        structure_data_entry (PreprocessingStructureData): _description_
        ost_aln_output_directory (Path): _description_

    Returns:
        DisorderedPreprocessingStructureData: _description_
    """
    # Parse the pred-GT results for all models
    ost_aln_output_directory_i = ost_aln_output_directory / pdb_id
    aln_filenames = [i.stem for i in list(ost_aln_output_directory_i.iterdir())]
    ost_aln_data = []
    gdts = np.zeros(len(aln_filenames))
    for idx, aln_filename in enumerate(aln_filenames):
        with open(ost_aln_output_directory / f"{pdb_id}/{aln_filename}.json") as f:
            ost_aln_data.append(json.load(f))
            gdts[idx] = float(ost_aln_data["oligo_gdtts"])

    # Find the one with the highest GDT
    best_idx = np.argmax(gdts)

    # Process the chain mapping
    chain_map_raw = ost_aln_data[best_idx]["chain_mapping"]
    chain_map = {}
    for gt_chain_id, model_chain_id in chain_map_raw.items():
        # TODO: make sure we don't need the numerical portions of the GT chain ID
        chain_map[gt_chain_id.split(".")[-1]] = model_chain_id

    # Add to the entry
    return DisorderedPreprocessingStructureData(
        status=structure_data_entry.status,
        release_date=structure_data_entry.release_date,
        experimental_method="N/A",
        resolution=structure_data_entry.resolution,
        chains=structure_data_entry.chains,
        interfaces=structure_data_entry.interfaces,
        token_count=structure_data_entry.token_count,
        gdt=gdts[best_idx],
        chain_map=chain_map,
        best_model_filename=aln_filenames[best_idx],
        distance_clash_map=None,
    )


class _DisorderedMetadataCacheBuilder:
    def __init__(self, output_directory: Path, ost_aln_output_directory: Path):
        self.output_directory = output_directory
        self.ost_aln_output_directory = ost_aln_output_directory

    @wraps(create_provisional_disordered_structure_data_entry)
    def __call__(
        self, input_data: tuple[str, PreprocessingStructureData]
    ) -> DisorderedPreprocessingStructureData:
        try:
            # Setup worker logger
            worker_logger = logging.getLogger(f"worker_{os.getpid()}")
            worker_logger.setLevel(logging.INFO)
            worker_logger.handlers = []
            worker_logger.propagate = False
            handler = logging.FileHandler(
                self.output_directory
                / f"worker_logs_provisional/worker_{os.getpid()}.log"
            )
            worker_logger.addHandler(handler)

            pdb_id, structure_data_entry = input_data

            worker_logger.info(f"Processing {pdb_id}.")

            provisional_entry = create_provisional_disordered_structure_data_entry(
                pdb_id, structure_data_entry, self.ost_aln_output_directory
            )
            return pdb_id, provisional_entry
        except Exception as e:
            worker_logger.info(f"Error processing {pdb_id}: {e}")

            worker_logger.error(
                f"Error processing {pdb_id}:"
                f"\n\nException:\n{str(e)}"
                f"\n\nType:\n{type(e).__name__}"
                f"\n\nTraceback:\n{traceback.format_exc()}"
            )

            failed_provisional_entry = DisorderedPreprocessingStructureData(
                status="failed",
                release_date=None,
                experimental_method=None,
                resolution=None,
                chains=None,
                interfaces=None,
                token_count=None,
                gdt=None,
                chain_map=None,
                best_model_filename=None,
                distance_clash_map=None,
            )

            return pdb_id, failed_provisional_entry


# TODO: add support for computing GDT with OST on the fly here
def build_provisional_disordered_metadata_cache(
    parent_metadata_cache: PreprocessingDataCache,
    pdb_id_list: list[str],
    ost_aln_output_directory: Path,
    output_directory: Path,
    num_workers: int,
    chunksize: int,
    log_file: Path,
) -> DisorderedPreprocessingDataCache:
    """
    Creates the disorder metadata cache from a parent metadata cache.

    Args:
        parent_metadata_cache (PreprocessingDataCache):
            The parent metadata cache from which to derive the disordered metadata
            cache.
        pdb_id_list (list[str]):
            A list of PDB IDs to subset the disordered metadata cache to.
        ost_aln_output_directory (Path):
            The directory where the OST structural alignment output files are stored.
        output_directory (Path):
            The output directory for the disordered metadata cache.
        num_workers (int):
            The number of workers to parallelize the structure data entry updates to.
        chunksize (int):
            The chunksize for the parallelization.
        logger (logging.Logger):
            The logger object.

    Returns (DisorderedPreprocessingDataCache):
        The disordered metadata cache with populated gdt and chain_map fields.
    """

    # Update the structure data
    structure_data = {}
    wrapped_builder = _DisorderedMetadataCacheBuilder(
        output_directory, ost_aln_output_directory
    )
    input_data = [
        (pdb_id, parent_metadata_cache.structure_data[pdb_id]) for pdb_id in pdb_id_list
    ]
    worker_log_directory = output_directory / "worker_logs_provisional"
    worker_log_directory.mkdir(exist_ok=True)

    with mp.Pool(num_workers) as pool:
        for pdb_id, provisional_entry in tqdm(
            pool.imap_unordered(
                wrapped_builder,
                input_data,
                chunksize=chunksize,
            ),
            total=len(pdb_id_list),
            desc="2/6: Building provisional disordered metadata cache",
        ):
            structure_data[pdb_id] = provisional_entry

    # Collate logs
    with log_file.open("a") as out_file:
        for worker_log in tqdm(
            worker_log_directory.iterdir(),
            desc="3/6: Collating provisional worker logs",
            total=len(list(worker_log_directory.iterdir())),
        ):
            out_file.write(f"Log file: {worker_log.name}\n")
            out_file.write(worker_log.read_text())
            worker_log.unlink()

        if not list(worker_log_directory.iterdir()):
            worker_log_directory.rmdir()

    return DisorderedPreprocessingDataCache(
        structure_data=structure_data,
        reference_molecule_data=parent_metadata_cache.reference_molecule_data,
    )


def preprocess_disordered_structure_and_write_outputs_af3(
    pdb_id: str,
    structure_data_entry: DisorderedPreprocessingStructureData,
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    gt_file_format: str,
    pred_file_format: str,
    output_directory: Path,
    ccd: CIFFile,
    pocket_distance_threshold: float,
    clash_distance_thresholds: list[float],
    transfer_annot_dict: dict[str, Any],
    delete_annot_list: list[str],
) -> dict[float, bool]:
    # Load GT and best GDT pred structures into AtomArrays and sanitize
    gt_atom_array = parse_target_structure(
        target_structures_directory=gt_structures_directory,
        pdb_id=pdb_id,
        structure_format=gt_file_format,
    )
    _, pred_atom_array = parse_pdb_af2(
        pred_structures_directory
        / f"{pdb_id}/{structure_data_entry.best_model_filename}.{pred_file_format}"
    )
    pred_atom_array = remove_hydrogens(pred_atom_array)
    pred_atom_array = remove_std_residue_terminal_atoms(pred_atom_array)

    # Sort atom order
    gt_atom_array = canonicalize_atom_order(gt_atom_array, ccd)
    pred_atom_array = canonicalize_atom_order(pred_atom_array, ccd)

    # Detect non-protein chains covalently linked to a protein chain and remove
    gt_atom_array = remove_covalent_nonprotein_chains(gt_atom_array)

    # Transfer annotations from GT protein to pred and remove unnecessary annotations
    # TODO: expose annot arguments
    gt_atom_array_protein = gt_atom_array.molecule_type_id == MoleculeType.PROTEIN
    pred_atom_array_annotated = remove_transfer_annotations(
        target_atom_array=pred_atom_array,
        source_atom_array=gt_atom_array_protein,
        chain_map=structure_data_entry.chain_map,
        transfer_annot_dict=transfer_annot_dict,
        delete_annot_list=delete_annot_list,
    )

    # Pocket align each GT non-protein chain to the predicted protein chains via
    # the GT protein chains
    gt_atom_array_nonprotein = gt_atom_array[
        gt_atom_array.molecule_type_id != MoleculeType.PROTEIN
    ]
    gt_atom_array_nonprotein_aligned = coalign_atom_arrays(
        fixed=pred_atom_array_annotated,
        mobile=gt_atom_array_protein,
        comobile=gt_atom_array_nonprotein,
        distance_threshold=pocket_distance_threshold,
        mobile_distance_atom_names=["N", "CA", "C"],
        alignment_mask_atom_names=["N", "CA", "C"],
    )
    chimeric_atom_array = struc.concatenate(
        [pred_atom_array_annotated, gt_atom_array_nonprotein_aligned]
    )

    # Calculate clashes
    distance_clash_map = calculate_distance_clash_map(
        query_atom_array=chimeric_atom_array[
            chimeric_atom_array.molecule_type_id == MoleculeType.PROTEIN
        ],
        target_atom_array=chimeric_atom_array[
            chimeric_atom_array.molecule_type_id != MoleculeType.PROTEIN
        ],
        distance_thresholds=clash_distance_thresholds,
    )

    # Save structure as cif and npz
    strucio.save_structure(
        output_directory / f"structures/{pdb_id}/{pdb_id}.cif",
        chimeric_atom_array,
    )
    write_atomarray_to_npz(
        chimeric_atom_array, output_directory / f"structures/{pdb_id}/{pdb_id}.npz"
    )

    # Return clash status
    return distance_clash_map


class _AF3PreprocessingDisorderedWrapper:
    def __init__(
        self,
        gt_structures_directory: Path,
        pred_structures_directory: Path,
        gt_file_format: str,
        pred_file_format: str,
        output_directory: Path,
        ccd: CIFFile,
        pocket_distance_threshold: float,
        clash_distance_thresholds: list[float],
        transfer_annot_dict: dict[str, Any],
        delete_annot_list: list[str],
    ):
        self.gt_structures_directory = gt_structures_directory
        self.pred_structures_directory = pred_structures_directory
        self.gt_file_format = gt_file_format
        self.pred_file_format = pred_file_format
        self.output_directory = output_directory
        self.ccd = ccd
        self.pocket_distance_threshold = pocket_distance_threshold
        self.clash_distance_thresholds = clash_distance_thresholds
        self.transfer_annot_dict = transfer_annot_dict
        self.delete_annot_list = delete_annot_list

    @wraps(preprocess_disordered_structure_and_write_outputs_af3)
    def __call__(
        self, input_data: tuple[str, DisorderedPreprocessingStructureData]
    ) -> tuple[str, dict[float, bool] | None]:
        try:
            pdb_id, structure_data_entry = input_data
            # Setup worker logger
            worker_logger = logging.getLogger(f"worker_{os.getpid()}")
            worker_logger.setLevel(logging.INFO)
            worker_logger.handlers = []
            worker_logger.propagate = False
            handler = logging.FileHandler(
                self.output_directory / f"worker_logs/worker_{os.getpid()}.log"
            )
            worker_logger.addHandler(handler)

            worker_logger.info(f"Processing {pdb_id}.")

            distance_clash_map = preprocess_disordered_structure_and_write_outputs_af3(
                pdb_id=pdb_id,
                structure_data_entry=structure_data_entry,
                gt_structures_directory=self.gt_structures_directory,
                pred_structures_directory=self.pred_structures_directory,
                gt_file_format=self.gt_file_format,
                pred_file_format=self.pred_file_format,
                output_directory=self.output_directory,
                ccd=self.ccd,
                pocket_distance_threshold=self.pocket_distance_threshold,
                clash_distance_thresholds=self.clash_distance_thresholds,
                transfer_annot_dict=self.transfer_annot_dict,
                delete_annot_list=self.delete_annot_list,
            )
            return pdb_id, distance_clash_map

        except Exception as e:
            worker_logger.error(
                f"Error processing {pdb_id}:"
                f"\n\nException:\n{str(e)}"
                f"\n\nType:\n{type(e).__name__}"
                f"\n\nTraceback:\n{traceback.format_exc()}"
            )
            return pdb_id, None


def preprocess_disordered_structures(
    metadata_cache: DisorderedPreprocessingDataCache,
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    gt_file_format: str,
    pred_file_format: str,
    output_directory: Path,
    ccd_file: Path,
    pocket_distance_threshold: float,
    clash_distance_thresholds: list[float],
    transfer_annot_dict: dict[str, Any],
    delete_annot_list: list[str],
    num_workers: int,
    chunksize: int,
    log_file: Path,
) -> DisorderedPreprocessingDataCache:
    """Preprocesses AF2 PDB predictions following AF3 SI 2.5.2.3.

    Args:
        metadata_cache (DisorderedPreprocessingDataCache): _description_
        gt_structures_directory (Path): _description_
        pred_structures_directory (Path): _description_
        gt_file_format (str): _description_
        pred_file_format (str): _description_
        output_directory (Path): _description_
        ccd_file (Path): _description_
        pocket_distance_threshold (float): _description_
        clash_distance_thresholds (list[float]): _description_
        transfer_annot_dict (dict[str, Any]): _description_
        delete_annot_list (list[str]): _description_
        num_workers (int): _description_
        chunksize (int): _description_
        log_file (Path): _description_

    Returns:
        DisorderedPreprocessingDataCache: _description_
    """
    # Get input data
    input_data = [
        (pdb_id, structure_data_entry)
        for pdb_id, structure_data_entry in metadata_cache.structure_data.items()
    ]
    ccd = CIFFile.read(ccd_file)
    worker_log_directory = output_directory / "worker_logs"
    worker_log_directory.mkdir(exist_ok=True)
    wrapper_disordered_structure_processor = _AF3PreprocessingDisorderedWrapper(
        gt_structures_directory=gt_structures_directory,
        pred_structures_directory=pred_structures_directory,
        gt_file_format=gt_file_format,
        pred_file_format=pred_file_format,
        output_directory=output_directory,
        ccd=ccd,
        pocket_distance_threshold=pocket_distance_threshold,
        clash_distance_thresholds=clash_distance_thresholds,
        transfer_annot_dict=transfer_annot_dict,
        delete_annot_list=delete_annot_list,
    )

    # Pre-create output directories
    structures_output_directory = output_directory / "structures/"
    structures_output_directory.mkdir(exist_ok=True)
    for pdb_id, _ in tqdm(
        input_data,
        total=len(input_data),
        desc="4/6: Creating structure data output directories",
    ):
        (structures_output_directory / pdb_id).mkdir(exist_ok=True)

    # Preprocess entries and update distance_clash_map
    with mp.Pool(num_workers) as pool:
        for pdb_id, distance_clash_map in tqdm(
            pool.imap_unordered(
                wrapper_disordered_structure_processor,
                input_data,
                chunksize=chunksize,
            ),
            total=len(input_data),
            desc="5/6: Processing disordered structures",
        ):
            metadata_cache.structure_data[
                pdb_id
            ].distance_clash_map = distance_clash_map

    # Collate logs
    with log_file.open("a") as out_file:
        for worker_log in tqdm(
            worker_log_directory.iterdir(),
            desc="6/6: Collating preprocessing worker logs",
            total=len(list(worker_log_directory.iterdir())),
        ):
            out_file.write(f"Log file: {worker_log.name}\n")
            out_file.write(worker_log.read_text())
            worker_log.unlink()

        if not list(worker_log_directory.iterdir()):
            worker_log_directory.rmdir()

    return metadata_cache


def preprocess_pdb_disordered_af3(
    metadata_cache_file: Path,
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    gt_file_format: str,
    pred_file_format: str,
    output_directory: Path,
    ost_aln_output_directory: Path,
    subset_file: Path | None,
    ccd_file: Path,
    pocket_distance_threshold: float,
    clash_distance_thresholds: list[float],
    transfer_annot_dict: dict[str, Any],
    delete_annot_list: list[str],
    num_workers: int,
    chunksize: int,
    log_file: Path,
):
    """The full pipeline for creating the disordered metadata cache."""

    # Configure the main process logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)  # Set the logging level for the file handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    # Find subset parent metadata cache
    parent_metadata_cache = PreprocessingDataCache.from_json(metadata_cache_file)
    shared_pdb_ids = find_parent_metadata_cache_subset(
        parent_metadata_cache=parent_metadata_cache,
        gt_structures_directory=gt_structures_directory,
        pred_structures_directory=pred_structures_directory,
        ost_aln_output_directory=ost_aln_output_directory,
        subset_file=subset_file,
        logger=logger,
    )

    # Build provisional metadata cache with GDT and chain_map
    provisional_metadata_cache = build_provisional_disordered_metadata_cache(
        parent_metadata_cache=parent_metadata_cache,
        pdb_id_list=shared_pdb_ids,
        ost_aln_output_directory=ost_aln_output_directory,
        output_directory=output_directory,
        num_workers=num_workers,
        chunksize=chunksize,
        log_file=log_file,
    )
    provisional_metadata_cache.to_json(
        output_directory / "provisional_metadata_cache.json"
    )

    # Preprocess disordered structures and update metadata cache entries
    metadata_cache = preprocess_disordered_structures(
        metadata_cache=provisional_metadata_cache,
        gt_structures_directory=gt_structures_directory,
        pred_structures_directory=pred_structures_directory,
        gt_file_format=gt_file_format,
        pred_file_format=pred_file_format,
        output_directory=output_directory,
        ccd_file=ccd_file,
        pocket_distance_threshold=pocket_distance_threshold,
        clash_distance_thresholds=clash_distance_thresholds,
        transfer_annot_dict=transfer_annot_dict,
        delete_annot_list=delete_annot_list,
        num_workers=num_workers,
        chunksize=chunksize,
        log_file=log_file,
    )
    metadata_cache.to_json(output_directory / "metadata_cache.json")

    logger.info("Done.")
