"""
Centralized module for pre-assembled workflows corresponding to structure cleanup
procedures of different models.
"""

import json
import logging
import multiprocessing as mp
import traceback
from functools import wraps
from pathlib import Path
from typing import Literal

from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFBlock, CIFFile
from rdkit import Chem
from tqdm import tqdm

from openfold3.core.data.io.sequence.fasta import write_multichain_fasta
from openfold3.core.data.io.structure.cif import (
    SkippedStructure,
    parse_mmcif,
    write_minimal_cif,
)
from openfold3.core.data.io.structure.mol import write_annotated_sdf
from openfold3.core.data.io.utils import encode_numpy_types
from openfold3.core.data.pipelines.preprocessing.utils import SharedSet
from openfold3.core.data.primitives.structure.cleanup import (
    convert_MSE_to_MET,
    fix_arginine_naming,
    remove_chains_with_CA_gaps,
    remove_clashing_chains,
    remove_crystallization_aids,
    remove_fully_unknown_polymers,
    remove_hydrogens,
    remove_non_CCD_atoms,
    remove_small_polymers,
    remove_std_residue_terminal_atoms,
    remove_waters,
)
from openfold3.core.data.primitives.structure.component import (
    AnnotatedMol,
    get_components,
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
)
from openfold3.core.data.primitives.structure.metadata import (
    get_chain_to_canonical_seq_dict,
    get_cif_block,
    get_experimental_method,
    get_pdb_id,
    get_release_date,
    get_resolution,
)
from openfold3.core.data.primitives.structure.unresolved import add_unresolved_atoms

logger = logging.getLogger(__name__)


def cleanup_structure_af3(
    atom_array: AtomArray, cif_data: CIFBlock, ccd: CIFFile
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

    Returns:
        AtomArray with all cleaning steps applied
    """
    atom_array = atom_array.copy()

    ## Structure cleanup
    convert_MSE_to_MET(atom_array)
    fix_arginine_naming(atom_array)
    atom_array = remove_waters(atom_array)

    if get_experimental_method(cif_data) == "X-RAY DIFFRACTION":
        atom_array = remove_crystallization_aids(atom_array)

    atom_array = remove_hydrogens(atom_array)
    atom_array = remove_small_polymers(atom_array, cif_data, max_residues=3)
    atom_array = remove_fully_unknown_polymers(atom_array)
    atom_array = remove_clashing_chains(
        atom_array, clash_distance=1.7, clash_percentage=0.3
    )
    atom_array = remove_non_CCD_atoms(atom_array, ccd)
    atom_array = remove_chains_with_CA_gaps(atom_array, distance_threshold=10.0)

    ## Structure formatting
    # Add unresolved atoms explicitly with NaN coordinates
    atom_array = add_unresolved_atoms(atom_array, cif_data, ccd)

    # Remove terminal atoms to ensure consistent atom count for standard tokens
    atom_array = remove_std_residue_terminal_atoms(atom_array)

    return atom_array


# TODO: extend docstring
def extract_chain_and_interface_metadata_af3(
    atom_array: AtomArray, cif_data: CIFBlock
) -> dict:
    """Extracts chain and interface metadata from a structure.

    This extracts all individual chains and interfaces from a structure and annotates
    the chains with their different IDs in the structure, as well as the molecule type
    """

    metadata_dict = {}

    # Get basic metadata
    metadata_dict["release_date"] = get_release_date(cif_data).strftime("%Y-%m-%d")
    metadata_dict["resolution"] = get_resolution(cif_data)

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
    for chain_id in all_chains:
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
    skip_components: set | None = None,
) -> tuple[dict, dict]:
    """Extracts component data from a structure."""

    def get_reference_molecule_metadata(
        mol: AnnotatedMol,
        conformer_strategy: Literal["default", "random_init", "use_fallback"],
    ) -> dict:
        """Convenience function to return the metadata for a reference molecule."""
        conf_metadata = {
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
    residue_components, std_ligands_to_chains, non_std_ligands_to_chains = (
        get_components(atom_array)
    )

    # Assign IDs to non-standard components based on the PDB ID and entity ID
    non_std_ligands_to_chains = {
        f"{pdb_id}_{entity}": chains
        for entity, chains in non_std_ligands_to_chains.items()
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
        mol, conformer_strategy = resolve_and_format_fallback_conformer(mol)
        reference_mol_metadata[mol_id] = get_reference_molecule_metadata(
            mol, conformer_strategy
        )

        # Write SDF file
        sdf_out_path = sdf_out_dir / f"{mol_id}.sdf"
        write_annotated_sdf(mol, sdf_out_path)

    return chain_to_component_id, reference_mol_metadata


# TODO: write docstring and more comments
def preprocess_structure_and_write_outputs_af3(
    input_cif: Path,
    ccd: CIFFile,
    out_dir: Path,
    reference_mol_out_dir: Path,
    max_polymer_chains: int | None = None,
    skip_components: set | None = None,
    write_additional_cifs: bool = False,
) -> tuple[dict, dict]:
    parsed_mmcif = parse_mmcif(
        input_cif, expand_bioassembly=True, max_polymer_chains=max_polymer_chains
    )

    cif_file = parsed_mmcif.cif_file

    cif_data = get_cif_block(cif_file)
    pdb_id = get_pdb_id(cif_file)
    release_date = get_release_date(cif_data).strftime("%Y-%m-%d")

    if isinstance(parsed_mmcif, SkippedStructure):
        logger.info(
            f"Skipping structure with more than {max_polymer_chains} polymer chains."
        )
        n_polymer_chains = parsed_mmcif.n_polymer_chains

        return {
            pdb_id: {
                "release_date": release_date,
                "status": f"skipped: (n_chains: {n_polymer_chains})",
            }
        }, {}
    else:
        atom_array = parsed_mmcif.atom_array

    atom_array = cleanup_structure_af3(atom_array, cif_data, ccd)
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

    # Add chain to ligand ID mapping to metadata
    for chain_id, ligand_id in chain_to_ligand_ids.items():
        chain_int_metadata_dict["chains"][chain_id]["reference_mol_id"] = ligand_id

    structure_metadata_dict = {
        pdb_id: {
            "release_date": release_date,
            "status": "success",
            **chain_int_metadata_dict,
        }
    }

    chain_to_canonical_seq = get_chain_to_canonical_seq_dict(atom_array, cif_data)

    # Write CIF and FASTA outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cif_path = out_dir / f"{pdb_id}.bcif"

    write_minimal_cif(atom_array, out_cif_path, format="bcif", data_block=pdb_id)
    if write_additional_cifs:
        write_minimal_cif(
            atom_array, out_dir / f"{pdb_id}.cif", format="cif", data_block=pdb_id
        )

    out_fasta_path = out_dir / f"{pdb_id}.fasta"
    write_multichain_fasta(out_fasta_path, chain_to_canonical_seq)

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
        write_additional_cifs:
            Boolean flag to write additional CIFs.
    """

    def __init__(
        self,
        ccd,
        reference_mol_out_dir,
        max_polymer_chains,
        skip_components,
        write_additional_cifs=False,
    ):
        self.ccd = ccd
        self.reference_mol_out_dir = reference_mol_out_dir
        self.max_polymer_chains = max_polymer_chains
        self.skip_components = skip_components
        self.write_additional_cifs = write_additional_cifs

    @wraps(preprocess_structure_and_write_outputs_af3)
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
                    write_additional_cifs=self.write_additional_cifs,
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
            return {"pdb_id": cif_file.stem, "status": "failed"}, {}


def preprocess_cif_dir_af3(
    cif_dir: Path,
    ccd_path: Path,
    out_dir: Path,
    max_polymer_chains: int | None = None,
    num_workers: int | None = None,
    chunksize: int = 20,
    write_additional_cifs: bool = False,
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
        write_additional_cifs:
            Whether to additionally write normal .cif files on top of the binary .bcif
            files
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

    reference_mol_out_dir = out_dir / "reference_mols"
    reference_mol_out_dir.mkdir(parents=True, exist_ok=True)

    cif_out_dir = out_dir / "cif_files"
    cif_out_dir.mkdir(parents=True, exist_ok=True)

    cif_output_dirs = []

    # Pre-resolve CIF-output dirs
    logger.debug("Finding cif output directories")
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
        write_additional_cifs=write_additional_cifs,
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
