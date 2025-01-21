"""
Centralized module for pre-assembled workflows corresponding to structure cleanup
procedures of different models.
"""

# TODO: organize this file so that we separate components for creating the metadata
# cache for each dataset
import json
import logging
import multiprocessing as mp
import traceback
from functools import wraps
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import boto3
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFBlock, CIFFile
from rdkit import Chem
from tqdm import tqdm

from openfold3.core.data.io.s3 import download_file_from_s3
from openfold3.core.data.io.sequence.fasta import write_multichain_fasta
from openfold3.core.data.io.structure.cif import (
    SkippedStructure,
    parse_mmcif,
    write_structure,
)
from openfold3.core.data.io.structure.mol import write_annotated_sdf
from openfold3.core.data.io.structure.pdb import parse_protein_monomer_pdb_tmp
from openfold3.core.data.io.utils import encode_numpy_types
from openfold3.core.data.pipelines.preprocessing.utils import SharedSet
from openfold3.core.data.primitives.caches.format import ProteinMonomerDatasetCache
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
    subset_large_structure,
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
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array
from openfold3.core.data.primitives.structure.unresolved import add_unresolved_atoms

logger = logging.getLogger(__name__)

_worker_session = None


def _init_worker(profile_name: str = "openfold") -> None:
    """Initialize the boto3 session in each worker."""
    global _worker_session
    _worker_session = boto3.Session(profile_name=profile_name)


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
    fix_arginine_naming(atom_array)  #
    atom_array = remove_waters(atom_array)

    if get_experimental_method(cif_data) == "X-RAY DIFFRACTION":
        atom_array = remove_crystallization_aids(atom_array)

    atom_array = remove_hydrogens(atom_array)  #
    atom_array = remove_small_polymers(atom_array, cif_data, max_residues=3)
    atom_array = remove_fully_unknown_polymers(atom_array)
    atom_array = remove_clashing_chains(
        atom_array, clash_distance=1.7, clash_percentage=0.3
    )
    atom_array = remove_non_CCD_atoms(atom_array, ccd)
    atom_array = remove_chains_with_CA_gaps(atom_array, distance_threshold=10.0)

    # Subset bioassemblies larger than 20 chains
    if len(np.unique(atom_array.chain_id)) > 20:
        # Tokenization is required for large-structure subsetting
        tokenize_atom_array(atom_array)
        atom_array = subset_large_structure(
            atom_array=atom_array, n_chains=20, interface_distance_threshold=15.0
        )

    ## Structure formatting
    # Add unresolved atoms explicitly with NaN coordinates
    atom_array = add_unresolved_atoms(atom_array, cif_data, ccd)

    # Remove terminal atoms to ensure consistent atom count for standard tokens in the
    # model
    atom_array = remove_std_residue_terminal_atoms(atom_array)  #

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


def preprocess_structure_and_write_outputs_af3(
    input_cif: Path,
    ccd: CIFFile,
    out_dir: Path,
    reference_mol_out_dir: Path,
    output_formats: list[Literal["cif", "bcif", "pkl"]],
    max_polymer_chains: int | None = None,
    skip_components: set | SharedSet | None = None,
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
        max_polymer_chains:
            The maximum number of polymer chains in the first bioassembly after which a
            structure is skipped by the parser.
        skip_components:
            A set of components to skip, if any. Useful to avoid repeated processing of
            components e.g. by using a SharedSet.
        write_additional_cifs:
            Whether to additionally write normal .cif files on top of the binary .bcif
            files, which can be helpful for manual inspection.

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
    # Parse the input CIF file
    parsed_mmcif = parse_mmcif(
        input_cif,
        expand_bioassembly=True,
        include_bonds=True,
        renumber_chain_ids=True,
        max_polymer_chains=max_polymer_chains,
    )
    cif_file = parsed_mmcif.cif_file

    # Basic structure-level metadata
    cif_data = get_cif_block(cif_file)
    pdb_id = get_pdb_id(cif_file)
    release_date = get_release_date(cif_data).strftime("%Y-%m-%d")

    # Handle structures that got skipped due to max_polymer_chains
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

    # Cleanup structure and extract metadata
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

    # Get canonicalized sequence for each chain (should match PDB SeqRes)
    chain_to_canonical_seq = get_chain_to_canonical_seq_dict(atom_array, cif_data)

    # Write CIF and FASTA outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    for output_format in output_formats:
        out_path = out_dir / f"{pdb_id}.{output_format}"
        write_structure(atom_array, out_path, data_block=pdb_id)

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
        output_formats:
            What formats to write the output files to. Allowed values are "cif", "bcif",
            and "pkl".
    """

    def __init__(
        self,
        ccd: CIFFile,
        reference_mol_out_dir: Path,
        max_polymer_chains: int | None,
        skip_components: set | SharedSet | None,
        output_formats: list[Literal["cif", "bcif", "pkl"]],
    ):
        self.ccd = ccd
        self.reference_mol_out_dir = reference_mol_out_dir
        self.max_polymer_chains = max_polymer_chains
        self.skip_components = skip_components
        self.output_formats = output_formats

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
                    output_formats=self.output_formats,
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
    output_formats: list[Literal["cif", "bcif", "pkl"]] = False,
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
            What formats to write the output files to. Allowed values are "cif", "bcif",
            and "pkl".
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

    @wraps(preparse_monomer)
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


def preprocess_pdb_disordered_af3(
    metadata_cache_file: Path,
    gt_structures_directory: Path,
    pred_structures_directory: Path,
    pred_file_name: str,
    output_directory: Path,
    ost_aln_output_directory: Path,
    subset_file: Path | None = None,
):
    pass
    # with open(metadata_cache_file) as f:
    #     metadata_cache = json.load(f)

    # output_dir.mkdir(parents=True, exist_ok=True)
    # pdb_ids = list(metadata_cache["structure_data"].keys())

    # for pdb_id in tqdm(pdb_ids):
    #     gt_structure = gt_structures_directory / f"{pdb_id}.cif"
    #     pred_structure = pred_structures_directory / pdb_id / pred_file_name
    #     output_file = output_dir / f"{pdb_id}.pkl"

    #     if subset_file is not None:
    #         with open(subset_file) as f:
    #             subset = json.load(f)
    #         if pdb_id not in subset:
    #             continue

    #     if not gt_structure.exists():
    #         logger.warning(f"Missing ground truth structure for {pdb_id}")
    #         continue

    #     if not pred_structure.exists():
    #         logger.warning(f"Missing predicted structure for {pdb_id}")
    #         continue

    #     try:
    #         _, gt_atom_array = parse_protein_monomer_pdb_tmp(gt_structure)
    #         _, pred_atom_array = parse_protein_monomer_pdb_tmp(pred_structure)

    #         write_structure(gt_atom_array, output_file, data_block="ground_truth")
    #         write_structure(pred_atom_array, output_file, data_block="predicted")
    #     except Exception as e:
    #         logger.warning(f"Failed to process {pdb_id}: {str(e)}")
    #         continue
