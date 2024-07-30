import json
import os
from pathlib import Path
from typing import NamedTuple

import click
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile
from tqdm import tqdm

import openfold3.core.data.preprocessing.io as io
from openfold3.core.data.preprocessing.metadata_extraction import (
    get_chain_to_author_chain_dict,
    get_chain_to_canonical_seq_dict,
    get_chain_to_entity_dict,
    get_chain_to_molecule_type_dict,
    get_chain_to_pdb_chain_dict,
    get_cif_block,
    get_pdb_id,
    get_release_date,
    get_resolution,
)
from openfold3.core.data.preprocessing.structure_preprocessing_pipelines import (
    cleanup_structure_af3,
)
from openfold3.core.data.preprocessing.structure_primitives import (
    assign_renumbered_chain_ids,
    get_interface_chain_id_pairs,
)

print("PID:", os.getpid())


# make this a dataclass?
class ProcessedStructure(NamedTuple):
    cif_file: CIFFile  # parsed CIF file information
    atom_array: AtomArray  # cleaned structure
    metadata_dict: dict  # structure-, chain-, and interface-level metadata
    chain_to_canonical_seq: dict  # mapping of chain IDs to canonical sequences


def process_structure(cif_path: Path, ccd: CIFFile) -> ProcessedStructure:
    cif_file, atom_array = io.parse_mmcif_bioassembly(
        cif_path, extra_fields=["auth_asym_id"]
    )

    # Get block with all the primary CIF data
    cif_data = get_cif_block(cif_file)

    # Apply full structure cleanup
    atom_array = cleanup_structure_af3(atom_array, cif_data, ccd)

    # Renumber chain IDs so that they're consistent with what you would get when parsing
    # the processed structure (which could have deleted chains in preprocessing)
    assign_renumbered_chain_ids(atom_array)

    # Get basic metadata
    metadata_dict = {
        "release_date": get_release_date(cif_data),
        "resolution": get_resolution(cif_data),
    }

    # Get chain-level metadata
    chain_to_pdb_chain = get_chain_to_pdb_chain_dict(atom_array)
    chain_to_author_chain = get_chain_to_author_chain_dict(atom_array)
    chain_to_entity = get_chain_to_entity_dict(atom_array)
    chain_to_molecule_type = get_chain_to_molecule_type_dict(atom_array)
    chain_to_canonical_seq = get_chain_to_canonical_seq_dict(atom_array, cif_data)

    # Create list of chains with metadata
    chain_metadata_list = []

    for chain in chain_to_pdb_chain:
        chain_metadata = {
            "molecule_type": chain_to_molecule_type[chain],
            "chain_id_renumbered": chain,
            "chain_id_pdb": chain_to_pdb_chain[chain],
            "chain_id_author": chain_to_author_chain[chain],
            "entity_id": chain_to_entity[chain],
        }
        chain_metadata_list.append(chain_metadata)

    # Create list of interfaces with metadata
    interface_chain_pairs = get_interface_chain_id_pairs(atom_array)
    interface_metadata_list = []

    for chain_1, chain_2 in interface_chain_pairs:
        interface_metadata = {
            "molecule_type": [
                chain_to_molecule_type[chain_1],
                chain_to_molecule_type[chain_2],
            ],
            "chain_id_renumbered": [chain_1, chain_2],
            "chain_id_pdb": [chain_to_pdb_chain[chain_1], chain_to_pdb_chain[chain_2]],
            "chain_id_author": [
                chain_to_author_chain[chain_1],
                chain_to_author_chain[chain_2],
            ],
            "entity_id": [chain_to_entity[chain_1], chain_to_entity[chain_2]],
        }
        interface_metadata_list.append(interface_metadata)

    # Add metadata lists to the metadata dictionary
    metadata_dict["chains"] = chain_metadata_list
    metadata_dict["interfaces"] = interface_metadata_list

    return ProcessedStructure(
        cif_file=cif_file,
        atom_array=atom_array,
        metadata_dict=metadata_dict,
        chain_to_canonical_seq=chain_to_canonical_seq,
    )


@click.command()
@click.option(
    "--mmcif_folder",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    help="Folder containing mmCIF files",
)
@click.option(
    "--ccd_path",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    help="Path to a Chemical Component Dictionary (CCD) .cif file",
)
@click.option(
    "--output_folder",
    type=click.Path(file_okay=False, dir_okay=True, exists=False, path_type=Path),
    help="Folder to save cleaned bCIF files and other extracted files in",
)
@click.option(
    "--include_cifs",
    type=bool,
    default=False,
    help="Whether to save additional .cif files alongside the .bcif files",
)
def main(
    mmcif_folder: Path, ccd_path: Path, output_folder: Path, include_cifs: bool = False
) -> None:
    ccd = CIFFile.read(ccd_path)

    metadata = {}

    # TODO: remove enumerate and break condition
    for i, cif_path in enumerate(tqdm(mmcif_folder.glob("*.cif"))):
        # if i < 21:
        #     continue

        if i == 30:
            break

        print(f"Processing {cif_path}")
        processed_structure = process_structure(cif_path, ccd)
        pdb_id = get_pdb_id(processed_structure.cif_file)

        output_subfolder = output_folder / pdb_id
        output_subfolder.mkdir(exist_ok=True, parents=True)

        metadata[pdb_id] = processed_structure.metadata_dict

        # Save cleaned bCIF file (this will be minimal and only include AtomSite
        # records)
        bcif_path = output_subfolder / f"{pdb_id}.bcif"
        io.write_minimal_cif(
            processed_structure.atom_array,
            bcif_path,
            format="bcif",
            include_bonds=True,
        )

        # Save additional .cif file if requested
        if include_cifs:
            cif_path = output_subfolder / f"{pdb_id}.cif"
            io.write_minimal_cif(
                processed_structure.atom_array,
                cif_path,
                format="cif",
                include_bonds=True,
            )

        # Save FASTA file
        fasta_path = io.write_annotated_chains_fasta(
            output_subfolder / f"{pdb_id}.fasta",
            processed_structure.chain_to_canonical_seq,
            pdb_id,
            processed_structure.metadata_dict["chains"],
        )

        print(f"Processed {pdb_id}: {bcif_path}, {fasta_path}")

    json_path = output_folder / "metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4, default=io.encode_numpy_types)


if __name__ == "__main__":
    main()
