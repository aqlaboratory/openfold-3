import json
import os
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import biotite.structure as struc
import click
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile
from pdbeccdutils.core import ccd_reader
from rdkit import Chem
from tqdm import tqdm

import openfold3.core.data.io.sequence.fasta
import openfold3.core.data.io.structure.cif
import openfold3.core.data.io.utils as utils
from openfold3.core.data.pipelines.preprocessing.structure import (
    cleanup_structure_af3,
)
from openfold3.core.data.primitives.structure.interface import (
    get_interface_chain_id_pairs,
)
from openfold3.core.data.primitives.structure.labels import (
    assign_renumbered_chain_ids,
    get_chain_to_author_chain_dict,
    get_chain_to_entity_dict,
    get_chain_to_molecule_type_dict,
    get_chain_to_pdb_chain_dict,
)
from openfold3.core.data.primitives.structure.ligand import (
    mol_from_atomarray,
    mol_from_parsed_component,
)
from openfold3.core.data.primitives.structure.metadata import (
    get_chain_to_canonical_seq_dict,
    get_cif_block,
    get_pdb_id,
    get_release_date,
    get_resolution,
)
from openfold3.core.data.resources.tables import MoleculeType

print("PID:", os.getpid())

ChainID = str
EntityID = int
ChemCompID = str


# TODO: move explanation to docstring
class ProcessedStructure(NamedTuple):
    cif_file: CIFFile  # parsed CIF file information
    atom_array: AtomArray  # cleaned structure
    metadata_dict: dict  # structure-, chain-, and interface-level metadata
    chain_to_canonical_seq: dict[
        ChainID, str
    ]  # mapping of chain IDs to canonical sequences
    ligand_ccd_components_to_chains: dict[ChemCompID, list[ChainID]]
    other_ccd_components: list[
        ChemCompID
    ]  # 3-letter codes of residue names that are in the CCD
    special_ligand_entities: list[EntityID]


def process_structure(cif_path: Path, ccd: CIFFile) -> ProcessedStructure:
    cif_file, atom_array = openfold3.core.data.io.structure.cif.parse_mmcif(
        cif_path, expand_bioassembly=True, extra_fields=["auth_asym_id"]
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

    # TODO: remove this and keep all metadata just in chains
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

    # Find ligands
    ligand_filter = atom_array.molecule_type_id == MoleculeType.LIGAND
    ligand_atom_array = atom_array[ligand_filter]
    ligand_ccd_components_to_chains = defaultdict(list)
    special_ligand_entities = set()

    if ligand_atom_array.array_length() != 0:
        lig_chain_starts = struc.get_chain_starts(
            ligand_atom_array, add_exclusive_stop=True
        )
        for chain_start, chain_end in zip(lig_chain_starts[:-1], lig_chain_starts[1:]):
            ligand_chain = ligand_atom_array[chain_start:chain_end]

            if struc.get_residue_count(ligand_chain) == 1:
                ccd_id = ligand_chain.res_name[0]
                chain_id = ligand_chain.chain_id_renumbered[0]
                ligand_ccd_components_to_chains[ccd_id].append(chain_id)
            else:
                special_ligand_entities.add(ligand_chain.entity_id[0])

    # Get other components
    other_ccd_components = set()
    for resname in np.unique(atom_array[~ligand_filter].res_name):
        other_ccd_components.add(resname)

    return ProcessedStructure(
        cif_file=cif_file,
        atom_array=atom_array,
        metadata_dict=metadata_dict,
        chain_to_canonical_seq=chain_to_canonical_seq,
        ligand_ccd_components_to_chains=ligand_ccd_components_to_chains,
        other_ccd_components=other_ccd_components,
        special_ligand_entities=special_ligand_entities,
    )


# TODO: remove
class CCDParsedComponentsFakeDict:
    """Temporary dict for faster debugging."""

    def __init__(self):
        self.components = {}
        self.ideal_path = Path(
            "/global/cfs/cdirs/m4351/ljarosch/of3_data/pdb_data/ideal_ligand_files/cif_files"
        )

    def __getitem__(self, key):
        if key in self.components:
            return self.components[key]
        else:
            ccd_result = ccd_reader.read_pdb_cif_file(
                str(self.ideal_path / f"{key}.cif")
            )
            component = ccd_result.component
            self.components[key] = component
            return component


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
    # Set output paths
    output_folder.mkdir(exist_ok=True, parents=True)
    metadata_json_path = output_folder / "metadata.json"
    ccd_components_path = output_folder / "ccd_component_sdfs"
    ccd_components_path.mkdir(exist_ok=True, parents=True)

    # Biotite-parsed CCD for regular metadata access
    ccd = CIFFile.read(ccd_path)

    # Preprocessed Component objects for every CCD component returned by pdbeccdutils
    # parsed_ccd_components = {
    #     ccd_id: result.component
    #     for ccd_id, result in ccd_reader.read_pdb_components_file(
    #         str(ccd_path)
    #     ).items()
    # }
    parsed_ccd_components = CCDParsedComponentsFakeDict()

    print("Parsed components.")

    all_metadata = {}

    saved_ccd_components = set()
    ccd_component_to_canonical_smiles = {}

    # TODO: remove enumerate and break condition
    for i, cif_path in enumerate(tqdm(mmcif_folder.glob("*.cif"))):
        if i == 30:
            break

        print(f"Processing {cif_path}")
        processed_structure = process_structure(cif_path, ccd)
        pdb_id = get_pdb_id(processed_structure.cif_file)
        structure_metadata = processed_structure.metadata_dict
        atom_array = processed_structure.atom_array

        output_subfolder = output_folder / pdb_id
        output_subfolder.mkdir(exist_ok=True, parents=True)
        special_component_sdfs_path = output_subfolder / "special_ligand_sdfs"
        special_component_sdfs_path.mkdir(exist_ok=True)

        ccd_id_to_chains = processed_structure.ligand_ccd_components_to_chains

        for ccd_id, chains in ccd_id_to_chains.items():
            if ccd_id in saved_ccd_components:
                canonical_smiles = ccd_component_to_canonical_smiles[ccd_id]
            else:
                mol = mol_from_parsed_component(
                    parsed_ccd_components[ccd_id], assign_fallback_conformer=True
                )
                canonical_smiles = mol.GetProp("canonical_smiles")
                ccd_component_to_canonical_smiles[ccd_id] = canonical_smiles

                with Chem.SDWriter(ccd_components_path / f"{ccd_id}.sdf") as writer:
                    writer.write(mol)

            for chain_metadata in structure_metadata["chains"]:
                if chain_metadata["chain_id_renumbered"] in chains:
                    chain_metadata["canonical_smiles"] = canonical_smiles

        for ccd_id in processed_structure.other_ccd_components:
            if ccd_id in saved_ccd_components:
                continue

            mol = mol_from_parsed_component(
                parsed_ccd_components[ccd_id], assign_fallback_conformer=True
            )

            with Chem.SDWriter(ccd_components_path / f"{ccd_id}.sdf") as writer:
                writer.write(mol)

        for entity_id in processed_structure.special_ligand_entities:
            entity_atom_array = atom_array[atom_array.entity_id == entity_id]
            mol = mol_from_atomarray(entity_atom_array, assign_fallback_conformer=True)

            mol_sdf_path = special_component_sdfs_path / f"{entity_id}.sdf"
            with Chem.SDWriter(mol_sdf_path) as writer:
                writer.write(mol)

            for chain_metadata in structure_metadata["chains"]:
                if chain_metadata["entity_id"] == entity_id:
                    chain_metadata["canonical_smiles"] = mol.GetProp("canonical_smiles")

        # Set final metadata for structure
        all_metadata[pdb_id] = structure_metadata

        # Save cleaned bCIF file (this will be minimal and only include AtomSite
        # records)
        bcif_path = output_subfolder / f"{pdb_id}.bcif"
        openfold3.core.data.io.structure.cif.write_minimal_cif(
            processed_structure.atom_array,
            bcif_path,
            format="bcif",
            include_bonds=True,
        )

        # Save additional .cif file if requested
        if include_cifs:
            cif_path = output_subfolder / f"{pdb_id}.cif"
            openfold3.core.data.io.structure.cif.write_minimal_cif(
                processed_structure.atom_array,
                cif_path,
                format="cif",
                include_bonds=True,
            )

        # Save FASTA file
        fasta_path = openfold3.core.data.io.sequence.fasta.write_annotated_chains_fasta(
            output_subfolder / f"{pdb_id}.fasta",
            processed_structure.chain_to_canonical_seq,
            pdb_id,
            processed_structure.metadata_dict["chains"],
        )

        print(f"Processed {pdb_id}: {bcif_path}, {fasta_path}")

    with open(metadata_json_path, "w") as f:
        json.dump(all_metadata, f, indent=4, default=utils.encode_numpy_types)

    # TODO: handle model conformer save date etc. (can't just handle dates on cache
    # level because of reference conformers -____-)


if __name__ == "__main__":
    main()
