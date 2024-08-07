"""This module contains IO functions for reading and writing fasta files."""

from pathlib import Path
from typing import TypedDict


class ChainMetadataDict(TypedDict):
    molecule_type: str  # "protein", "RNA", "DNA", "ligand"
    chain_id_renumbered: int  # renumbered chain ID assigned by parse_mmcif_bioassembly
    chain_id_pdb: str  # PDB-assigned chain ID "label_asym_id"
    chain_id_author: str  # Author-assigned chain ID "auth_asym_id"


def write_annotated_chains_fasta(
    output_path: Path,
    chain_to_sequence: dict,
    pdb_id: str,
    chain_metadata_list: list[ChainMetadataDict],
) -> Path:
    """Writes a FASTA file annotating chain IDs and molecule types

    Will write all chains in the chain_metadata_list to a single FASTA file. The header
    will follow the format
    `>{pdb_id}_{chain_id_pdb}_{chain_id_author}_{chain_id_renum}_{molecule_type}`.

    Args:
        output_path:
            Path to write the FASTA file to.
        chain_to_sequence:
            Dictionary mapping chain IDs to sequences.
        pdb_id:
            PDB ID of the structure.
        chain_metadata_list:
            List of dictionaries containing chain metadata. Each dictionary should
            contain the following keys:
                - molecule_type: "protein", "RNA", "DNA", "ligand"
                - chain_id_renumbered: renumbered chain ID assigned by
                  parse_mmcif_bioassembly
                - chain_id_pdb: PDB-assigned chain ID "label_asym_id"
                - chain_id_author: Author-assigned chain ID "auth_asym_id"

    Returns:
        Path to the written FASTA file.
    """
    fasta_lines = []

    # Go through list of chains and extract sequences for all polymers
    for chain_data in chain_metadata_list:
        mol_type = chain_data["molecule_type"]

        # Skip ligands
        if mol_type == "ligand":
            continue

        chain_id_renum = chain_data["chain_id_renumbered"]
        chain_id_pdb = chain_data["chain_id_pdb"]
        chain_id_author = chain_data["chain_id_author"]

        # Header stores multiple chain IDs and molecule type
        chain_header = (
            f">{pdb_id}_{chain_id_pdb}_{chain_id_author}_{chain_id_renum}_"
            f"{mol_type}\n"
        )
        seq = chain_to_sequence[chain_id_renum] + "\n"

        fasta_lines.extend([chain_header, seq])

    with open(output_path, "w") as f:
        f.writelines(fasta_lines)

    return output_path
