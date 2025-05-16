import logging
from typing import Annotated, Any, NamedTuple, Optional

from pydantic import BaseModel, BeforeValidator, FilePath

from openfold3.core.config.path_definitions import (
    DirectoryPathOrNone,
    FilePathOrNone,
)
from openfold3.core.data.resources.residues import MoleculeType


def _ensure_list(value: Any) -> Any:
    if not isinstance(value, list):
        logging.info("Single value: {value} will be converted to a list")
        return [value]
    else:
        return value


def _convert_molecule_type(value: Any) -> Any:
    if isinstance(value, MoleculeType):
        return value
    elif isinstance(value, str):
        try:
            return MoleculeType[value.upper()]
        except KeyError:
            logging.warning(
                f"Found invalid {value=} for molecule type, skipping this example."
            )
            return None


# Definition for Bonds
class Atom(NamedTuple):
    chain_id: str
    residue_id: int
    atom_id: int


class Bond(NamedTuple):
    atom1: Atom
    atom2: Atom


class Chain(BaseModel):
    molecule_type: Annotated[MoleculeType, BeforeValidator(_convert_molecule_type)]
    chain_ids: Annotated[list[str], BeforeValidator(_ensure_list)]
    sequence: Optional[str] = None
    smiles: Optional[str] = None
    ccd_codes: Optional[Annotated[list[str], BeforeValidator(_ensure_list)]] = None
    # Msa definition
    use_msas: bool = True
    unpaired_msa_file_paths: Optional[list[FilePath]] = None
    paired_msa_file_path: Optional[
        Annotated[list[FilePath], BeforeValidator(_ensure_list)]
    ] = None
    # # Template definition
    # use_templates: bool = False
    # templates: ...
    sdf_file_path: Optional[FilePath] = None

    # TODO(jennifer): Add validations to this class
    # - if molecule type is protien / dna / rna - must specify sequence
    # - if molecule type is ligand - either ccd or smiles needs to be specifified


class Query(BaseModel):
    chains: list[Chain]
    covalent_bonds: Optional[list[Bond]] = None


class InferenceQuerySet(BaseModel):
    seeds: list[int] = [42]
    queries: dict[str, Query]
    ccd_file_path: FilePathOrNone = None
    msa_directory_path: DirectoryPathOrNone = None
