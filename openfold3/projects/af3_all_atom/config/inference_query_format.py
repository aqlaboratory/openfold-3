from typing import Annotated, NamedTuple, Optional

from pydantic import (
    BaseModel,
    BeforeValidator,
    DirectoryPath,
    FilePath,
    field_serializer,
)

from openfold3.core.config.config_utils import (
    FilePathOrNone,
    _convert_molecule_type,
    _ensure_list,
)
from openfold3.core.data.resources.residues import MoleculeType


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
    paired_msa_file_paths: Optional[
        Annotated[list[FilePath | DirectoryPath], BeforeValidator(_ensure_list)]
    ] = None
    main_msa_file_paths: Optional[list[FilePath | DirectoryPath]] = None
    # # Template definition
    # templates: ...
    sdf_file_path: Optional[FilePath] = None

    @field_serializer("molecule_type")
    def serialize_enum_name(self, v: MoleculeType, _info):
        return v.name

    # TODO(jennifer): Add validations to this class
    # - if molecule type is protein / dna / rna - must specify sequence
    # - if molecule type is ligand - either ccd or smiles needs to be specifified


class Query(BaseModel):
    chains: list[Chain]
    use_msas: bool = True
    use_paired_msas: bool = True
    use_main_msas: bool = True
    # use_templates: bool = False
    covalent_bonds: Optional[list[Bond]] = None


class InferenceQuerySet(BaseModel):
    seeds: list[int] = [42]
    queries: dict[str, Query]
    ccd_file_path: FilePathOrNone = None
    # msa_directory_path: DirectoryPathOrNone = None  # not yet supported

    @classmethod
    def from_json(cls, json_path: FilePath) -> "InferenceQuerySet":
        """Load InferenceQuerySet from a JSON file."""
        with open(json_path) as f:
            data = f.read()
        return cls.model_validate_json(data)
