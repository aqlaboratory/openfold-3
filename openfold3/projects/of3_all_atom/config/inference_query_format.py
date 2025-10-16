import json
from pathlib import Path
from typing import Annotated, Any, NamedTuple

from pydantic import (
    BaseModel,
    BeforeValidator,
    DirectoryPath,
    FilePath,
    field_serializer,
)

from openfold3.core.config.config_utils import (
    _cast_keys_to_int,
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
    model_config = {
        "use_enum_values": False,
        "extra": "forbid",
    }
    molecule_type: Annotated[MoleculeType, BeforeValidator(_convert_molecule_type)]
    chain_ids: Annotated[list[str], BeforeValidator(_ensure_list)]
    sequence: str | None = None
    non_canonical_residues: (
        Annotated[dict[int, str], BeforeValidator(_cast_keys_to_int)] | None
    ) = None
    smiles: str | None = None
    ccd_codes: Annotated[list[str], BeforeValidator(_ensure_list)] | None = None
    paired_msa_file_paths: (
        Annotated[list[FilePath | DirectoryPath], BeforeValidator(_ensure_list)] | None
    ) = None
    main_msa_file_paths: (
        Annotated[list[FilePath | DirectoryPath], BeforeValidator(_ensure_list)] | None
    ) = None
    template_alignment_file_path: FilePath | None = None
    template_entry_chain_ids: (
        Annotated[list[str], BeforeValidator(_ensure_list)] | None
    ) = None
    sdf_file_path: FilePath | None = None

    @field_serializer("molecule_type", return_type=str)
    def serialize_enum_name(self, v: MoleculeType, _info):
        return v.name

    # TODO(jennifer): Add validations to this class
    # - if molecule type is protein / dna / rna - must specify sequence
    # - if molecule type is ligand - either ccd or smiles needs to be specifified


class Query(BaseModel):
    query_name: str | None = None
    chains: list[Chain]
    use_msas: bool = True
    use_paired_msas: bool = True
    use_main_msas: bool = True
    covalent_bonds: list[Bond] | None = None


class InferenceQuerySet(BaseModel):
    seeds: list[int] = [42]
    queries: dict[str, Query]

    @classmethod
    def from_json(cls, json_path: FilePath) -> "InferenceQuerySet":
        """Load InferenceQuerySet from a JSON file."""
        with open(json_path) as f:
            data = f.read()
        return cls.model_validate_json(data)

    @classmethod
    def from_string(cls, json_str: str) -> "InferenceQuerySet":
        """Load InferenceQuerySet from a json string"""
        return cls.model_validate_json(json_str)

    def model_post_init(self, __context: Any) -> None:
        """Add query name to the query objects."""
        for name, query in self.queries.items():
            query.query_name = name


def remove_completed_queries_from_query_json(
    seeds: list[int] | int,
    num_diffusion_samples: int | None,
    query_json: Path,
    output_dir: Path,
    structure_format: str,
):
    """remove completed queries from the query json file

    Args:
        seed (int|List)
        num_diffusion_samples (int)
        query_json (Path)
        output_dir (Path)
    """

    if isinstance(seeds, int):
        seeds = [seeds]

    if not num_diffusion_samples:
        num_diffusion_samples = 5

    with open(query_json) as f:
        query_input_cfg = json.load(f)

    removed_structures = []
    for query_id in query_input_cfg["queries"]:
        ## a structure must be present for all seeds and all diffusion samples
        ## to count as completed
        structure_exists = True
        for seed in seeds:
            output_subdir = output_dir / query_id / f"seed_{seed}"
            for s in range(num_diffusion_samples):
                file_prefix = output_subdir / f"{query_id}_seed_{seed}_sample_{s + 1}"
                structure_file = Path(f"{file_prefix}_model.{structure_format}")
                structure_exists = structure_file.exists() and structure_exists

        if structure_exists:
            removed_structures.append(query_id)

    ## can't modify dict during loop
    for query_id in removed_structures:
        del query_input_cfg["queries"][query_id]

    ## handle case where all queries are completed
    if query_input_cfg["queries"]:
        print(
            "Skipping existing structures is enabled.Will skip "
            "the following {len(removed_structures)} structures:"
        )
        print(removed_structures)
        return json.dumps(query_input_cfg)
    else:
        print("All structures available. Quitting")
        return
