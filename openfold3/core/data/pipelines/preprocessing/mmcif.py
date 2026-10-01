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

"""Preflight validation for mmCIF structure preprocessing inputs."""

from pathlib import Path

from biotite.structure.io.pdbx import CIFFile

# Minimum structural schema expected by the OF3 preprocessing workflow. This includes
# direct parser inputs, metadata read later in the pipeline, and categories that link
# polymer entities to chains. Optional metadata (for example resolution and bioassembly
# records) is intentionally omitted.
REQUIRED_MMCIF_DATA_ITEMS: dict[str, tuple[str, ...]] = {
    "atom_site": (
        "group_PDB",
        "type_symbol",
        "label_atom_id",
        "label_comp_id",
        "label_asym_id",
        "label_entity_id",
        "label_seq_id",
        "Cartn_x",
        "Cartn_y",
        "Cartn_z",
        "pdbx_PDB_model_num",
    ),
    "chem_comp": ("id", "type"),
    "entity_poly": ("entity_id", "pdbx_seq_one_letter_code_can"),
    "entity_poly_seq": ("entity_id", "num", "mon_id"),
    "exptl": ("method",),
    "pdbx_audit_revision_history": ("revision_date",),
}

REQUIRED_BIOASSEMBLY_DATA_ITEMS: dict[str, tuple[str, ...]] = {
    "pdbx_struct_assembly_gen": (
        "assembly_id",
        "oper_expression",
        "asym_id_list",
    ),
    "pdbx_struct_oper_list": (
        "id",
        "matrix[1][1]",
        "matrix[1][2]",
        "matrix[1][3]",
        "vector[1]",
        "matrix[2][1]",
        "matrix[2][2]",
        "matrix[2][3]",
        "vector[2]",
        "matrix[3][1]",
        "matrix[3][2]",
        "matrix[3][3]",
        "vector[3]",
    ),
}


class MMCIFPreflightError(ValueError):
    """Raised when an mmCIF lacks data required by structure preprocessing."""


def validate_mmcif_for_preprocessing(
    cif_file: CIFFile,
    source: Path | str | None = None,
    *,
    expand_bioassembly: bool = True,
    max_polymer_chains: int | None = None,
) -> None:
    """Validate the minimum mmCIF schema required by OF3 preprocessing.

    All missing categories and data items are collected before an exception is raised,
    so users can repair a custom mmCIF in one pass.

    Args:
        cif_file:
            Parsed mmCIF file to validate.
        source:
            Optional source path used to make the error message more actionable.
        expand_bioassembly:
            Whether to validate biological-assembly data when it is present.
        max_polymer_chains:
            If set, require the assembly count used by the early chain-count filter.

    Raises:
        MMCIFPreflightError:
            If the file does not contain exactly one named data block, or required
            categories, data items, or rows are missing.
    """
    problems = []
    block_names = list(cif_file.keys())

    if len(block_names) != 1:
        found_blocks = ", ".join(block_names) if block_names else "none"
        problems.append(
            "expected exactly one data block because its name is used as the "
            f"structure ID; found {len(block_names)} ({found_blocks})"
        )
    else:
        block_name = block_names[0]
        if not block_name.strip():
            problems.append("the data block name is empty")

        cif_block = cif_file[block_name]
        required_data_items = dict(REQUIRED_MMCIF_DATA_ITEMS)
        if expand_bioassembly and "pdbx_struct_assembly_gen" in cif_block:
            required_data_items.update(REQUIRED_BIOASSEMBLY_DATA_ITEMS)
        if max_polymer_chains is not None:
            required_data_items["pdbx_struct_assembly"] = ("oligomeric_count",)

        for category_name, required_items in required_data_items.items():
            if category_name not in cif_block:
                problems.append(f"missing required category `_{category_name}`")
                continue

            category = cif_block[category_name]
            if category.row_count == 0:
                problems.append(f"required category `_{category_name}` has no rows")

            for item_name in required_items:
                if item_name not in category:
                    problems.append(
                        f"missing required data item `_{category_name}.{item_name}`"
                    )

    if problems:
        location = f" for {source}" if source is not None else ""
        details = "\n".join(f"- {problem}" for problem in problems)
        raise MMCIFPreflightError(
            f"mmCIF preflight validation failed{location}:\n{details}"
        )
