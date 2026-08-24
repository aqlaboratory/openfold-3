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

"""Derives inference-query `Chain` entries from a parsed structure.

This is the inverse direction of ``query.py`` (which turns a `Query` into an
`AtomArray`): given a structure already read from a CIF file, classify each
chain as protein/dna/rna or ligand and produce one `Chain` per polymer chain
plus one `Chain` per distinct hetero/ligand group, suitable for assembling an
inference `query.json` from an existing structure.
"""

from pathlib import Path
from typing import NamedTuple

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from biotite.structure import AtomArray

from openfold3.core.data.resources.lists import (
    CRYSTALLIZATION_AIDS,
    IONS,
    LIGAND_EXCLUSION_LIST,
)
from openfold3.core.data.resources.residues import (
    DNA_RESTYPE_3TO1,
    PROTEIN_RESTYPE_3TO1,
    RNA_RESTYPE_3TO1,
    MoleculeType,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import Chain

#: Crystallization aids, common additives, and monoatomic ions dropped from
#: extracted ligand chains by default (AlphaFold3 SI Tables 9-10 + the ion list).
EXCLUDED_HETERO_CODES = frozenset(
    {*CRYSTALLIZATION_AIDS, *LIGAND_EXCLUSION_LIST, *IONS}
)


class ChainExtractionResult(NamedTuple):
    """Chains extracted from a structure, plus warnings worth a human's attention."""

    chains: list[Chain]
    warnings: list[str]


def _build_polymer_chain(
    chain_id: str, res_names: list[str], is_nucleic: bool
) -> Chain:
    """Builds a protein/dna/rna `Chain` from an ordered list of 3-letter residue names.

    Any residue name outside the canonical alphabet is recorded under
    `non_canonical_residues` (1-based index -> 3-letter code) rather than
    dropped, so the chain's sequence length always matches its residue count.
    """
    if not is_nucleic:
        table, unknown_code, molecule_type = (
            PROTEIN_RESTYPE_3TO1,
            "X",
            MoleculeType.PROTEIN,
        )
    else:
        dna_votes = sum(r in DNA_RESTYPE_3TO1 for r in res_names)
        rna_votes = sum(r in RNA_RESTYPE_3TO1 for r in res_names)
        molecule_type = MoleculeType.DNA if dna_votes >= rna_votes else MoleculeType.RNA
        table = (
            DNA_RESTYPE_3TO1 if molecule_type == MoleculeType.DNA else RNA_RESTYPE_3TO1
        )
        unknown_code = "N"

    seq_chars = []
    non_canonical_residues: dict[int, str] = {}
    for i, res_name in enumerate(res_names, start=1):
        if res_name in table:
            seq_chars.append(table[res_name])
        else:
            seq_chars.append(unknown_code)
            non_canonical_residues[i] = res_name

    return Chain(
        molecule_type=molecule_type,
        chain_ids=[chain_id],
        sequence="".join(seq_chars),
        non_canonical_residues=non_canonical_residues or None,
    )


def chains_from_atom_array(
    atom_array: AtomArray, keep_excluded: bool = False
) -> ChainExtractionResult:
    """Classifies each chain in a parsed structure into inference-query `Chain`s.

    Polymer chains (protein/dna/rna) are classified via biotite's CCD-based
    amino-acid/nucleotide detection, which recognizes modified residues (e.g.
    MSE, 5-iodouridine) as polymer residues rather than misfiling them as
    ligands. Hetero (non-polymer) residues are grouped by `label_asym_id` --
    not the author `chain_id` -- since ligands/ions frequently share the
    polymer's author chain letter in a CIF file but are distinct entities.

    Each polymer chain's `sequence` is built strictly from residues present in
    `atom_array` -- i.e. residues with modeled coordinates in this particular
    structure -- not from `entity_poly`'s canonical/SEQRES sequence. Disordered
    termini or internal loop residues with no coordinates are simply absent, so
    the sequence can be shorter than, and discontinuous with, the full
    biological sequence. Reconstructing the canonical sequence would require
    reading `entity_poly.pdbx_seq_one_letter_code_can` instead (see
    `metadata.get_entity_to_canonical_seq_dict` for that direction); this
    function does not do that.

    Args:
        atom_array:
            Structure to extract chains from, expected to carry a
            `label_asym_id` annotation (e.g. from
            ``pdbx.get_structure(..., extra_fields=["label_asym_id"])``) and to
            already have solvent removed.
        keep_excluded:
            If False (default), crystallization aids, common additives, and
            monoatomic ions (see `EXCLUDED_HETERO_CODES`) are dropped instead
            of becoming ligand chains, and reported in `warnings` instead.

    Returns:
        The extracted `Chain`s (polymer chains first, in chain order, then
        ligand chains in first-appearance order) and any warnings about
        non-canonical residues, multi-residue ligand groups, or dropped
        hetero groups.
    """
    polymer_mask = struc.filter_amino_acids(atom_array) | struc.filter_nucleotides(
        atom_array
    )
    polymer_atoms = atom_array[polymer_mask]
    hetero_atoms = atom_array[~polymer_mask]

    chains: list[Chain] = []
    warnings: list[str] = []

    for chain_atoms in struc.chain_iter(polymer_atoms):
        chain_id = str(chain_atoms.chain_id[0])
        _, raw_res_names = struc.get_residues(chain_atoms)
        res_name_list = [str(r) for r in raw_res_names]
        is_nucleic = (
            struc.filter_nucleotides(chain_atoms).sum()
            > struc.filter_amino_acids(chain_atoms).sum()
        )

        chain = _build_polymer_chain(chain_id, res_name_list, is_nucleic)
        if chain.non_canonical_residues:
            warnings.append(
                f"chain {chain_id}: non-canonical residues at positions "
                f"{chain.non_canonical_residues} - verify mapping."
            )
        chains.append(chain)

    used_ids = {chain_id for chain in chains for chain_id in chain.chain_ids}
    dropped: list[tuple[str, str]] = []
    ligand_res_names: dict[str, list[str]] = {}
    for res_atoms in struc.residue_iter(hetero_atoms):
        res_name = str(res_atoms.res_name[0])
        label_asym = str(res_atoms.label_asym_id[0])

        if res_name in EXCLUDED_HETERO_CODES and not keep_excluded:
            dropped.append((label_asym, res_name))
            continue

        ligand_res_names.setdefault(label_asym, []).append(res_name)

    for label_asym, res_names in ligand_res_names.items():
        out_id = label_asym if label_asym not in used_ids else f"{label_asym}_lig"
        used_ids.add(out_id)
        chains.append(
            Chain(
                molecule_type=MoleculeType.LIGAND,
                chain_ids=[out_id],
                ccd_codes=res_names,
            )
        )
        if len(res_names) > 1:
            warnings.append(
                f"chain {label_asym}: multi-residue ligand group ({res_names}) - "
                "polymeric ligand support in OF3 is limited, verify this is intended."
            )

    if dropped:
        warnings.append(
            "Dropped as crystallization aids / ions / common additives (AF3 SI "
            "Tables 9-10 + ion list) - pass keep_excluded=True or add back "
            f"manually if biologically relevant (e.g. a catalytic metal): {dropped}"
        )

    return ChainExtractionResult(chains=chains, warnings=warnings)


def chains_from_cif(
    cif_path: str | Path, keep_excluded: bool = False
) -> ChainExtractionResult:
    """Reads a CIF file and extracts inference-query `Chain`s from it.

    Convenience wrapper around `chains_from_atom_array` for the common case of
    starting from a file path rather than an already-parsed `AtomArray`. See
    `chains_from_atom_array` for the extraction/classification behavior.

    Args:
        cif_path:
            Path to a `.cif` file.
        keep_excluded:
            See `chains_from_atom_array`.

    Returns:
        See `chains_from_atom_array`.
    """
    cif_file = pdbx.CIFFile.read(cif_path)
    atom_array = pdbx.get_structure(
        cif_file,
        model=1,
        use_author_fields=True,
        include_bonds=False,
        extra_fields=["label_asym_id"],
    )
    atom_array = atom_array[~struc.filter_solvent(atom_array)]
    return chains_from_atom_array(atom_array, keep_excluded=keep_excluded)
