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

"""Tests for `query_extraction.py` against real mmCIF fixtures.

Each case documents a specific behavior the fixture was chosen to exercise --
see the module docstring of `query_extraction.py` for the extraction rules
being verified.
"""

from pathlib import Path

import biotite.structure.io.pdbx as pdbx

import openfold3
from openfold3.core.data.primitives.structure.metadata import get_cif_block
from openfold3.core.data.primitives.structure.query_extraction import (
    chains_from_cif,
)
from openfold3.core.data.resources.residues import MoleculeType
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
)

MMCIFS_DIR = Path(openfold3.__file__).parent / "tests" / "test_data" / "mmcifs"


def _seqres_length(cif_path: Path) -> int:
    """Independent oracle: the length of the full `entity_poly` canonical
    (SEQRES) sequence, as opposed to `chains_from_cif`'s coordinate-derived
    one. Only meaningful for single-polymer-entity fixtures like 7l39, where
    there's no ambiguity about which entity's sequence this is.
    """
    cif_file = pdbx.CIFFile.read(cif_path)
    cif_block = get_cif_block(cif_file)
    seqres = cif_block["entity_poly"]["pdbx_seq_one_letter_code_can"].as_array()[0]
    return len(seqres.replace("\n", ""))


def test_1ubq_pure_protein_monomer():
    """1ubq: single protein chain, no ligands/hetero groups, no missing density."""
    result = chains_from_cif(MMCIFS_DIR / "1ubq.cif")

    assert len(result.chains) == 1
    (chain,) = result.chains
    assert chain.chain_ids == ["A"]
    assert chain.molecule_type == MoleculeType.PROTEIN
    assert chain.sequence == (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    )
    assert chain.non_canonical_residues is None
    assert result.warnings == []


def test_7l39_separates_ligand_from_protein_and_drops_crystallization_aids():
    """7l39: TRS/CL/MBN/BME hetero groups all share author chain "A" with the
    protein in the raw CIF. MBN (toluene) is the real ligand and must come out
    as its own chain; TRS (buffer), CL (ion), and BME (reducing agent) are
    crystallization aids/ions and should be dropped by default, not merged
    into the protein sequence as "non-canonical residues".
    """
    result = chains_from_cif(MMCIFS_DIR / "7l39.cif")

    by_id = {c.chain_ids[0]: c for c in result.chains}
    assert set(by_id) == {"A", "D"}

    protein = by_id["A"]
    assert protein.molecule_type == MoleculeType.PROTEIN
    # Ends at the last residue with resolved coordinates in this deposition;
    # does not include unresolved C-terminal residues from the full construct.
    assert protein.sequence.endswith("TGTWDAYK")
    assert len(protein.sequence) == 162
    assert protein.non_canonical_residues is None
    # 7L39 is the L99A cavity mutant (the mutation that creates the pocket
    # toluene binds in) -- position 99 (1-indexed) must read A, not wild-type L.
    assert protein.sequence[98] == "A"

    # Confirm against an independent oracle that this is really the
    # coordinate-derived sequence, not the full entity_poly SEQRES sequence
    # (which includes 2 more, unresolved, C-terminal residues here).
    seqres_length = _seqres_length(MMCIFS_DIR / "7l39.cif")
    assert seqres_length == 164
    assert len(protein.sequence) != seqres_length

    ligand = by_id["D"]
    assert ligand.molecule_type == MoleculeType.LIGAND
    assert ligand.ccd_codes == ["MBN"]

    [warning] = result.warnings
    assert all(code in warning for code in ("TRS", "CL", "BME"))
    assert "MBN" not in warning


def test_7l39_keep_excluded_restores_crystallization_aids_as_ligands():
    """The same TRS/CL/BME groups become ligand chains when explicitly asked for."""
    result = chains_from_cif(MMCIFS_DIR / "7l39.cif", keep_excluded=True)

    ccd_codes_by_id = {
        c.chain_ids[0]: c.ccd_codes
        for c in result.chains
        if c.molecule_type == MoleculeType.LIGAND
    }
    assert ccd_codes_by_id == {
        "B": ["TRS"],
        "C": ["CL"],
        "D": ["MBN"],
        "E": ["BME"],
    }
    assert result.warnings == []


def test_1pzp_duplicate_ligand_copies_become_separate_chains():
    """1pzp: FTA (the pocket-constraint example's ligand) is present in two
    copies in the asymmetric unit, at distinct label_asym_ids -- both should
    be kept as separate ligand chains, not deduplicated or merged.
    """
    result = chains_from_cif(MMCIFS_DIR / "1pzp.cif")

    ligands = [c for c in result.chains if c.molecule_type == MoleculeType.LIGAND]
    assert len(ligands) == 2
    assert {c.ccd_codes[0] for c in ligands} == {"FTA"}
    assert len({c.chain_ids[0] for c in ligands}) == 2

    proteins = [c for c in result.chains if c.molecule_type == MoleculeType.PROTEIN]
    assert len(proteins) == 1
    assert len(proteins[0].sequence) == 263


def test_2q2k_dna_with_non_canonical_residue_and_homomeric_protein():
    """2q2k: DNA chain with 5-iodouridine (5IU, a modified nucleotide) at two
    positions -- must be classified as DNA (not dropped as a ligand) with
    those positions recorded under non_canonical_residues, and the two nearly-
    identical protein chains (a real single-residue length difference, not a
    bug) must both come through. EPE (a crystallization aid) must be dropped.
    """
    result = chains_from_cif(MMCIFS_DIR / "2q2k.cif")

    by_id = {c.chain_ids[0]: c for c in result.chains}
    assert set(by_id) == {"F", "A", "B"}

    dna = by_id["F"]
    assert dna.molecule_type == MoleculeType.DNA
    assert dna.sequence == "AGTATANACNAGTATATACT"
    assert dna.non_canonical_residues == {7: "5IU", 10: "5IU"}

    assert by_id["A"].molecule_type == MoleculeType.PROTEIN
    assert by_id["B"].molecule_type == MoleculeType.PROTEIN
    assert len(by_id["A"].sequence) == 45
    assert len(by_id["B"].sequence) == 44

    assert not any(c.molecule_type == MoleculeType.LIGAND for c in result.chains)
    dropped_warning = next(w for w in result.warnings if "Dropped" in w)
    assert "EPE" in dropped_warning


def test_5kc1_many_chains_and_ions_stress_case():
    """5kc1: 12 short protein chains plus ~50 ion/crystallization-aid hetero
    groups (NA, CL, NO3, EDO, IOD, NH4, SO4, ...) -- none of the ions should
    survive as ligand chains, and no protein chain should be misclassified.
    """
    result = chains_from_cif(MMCIFS_DIR / "5kc1.cif")

    proteins = [c for c in result.chains if c.molecule_type == MoleculeType.PROTEIN]
    ligands = [c for c in result.chains if c.molecule_type == MoleculeType.LIGAND]
    assert len(proteins) == 12
    assert ligands == []

    dropped_warning = next(w for w in result.warnings if "Dropped" in w)
    for code in ("NA", "CL", "NO3", "EDO", "IOD", "NH4", "SO4"):
        assert code in dropped_warning


def test_extracted_chains_validate_as_a_real_inference_query():
    """Extracted chains must be usable as-is to build a valid InferenceQuerySet
    -- the actual downstream consumer of a query.json's chains list.
    """
    result = chains_from_cif(MMCIFS_DIR / "7l39.cif")

    query_set = InferenceQuerySet.model_validate(
        {"queries": {"pdb_7l39": {"chains": result.chains}}}
    )
    assert list(query_set.queries) == ["pdb_7l39"]
