import pytest

from openfold3.core.data.pipelines.preprocessing.caches import pdb_weighted
from openfold3.core.data.primitives.caches.format import (
    PreprocessingDataCache,
    PreprocessingReferenceMoleculeData,
)


def test_canonical_amino_acid_residues_contains_twenty_residues():
    assert len(pdb_weighted.CANONICAL_AMINO_ACID_RESIDUES_3) == 20
    assert "UNK" not in pdb_weighted.CANONICAL_AMINO_ACID_RESIDUES_3


def test_ensure_standard_amino_acid_reference_molecules_seeds_missing_entries(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        pdb_weighted,
        "CANONICAL_AMINO_ACID_RESIDUES_3",
        ("ALA", "GLY"),
    )
    ccd = object()
    monkeypatch.setattr(
        pdb_weighted.CIFFile,
        "read",
        lambda _: ccd,
    )

    generated_residues = []

    def fake_mol_from_ccd_entry(residue, actual_ccd):
        assert actual_ccd is ccd
        generated_residues.append(residue)
        return f"mol-{residue}"

    monkeypatch.setattr(
        pdb_weighted,
        "mol_from_ccd_entry",
        fake_mol_from_ccd_entry,
    )
    monkeypatch.setattr(
        pdb_weighted,
        "prepare_reference_molecule",
        lambda mol: (
            mol,
            {
                "conformer_gen_strategy": "use_fallback",
                "fallback_conformer_pdb_id": None,
                "canonical_smiles": mol,
                "residue_count": 1,
            },
        ),
    )
    monkeypatch.setattr(
        pdb_weighted,
        "write_annotated_sdf",
        lambda _, path: path.touch(),
    )

    cache = PreprocessingDataCache(structure_data={}, reference_molecule_data={})
    reference_molecule_dir = tmp_path / "reference_mols"
    ccd_path = tmp_path / "components.cif"

    pdb_weighted.ensure_standard_amino_acid_reference_molecules(
        preprocessing_cache=cache,
        reference_molecule_dir=reference_molecule_dir,
        ccd_path=ccd_path,
    )

    assert generated_residues == ["ALA", "GLY"]
    assert set(cache.reference_molecule_data) == {"ALA", "GLY"}
    assert cache.reference_molecule_data["GLY"].canonical_smiles == "mol-GLY"
    assert (reference_molecule_dir / "ALA.sdf").is_file()
    assert (reference_molecule_dir / "GLY.sdf").is_file()

    dataset_cache = pdb_weighted.build_provisional_clustered_dataset_cache(
        preprocessing_cache=cache,
        dataset_name="test",
    )
    assert set(dataset_cache.reference_molecule_data) == {"ALA", "GLY"}
    assert not dataset_cache.reference_molecule_data["GLY"].set_fallback_to_nan

    pdb_weighted.ensure_standard_amino_acid_reference_molecules(
        preprocessing_cache=cache,
        reference_molecule_dir=reference_molecule_dir,
        ccd_path=None,
    )
    assert generated_residues == ["ALA", "GLY"]


def test_ensure_standard_amino_acid_references_preserves_existing_artifacts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        pdb_weighted,
        "CANONICAL_AMINO_ACID_RESIDUES_3",
        ("ALA", "GLY"),
    )
    existing_ala_metadata = PreprocessingReferenceMoleculeData(
        conformer_gen_strategy="use_fallback",
        fallback_conformer_pdb_id=None,
        canonical_smiles="existing-ALA",
        residue_count=1,
    )
    cache = PreprocessingDataCache(
        structure_data={},
        reference_molecule_data={"ALA": existing_ala_metadata},
    )
    reference_molecule_dir = tmp_path / "reference_mols"
    reference_molecule_dir.mkdir()
    existing_gly_sdf = reference_molecule_dir / "GLY.sdf"
    existing_gly_sdf.write_text("existing GLY SDF")

    monkeypatch.setattr(pdb_weighted.CIFFile, "read", lambda _: object())
    monkeypatch.setattr(
        pdb_weighted,
        "mol_from_ccd_entry",
        lambda residue, _: f"mol-{residue}",
    )
    monkeypatch.setattr(
        pdb_weighted,
        "prepare_reference_molecule",
        lambda mol: (
            mol,
            {
                "conformer_gen_strategy": "use_fallback",
                "fallback_conformer_pdb_id": None,
                "canonical_smiles": mol,
                "residue_count": 1,
            },
        ),
    )
    written_sdfs = []

    def fake_write_sdf(_, path):
        written_sdfs.append(path.name)
        path.touch()

    monkeypatch.setattr(pdb_weighted, "write_annotated_sdf", fake_write_sdf)

    pdb_weighted.ensure_standard_amino_acid_reference_molecules(
        preprocessing_cache=cache,
        reference_molecule_dir=reference_molecule_dir,
        ccd_path=tmp_path / "components.cif",
    )

    assert cache.reference_molecule_data["ALA"] is existing_ala_metadata
    assert cache.reference_molecule_data["GLY"].canonical_smiles == "mol-GLY"
    assert written_sdfs == ["ALA.sdf"]
    assert existing_gly_sdf.read_text() == "existing GLY SDF"


def test_ensure_standard_amino_acid_reference_molecules_requires_ccd(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        pdb_weighted,
        "CANONICAL_AMINO_ACID_RESIDUES_3",
        ("GLY",),
    )
    cache = PreprocessingDataCache(structure_data={}, reference_molecule_data={})

    with pytest.raises(
        ValueError,
        match=r"Provide `ccd_path`.*Missing or incomplete residues: \['GLY'\]",
    ):
        pdb_weighted.ensure_standard_amino_acid_reference_molecules(
            preprocessing_cache=cache,
            reference_molecule_dir=tmp_path / "reference_mols",
            ccd_path=None,
        )
