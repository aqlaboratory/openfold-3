import numpy as np
import pytest
from biotite.structure.io.pdbx import CIFBlock, CIFCategory, CIFFile

from openfold3.core.data.pipelines.preprocessing.mmcif import (
    REQUIRED_MMCIF_DATA_ITEMS,
    MMCIFPreflightError,
    validate_mmcif_for_preprocessing,
)


def _valid_cif_file() -> CIFFile:
    categories = {
        category_name: CIFCategory(
            {item_name: np.array(["1"]) for item_name in required_items}
        )
        for category_name, required_items in REQUIRED_MMCIF_DATA_ITEMS.items()
    }
    return CIFFile({"TEST": CIFBlock(categories)})


def test_validate_mmcif_for_preprocessing_accepts_complete_schema():
    validate_mmcif_for_preprocessing(_valid_cif_file())


def test_validate_mmcif_for_preprocessing_reports_all_missing_items():
    cif_file = _valid_cif_file()
    del cif_file["TEST"]["exptl"]
    del cif_file["TEST"]["atom_site"]["Cartn_x"]
    del cif_file["TEST"]["entity_poly"]["pdbx_seq_one_letter_code_can"]

    with pytest.raises(MMCIFPreflightError) as exc_info:
        validate_mmcif_for_preprocessing(cif_file, source="custom.cif")

    message = str(exc_info.value)
    assert "custom.cif" in message
    assert "missing required category `_exptl`" in message
    assert "missing required data item `_atom_site.Cartn_x`" in message
    assert (
        "missing required data item `_entity_poly.pdbx_seq_one_letter_code_can`"
        in message
    )


def test_validate_mmcif_for_preprocessing_rejects_multiple_data_blocks():
    cif_file = _valid_cif_file()
    cif_file["SECOND"] = CIFBlock({})

    with pytest.raises(
        MMCIFPreflightError,
        match=r"expected exactly one data block.*found 2 \(TEST, SECOND\)",
    ):
        validate_mmcif_for_preprocessing(cif_file)


def test_validate_mmcif_for_preprocessing_checks_partial_bioassembly_schema():
    cif_file = _valid_cif_file()
    cif_file["TEST"]["pdbx_struct_assembly_gen"] = CIFCategory(
        {"assembly_id": np.array(["1"]), "asym_id_list": np.array(["A"])}
    )

    with pytest.raises(MMCIFPreflightError) as exc_info:
        validate_mmcif_for_preprocessing(cif_file)

    message = str(exc_info.value)
    assert (
        "missing required data item `_pdbx_struct_assembly_gen.oper_expression`"
        in message
    )
    assert "missing required category `_pdbx_struct_oper_list`" in message


def test_validate_mmcif_for_preprocessing_checks_requested_chain_count_metadata():
    with pytest.raises(
        MMCIFPreflightError,
        match=r"missing required category `_pdbx_struct_assembly`",
    ):
        validate_mmcif_for_preprocessing(
            _valid_cif_file(),
            max_polymer_chains=1000,
        )
