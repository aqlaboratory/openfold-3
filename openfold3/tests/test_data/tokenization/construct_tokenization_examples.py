from pathlib import Path

from openfold3.core.data.io.structure.atom_array import write_atomarray_to_npz
from openfold3.core.data.io.structure.cif import parse_mmcif
from openfold3.core.data.primitives.structure import cleanup, unresolved
from openfold3.core.data.primitives.structure.component import BiotiteCCDWrapper
from openfold3.core.data.primitives.structure.metadata import get_cif_block
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array

TEST_DIR = Path(__file__).parent


def construct_sanitized_atom_array(example_name: str):
    cif_file, atom_array = parse_mmcif(
        f"raw/{example_name.upper()}.cif",  # <- change this directory as needed
        expand_bioassembly=True,
    )
    cif_data = get_cif_block(cif_file)
    ccd_file = BiotiteCCDWrapper()
    cleanup.convert_MSE_to_MET(atom_array)

    atom_array = cleanup.fix_arginine_naming(atom_array)
    atom_array = cleanup.remove_waters(atom_array)
    atom_array = cleanup.remove_hydrogens(atom_array)
    atom_array = cleanup.remove_small_polymers(atom_array)
    atom_array = cleanup.remove_fully_unknown_polymers(atom_array)
    atom_array = cleanup.remove_non_CCD_atoms(atom_array, ccd_file)
    atom_array = cleanup.canonicalize_atom_order(atom_array, ccd_file)

    atom_array = unresolved.add_unresolved_atoms(atom_array, cif_data, ccd_file)
    atom_array = cleanup.remove_std_residue_terminal_atoms(atom_array)
    return atom_array


if __name__ == "__main__":
    example_names = ["1ema", "1pwc", "5seb", "5tdj", "6znc"]
    for name in example_names:
        sanitized_array = construct_sanitized_atom_array(name)
        write_atomarray_to_npz(
            sanitized_array, TEST_DIR / "inputs" / f"{name}_raw_bonds_unfiltered.npz"
        )

        tokenize_atom_array(sanitized_array)
        write_atomarray_to_npz(
            sanitized_array,
            TEST_DIR / "outputs" / f"{name}_tokenized_bonds_unfiltered.npz",
        )
