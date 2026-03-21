## Script to generate test examples

python openfold3/tests/test_data/tokenization/construct_tokenization_examples.py

Assumes raw `.cif` files are stored under top level directory.

## Input Files

Generated from raw `.cif` files by parsing into an `atom_array` using:

- `openfold3.core.data.io.structure.cif.parse_mmcif`

followed by sanitizing using:

- `openfold3.core.data.primitives.structure.cleanup`
  - `.convert_MSE_to_MET`
  - `.fix_arginine_naming`
  - `.remove_waters`
  - `.remove_crystallization_aids` (only if X-ray)
  - `.remove_hydrogens`
  - `.remove_small_polymers` (with max residues = 3)
  - `.remove_fully_unknown_polymers`
  - `.remove_non_CCD_atoms`
  - `.canonicalize_atom_order`
- `unresolved.add_unresolved_atoms` # update to show it comes from unresolved module
- `cleanup.remove_std_residue_terminal_atoms`

---

## Output Files

Generated from the parsed, sanitized `atom_arrays` using:

- `openfold3.core.data.primitives.structure.tokenization.tokenize_atom_array`

---

## Commit

[See this commit for the code used to generate the input/output files.](https://github.com/aqlaboratory/openfold-3/pull/124/changes/53ca255b4e30bb7b9585bb87120808dcbecd5dda)
