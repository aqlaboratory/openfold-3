-- input files --

Generated from raw .cif files by parsing into an atom_array using

openfold3.core.data.io.structure.cif.parse_mmcif

followed by sanitizing using openfold3.core.data.primitives.structure.cleanup

.convert_MSE_to_MET
.fix_arginine_naming
.remove_waters
.remove_crystallization_aids if X-ray
.remove_hydrogens
.remove_small_polymers with max residues = 3
.remove_fully_unknown_polymers
.remove_non_CCD_atoms
.canonicalize_atom_order
.add_unresolved_atoms
.remove_std_residue_terminal_atoms

-- output files --

Generated from the parsed, sanitized atom_arrrays using

openfold3.core.data.primitives.structure.tokenization.tokenize_atom_array

-- Commit -- 

[See this commit for the code used to generate the input/output files.](https://github.com/aqlaboratory/openfold3/pull/239/commits/63e2841fecff266e8c51e3532f716143a9468a95)