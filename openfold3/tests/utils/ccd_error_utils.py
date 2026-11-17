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

"""Builders for the expected error messages of the ligand CCD checks.

The messages are produced by ``check_ligand_atoms_in_ccd`` and
``remove_non_CCD_atoms`` in ``openfold3.core.data.primitives.structure.cleanup``.
To add a test case, describe the offending residues here instead of writing the
message out: the result is meant for ``pytest.raises(..., match=re.escape(...))``,
which only needs a fragment of the full message.
"""


def ligand_mismatch_message(
    *ligands: tuple[str, int, str, list[str]],
) -> str:
    """Expected fragment of the error for ligand atoms that are not in the CCD entry.

    Args:
        *ligands:
            One (chain_id, res_id, res_name, atom names) tuple per offending ligand,
            where the atom names are those that the CCD entry of res_name lacks.
            Give them in the order of the AtomArray.

    Returns:
        The tuples as they are listed in the error, e.g.
        "('B', 2, 'LIG', ['S1', 'O3'])".
    """
    return ", ".join(str(ligand) for ligand in ligands)


def residue_not_in_ccd_message(res_name: str, chain_id: str, res_id: int) -> str:
    """Expected fragment of the error for a residue name that is not in the CCD."""
    return f"Residue {res_name} (chain {chain_id}, residue {res_id}) is not in the CCD"
