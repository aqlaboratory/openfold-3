# Copyright 2025 AlQuraishi Laboratory
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

from pathlib import Path

import pytest

from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz
from openfold3.core.data.primitives.structure.tokenization import tokenize_atom_array
from openfold3.tests.custom_assert_utils import assert_atomarray_equal

TEST_DIR = Path(__file__).parent / "test_data/tokenization"

paths = []
ids = ["1ema", "1pwc", "5seb", "5tdj", "6znc"]
for id in ids:
    paths.append(
        (
            TEST_DIR / "inputs" / f"{id}_raw_bonds_unfiltered.npz",
            TEST_DIR / "outputs" / f"{id}_tokenized_bonds_unfiltered.npz",
        )
    )


@pytest.mark.parametrize(
    "input_atom_array_path, precomputed_atom_array_path",
    paths,
    ids=ids,
)
def test_tokenizer(
    input_atom_array_path: Path,
    precomputed_atom_array_path: Path,
):
    """Checks that the tokenizer adds the correct token annotations to the atom array.

    Args:
        input_atom_array_path (Path):
            Path to the input atom array that is to be tokenized.
        precomputed_atom_array_path (Path):
            Path to the precomputed atom array with the expected token annotations.
    """
    atom_array_in = read_atomarray_from_npz(input_atom_array_path)
    atom_array_out = read_atomarray_from_npz(precomputed_atom_array_path)

    tokenize_atom_array(atom_array_in)

    assert_atomarray_equal(atom_array_out, atom_array_in)
