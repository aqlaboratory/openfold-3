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

import pytest

from openfold3.core.data.pipelines.preprocessing.structure import _discover_cif_files


def test_discover_cif_files_recursively_in_deterministic_order(tmp_path):
    nested_dir = tmp_path / "nested" / "entry"
    nested_dir.mkdir(parents=True)

    top_level_cif = tmp_path / "2def.cif"
    nested_cif = nested_dir / "1abc.cif"
    ignored_file = nested_dir / "notes.txt"
    top_level_cif.touch()
    nested_cif.touch()
    ignored_file.touch()

    assert _discover_cif_files(tmp_path) == [top_level_cif, nested_cif]


def test_discover_cif_files_rejects_duplicate_stems(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    (first_dir / "1abc.cif").touch()
    (second_dir / "1abc.cif").touch()

    with pytest.raises(
        ValueError,
        match=r"unique stems.*1abc: first/1abc\.cif, second/1abc\.cif",
    ):
        _discover_cif_files(tmp_path)
