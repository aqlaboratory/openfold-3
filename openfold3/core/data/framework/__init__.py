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

# %%
from pathlib import Path


def _import_all_py_files_from_dir(directory: Path):
    """Imports all Python files in the specified directory."""
    for path in directory.glob("*.py"):  # Finds all `.py` files
        __import__(".".join(list(path.parts[-6:-1]) + [path.parts[-1].split(".")[0]]))
    return


_import_all_py_files_from_dir(
    Path(__file__).parent.parent / Path("framework/single_datasets")
)

# %%
