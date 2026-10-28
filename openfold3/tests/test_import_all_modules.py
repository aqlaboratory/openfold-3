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

"""Every openfold3 module imports (issue #479).

Catches failures that only surface at import time, e.g. a stdlib API or an
evaluated annotation that needs a newer Python, or a newer dependency, than
`requires-python` admits. CI runs this against an installed wheel on the
minimum supported Python (see .github/workflows/python-compat.yml).
"""

import importlib
import pkgutil

import pytest

import openfold3

# Not installed by a plain `pip install openfold3` on CPU
OPTIONAL_DEPENDENCIES = {
    "cuequivariance",
    "cuequivariance_ops_torch",
    "cuequivariance_torch",
    "deepspeed",
    "mpi4py",
    "triton",
}

# TODO: delete these modules; they import openfold3.core.data.legacy, which was
# removed in fc3b93ea
BROKEN_MODULES = {
    "openfold3.core.data.tools.hhsearch",
    "openfold3.core.data.tools.hmmsearch",
    "openfold3.core.data.tools.jackhmmer",
    "openfold3.core.data.tools.parse_msa_files",
}


def _all_modules() -> list[str]:
    # Packages that fail to import are still listed (and fail below); only
    # their submodules go undiscovered
    modules = pkgutil.walk_packages(
        openfold3.__path__, "openfold3.", onerror=lambda name: None
    )
    return sorted(m.name for m in modules if not m.name.startswith("openfold3.tests"))


@pytest.mark.parametrize(
    "module_name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                raises=ModuleNotFoundError, strict=True, reason="imports legacy"
            ),
        )
        if name in BROKEN_MODULES
        else name
        for name in _all_modules()
    ],
)
def test_module_imports(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] in OPTIONAL_DEPENDENCIES:
            pytest.skip(f"optional dependency {e.name} is not installed")
        raise
