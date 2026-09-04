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

from __future__ import annotations

import pytest

from openfold3.steering.potentials import CLASS_REGISTRY, Potential, register


@pytest.fixture
def isolated_registry():
    """Let a test register throwaway potentials without leaking them.

    CLASS_REGISTRY is module-global and is iterated by feature-key
    derivation and by settings validation, so a test-only potential left
    behind would change unrelated tests' behaviour.
    """
    original = dict(CLASS_REGISTRY)
    try:
        yield CLASS_REGISTRY
    finally:
        CLASS_REGISTRY.clear()
        CLASS_REGISTRY.update(original)


@pytest.fixture
def register_throwaway(isolated_registry):
    """Register a throwaway potential under a caller-chosen snake_case name.

    Each test picks its own name so no two tests can collide through the
    global registry, independently of the ordering pytest happens to choose.
    """

    def _register(name: str, arity: int = 2) -> type[Potential]:
        class_name = "".join(part.title() for part in name.split("_"))
        potential = type(
            class_name,
            (Potential,),
            {"arity": arity, "compute_variable": lambda self, positions: None},
        )
        return register(name)(potential)

    return _register
