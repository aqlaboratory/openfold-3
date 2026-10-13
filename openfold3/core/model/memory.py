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

import sys


def assert_sole_holder(t, *, in_container=False) -> None:
    """Assert that `t` is the only live Python reference to this object.

    Used in offload-inference loops to verify that a tensor has no unexpected
    references before it is moved to CPU, ensuring in-place memory reuse works
    as intended.

    ``in_container`` is required to indicate if the reference is a container
    subscript like ``container[i]``, which creates an additional counted
    reference on the evaluation stack, so the expected count differs.

    The expected count changes across Python versions because of the LOAD_FAST
    optimisation introduced in Python 3.14: in older versions the function call
    itself keeps an extra reference on the caller's stack and getrefcount adds
    its own temporary, whereas Python 3.14 steals the caller's reference and no
    longer creates a temporary for FAST_LOCAL arguments.  See:
    https://docs.python.org/3.14/whatsnew/3.14.html#whatsnew314-refcount
    """
    expected = 3
    if sys.version_info >= (3, 14):
        expected = 2 if in_container else 1

    assert sys.getrefcount(t) == expected
