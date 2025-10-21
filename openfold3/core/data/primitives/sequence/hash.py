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

import hashlib


def get_sequence_hash(sequence_str: str) -> str:
    """Generates a SHA-256 hash for the given sequence string."""
    hasher = hashlib.sha256()
    hasher.update(sequence_str.encode("utf-8"))
    return hasher.hexdigest()
