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

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def get_sequence_hash(sequence_str: str) -> str:
    """Generates a SHA-256 hash for the given sequence string."""
    hasher = hashlib.sha256()
    hasher.update(sequence_str.encode("utf-8"))
    return hasher.hexdigest()


def get_file_content_hash(*parts: Path | str | None) -> str:
    """Generates a SHA-256 hash over file contents interleaved with plain strings.

    Files are hashed by content rather than by path so that editing a file in place
    invalidates anything keyed on it. A `Path` that cannot be read contributes its
    path string instead, so the hash stays defined for missing files. Each part is
    length-delimited, so no concatenation of parts can be confused for another.

    Args:
        *parts (Path | str | None):
            Files to hash by content, and plain strings (labels, selectors) to hash
            verbatim. None parts are hashed as a distinct "absent" marker.

    Returns:
        str:
            The hex digest.
    """
    hasher = hashlib.sha256()
    for part in parts:
        if part is None:
            hasher.update(b"\x00none\x00")
            continue
        if isinstance(part, Path):
            try:
                with open(part, "rb") as f:
                    hasher.update(b"\x00file\x00")
                    while chunk := f.read(_CHUNK_SIZE):
                        hasher.update(chunk)
                continue
            except OSError:
                # Unreadable/missing: fall back to the path so the hash stays defined.
                part = str(part)
        encoded = str(part).encode("utf-8")
        hasher.update(b"\x00str\x00" + str(len(encoded)).encode("ascii") + b"\x00")
        hasher.update(encoded)
    return hasher.hexdigest()
