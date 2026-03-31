# Copyright 2026 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

"""Common utilities for data pipeline tools."""

import contextlib
import datetime
import getpass
import logging
import shutil
import tempfile
import time
from pathlib import Path


def get_of3_tmpdir(subdir: str | None = None) -> Path:
    """Return a user-namespaced OpenFold3 temporary directory.

    Follows the same convention as pytest (``/tmp/pytest-of-<user>/``):
    ``<tmpdir>/of3-of-<user>/<subdir>``.  Respects ``$TMPDIR`` and other
    platform conventions via :func:`tempfile.gettempdir`.

    The returned directory is created on disk if it does not already exist.
    """
    base = Path(tempfile.gettempdir()) / f"of3-of-{getpass.getuser()}"
    path = base / subdir if subdir else base
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def tmpdir_manager(base_dir: str | None = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
    logging.info("Started %s", msg)
    tic = time.perf_counter()
    yield
    toc = time.perf_counter()
    logging.info("Finished %s in %.3f seconds", msg, toc - tic)


def to_date(s: str):
    return datetime.datetime(year=int(s[:4]), month=int(s[5:7]), day=int(s[8:10]))
