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
#
# This module adapts logic from Biotite's CCD setup utilities and is therefore
# additionally subject to the following BSD 3-Clause License notice:
#
# BSD 3-Clause License
# ====================
#
# Copyright 2017, The Biotite contributors All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Utilities for preparing CCD files for Biotite's ``set_ccd_path()``."""

import contextlib
import logging
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import biotite.structure as struc
import numpy as np
from biotite.structure.io.pdbx import (
    BinaryCIFBlock,
    BinaryCIFCategory,
    BinaryCIFColumn,
    BinaryCIFFile,
    CIFFile,
    MaskValue,
    compress,
)

DEFAULT_BIOTITE_CCD_CATEGORIES = ("chem_comp", "chem_comp_atom", "chem_comp_bond")


def configure_biotite_ccd(
    ccd_file_path: Path | str | None,
    categories: Sequence[str] = DEFAULT_BIOTITE_CCD_CATEGORIES,
) -> tempfile.TemporaryDirectory | None:
    """Configure Biotite's global CCD path from a custom CCD file.

    Args:
        ccd_file_path:
            Path to a custom CCD file. Supported suffixes are ``.bcif`` and
            ``.cif``. If ``None``, no configuration changes are applied.
        categories:
            Category names to keep when converting ``.cif`` input into temporary
            BinaryCIF format for Biotite.

    Returns:
        A ``TemporaryDirectory`` handle if conversion from ``.cif`` was performed;
        otherwise ``None``. The caller should keep this handle alive while Biotite
        uses the temporary BinaryCIF path.
    """
    if ccd_file_path is None:
        return None

    ccd_path = Path(ccd_file_path)
    if not ccd_path.exists():
        raise FileNotFoundError(f"CCD file not found: {ccd_path}")
    if not ccd_path.is_file():
        raise ValueError(f"CCD path is not a file: {ccd_path}")

    tmp_ccd_dir_handle: tempfile.TemporaryDirectory | None = None
    ccd_suffix = ccd_path.suffix.lower()
    if ccd_suffix == ".bcif":
        biotite_ccd_path = ccd_path
    elif ccd_suffix == ".cif":
        # Keep converted file in a temp directory so Biotite can read it by path.
        tmp_ccd_dir_handle = tempfile.TemporaryDirectory(prefix="of3_biotite_ccd_")
        biotite_ccd_path = Path(tmp_ccd_dir_handle.name) / "components.bcif"
        logging.warning(
            "Converting custom CCD to temporary BinaryCIF for Biotite "
            "(this may take over a minute).",
        )
        concatenate_ccd(
            ccd_path=ccd_path,
            categories=categories,
        ).write(biotite_ccd_path)
    else:
        raise ValueError(
            f"Unsupported CCD file extension for {ccd_path}. "
            "Expected '.cif' or '.bcif'."
        )

    struc.info.set_ccd_path(biotite_ccd_path)
    try:
        component_count = len(struc.info.all_residues())
    except Exception as exc:
        raise ValueError(
            f"Failed to load Biotite CCD from {biotite_ccd_path}. "
            "Ensure the file exists and is a valid BinaryCIF CCD."
        ) from exc
    if component_count == 0:
        raise ValueError(f"Biotite CCD at {biotite_ccd_path} contained no components.")

    logging.info(
        "Set Biotite CCD path to %s with %d components.",
        biotite_ccd_path,
        component_count,
    )

    # Returning the handle keeps the temporary directory alive for the lifetime
    # of the inference dataset.
    return tmp_ccd_dir_handle


def concatenate_ccd(
    ccd_path: Path | str,
    categories: Sequence[str] | None = None,
) -> BinaryCIFFile:
    """Create a BinaryCIF CCD where each category is concatenated across all blocks.

    Args:
        ccd_path:
            Local path to a CCD CIF file.
        categories:
            Optional category names to include. If omitted, all categories found in
            the CCD are included.

    Returns:
        A compressed BinaryCIF representation of the CCD.
    """
    logging.info("Reading CCD from file...")
    ccd_path = Path(ccd_path)
    ccd_file = CIFFile.read(StringIO(ccd_path.read_text()))

    compressed_block = BinaryCIFBlock()
    if categories is None:
        categories = _list_all_category_names(ccd_file)

    for category_name in categories:
        logging.info("Concatenating and compressing '%s' category...", category_name)
        concatenated_category = _concatenate_blocks_into_category(
            ccd_file, category_name
        )
        compressed_block[category_name] = compress(concatenated_category)

    compressed_file = BinaryCIFFile()
    compressed_file["components"] = compressed_block
    return compressed_file


def _concatenate_blocks_into_category(
    pdbx_file: CIFFile,
    category_name: str,
) -> BinaryCIFCategory:
    """Concatenate one category across all CCD blocks."""
    columns_names = _list_all_column_names(pdbx_file, category_name)
    data_chunks = defaultdict(list)
    mask_chunks = defaultdict(list)

    for block in pdbx_file.values():
        if category_name not in block:
            continue

        category = block[category_name]
        for column_name in columns_names:
            if column_name in category:
                column = category[column_name]
                data_chunks[column_name].append(column.data.array)
                if column.mask is not None:
                    mask_chunks[column_name].append(column.mask.array)
                else:
                    mask_chunks[column_name].append(
                        np.full(category.row_count, MaskValue.PRESENT, dtype=np.uint8)
                    )
            else:
                # Column missing in this block: treat values as missing.
                data_chunks[column_name].append(
                    np.full(category.row_count, "", dtype="U1")
                )
                mask_chunks[column_name].append(
                    np.full(category.row_count, MaskValue.MISSING, dtype=np.uint8)
                )

    bcif_columns = {}
    for col_name in columns_names:
        data = np.concatenate(data_chunks[col_name])
        mask = np.concatenate(mask_chunks[col_name])
        data = _into_fitting_type(data, mask)
        if np.all(mask == MaskValue.PRESENT):
            mask = None
        bcif_columns[col_name] = BinaryCIFColumn(data, mask)

    return BinaryCIFCategory(bcif_columns)


def _list_all_column_names(pdbx_file: CIFFile, category_name: str) -> list[str]:
    """Get all columns that exist in any block for a given category."""
    columns_names = set()
    for block in pdbx_file.values():
        if category_name in block:
            columns_names.update(block[category_name].keys())
    return sorted(columns_names)


def _list_all_category_names(pdbx_file: CIFFile) -> list[str]:
    """Get all categories that exist in any block."""
    category_names = set()
    for block in pdbx_file.values():
        category_names.update(block.keys())
    return sorted(category_names)


def _into_fitting_type(string_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Try to find a numeric dtype for string values where possible."""
    mask_bool = mask == MaskValue.PRESENT
    values = string_array[mask_bool]
    try:
        values = values.astype(int)
    except ValueError:
        with contextlib.suppress(ValueError):
            values = values.astype(float)

    array = np.zeros(string_array.shape, dtype=values.dtype)
    array[mask_bool] = values
    return array
