"""
Script to preprocess the CCD to a format that can be used with Biotite's set_ccd_path.
This is required if one wants Biotite's internal functions to use a specific CCD
version, e.g. the one matching a particular PDB release. This script uses functions from
Biotite's setup_ccd.py, and therefore is subject to the following license:

BSD 3-Clause License
====================

Copyright 2017, The Biotite contributors All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of
   conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list
   of conditions and the following disclaimer in the documentation and/or other
   materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be
   used to endorse or promote products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
import contextlib
import logging
from collections import defaultdict
from io import StringIO
from pathlib import Path

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


def concatenate_ccd(ccd_path: Path, categories=None):
    """
    Create the CCD in BinaryCIF format with each category containing the data of all
    blocks.

    Parameters
    ----------
    ccd_path : Path or str
        The local path to a CCD cif file.
    categories : list of str, optional
        The names of the categories to include. By default, all categories from the CCD
        are included.

    Returns
    -------
    compressed_file : BinaryCIFFile
        The compressed CCD in BinaryCIF format.
    """
    logging.info("Reading CCD from file...")
    ccd_path = Path(ccd_path)
    ccd_cif_text = ccd_path.read_text()

    ccd_file = CIFFile.read(StringIO(ccd_cif_text))

    compressed_block = BinaryCIFBlock()
    if categories is None:
        categories = _list_all_category_names(ccd_file)
    for category_name in categories:
        logging.info(f"Concatenating and compressing '{category_name}' category...")
        concatenated_category = _concatenate_blocks_into_category(
            ccd_file, category_name
        )
        compressed_block[category_name] = compress(concatenated_category)

    logging.info("Creating BinaryCIF file with concatenated CCD...")
    compressed_file = BinaryCIFFile()
    compressed_file["components"] = compressed_block

    return compressed_file


def _concatenate_blocks_into_category(pdbx_file, category_name):
    """
    Concatenate the given category from all blocks into a single category.

    Parameters
    ----------
    pdbx_file : PDBxFile
        The PDBx file, whose blocks should be concatenated.
    category_name : str
        The name of the category to concatenate.

    Returns
    -------
    category : BinaryCIFCategory
        The concatenated category.
    """
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
                # Column missing in this block: treat as missing
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


def _list_all_column_names(pdbx_file, category_name):
    """
    Get all columns that exist in any block for a given category.
    """
    columns_names = set()
    for block in pdbx_file.values():
        if category_name in block:
            columns_names.update(block[category_name].keys())
    return sorted(columns_names)


def _list_all_category_names(pdbx_file):
    """
    Get all categories that exist in any block.
    """
    category_names = set()
    for block in pdbx_file.values():
        category_names.update(block.keys())
    return sorted(category_names)


def _into_fitting_type(string_array, mask):
    """
    Try to find a numeric type for a string ndarray, if possible.

    Parameters
    ----------
    string_array : ndarray, dtype=string
        The array to convert.
    mask : ndarray, dtype=uint8
        Only values in `string_array` where the mask is MaskValue.PRESENT are
        considered for type conversion.

    Returns
    -------
    array : ndarray
        The array converted into an appropriate dtype.
    """
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Converts a CCD CIF-file into BinaryCIF format that can be used with "
            "biotite's set_ccd_path."
        )
    )
    parser.add_argument(
        "ccd_path",
        type=Path,
        help="Local path to a CCD cif file.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output file path.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    compressed_ccd = concatenate_ccd(
        ccd_path=args.ccd_path,
        categories=["chem_comp", "chem_comp_atom", "chem_comp_bond"],
    )
    compressed_ccd.write(args.output)


if __name__ == "__main__":
    main()
