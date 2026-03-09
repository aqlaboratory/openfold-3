"""
CLI wrapper that converts a CCD CIF file into BinaryCIF for Biotite.
"""

import argparse
import logging
from pathlib import Path

from openfold3.core.data.primitives.structure.biotite_ccd import (
    DEFAULT_BIOTITE_CCD_CATEGORIES,
    concatenate_ccd,
)


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
        help="Local path to a CCD CIF file.",
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
        categories=DEFAULT_BIOTITE_CCD_CATEGORIES,
    )
    compressed_ccd.write(args.output)


if __name__ == "__main__":
    main()
