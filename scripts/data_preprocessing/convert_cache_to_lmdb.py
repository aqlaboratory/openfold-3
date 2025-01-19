"""Generates an LMDB cache from an existing dataset cache json file.

Usage:
    export PYTHONPATH="/path/to/openfold3":$PYTHONPATH
    python scripts/data_preprocessing/convert_cache_to_lmdb.py \
        --dataset_cache <dataset_cache> \
        --output_lmdb_dir <output_dir> \
        --map_size 2*(1024**3)
"""

from pathlib import Path

import click

from openfold3.core.data.primitives.caches.lmdb import convert_datacache_to_lmdb


@click.command()
@click.option(
    "--dataset_cache",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Starting dataset cache json file",
)
@click.option(
    "--output_lmdb_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory to write LMDB of cache",
)
@click.option(
    "--map_size",
    type=int,
    default=2 * (1024**3),
    help="Size of json file in bytes, specify a value "
    "slightly larger than actual size of json file.",
)
def main(
    dataset_cache: Path,
    output_lmdb_dir: Path,
    map_size: int,
) -> None:
    """Creates a LMDB json dict from an existing dataset cache"""
    convert_datacache_to_lmdb(dataset_cache, output_lmdb_dir, map_size)


if __name__ == "__main__":
    main()
