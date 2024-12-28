# %%
import json
from pathlib import Path

import click

from openfold3.core.data.pipelines.preprocessing.structure import (
    preprocess_pdb_monomer_distilation,
)


# %%
@click.command()
@click.option(
    "--dataset_cache",
    type=str,
    help="Path to the dataset cache file.",
)
@click.option(
    "--output_dir",
    type=str,
    help="Path to the output directory.",
)
@click.option(
    "--num_workers",
    type=int,
    default=1,
    help="Number of workers to use for parallel processing.",
)
@click.option(
    "--s3_config",
    type=str,
    help="Path to the s3 client config file.",
)
def main(dataset_cache: str, output_dir: str, s3_config: str, num_workers: int = 1):
    with open(s3_config) as f:
        s3_config = json.load(f)
    preprocess_pdb_monomer_distilation(
        dataset_cache=Path(dataset_cache),
        output_dir=Path(output_dir),
        num_workers=num_workers,
        s3_config=s3_config,
    )


if __name__ == "__main__":
    main()
# %%
