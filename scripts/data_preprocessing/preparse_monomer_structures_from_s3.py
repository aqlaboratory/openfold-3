# %%
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
def main(dataset_cache: str, output_dir: str):
    preprocess_pdb_monomer_distilation(
        dataset_cache=Path(dataset_cache), output_dir=Path(output_dir), num_workers=1
    )


if __name__ == "__main__":
    main()
# %%
