#%%
from openfold3.core.data.pipelines.preprocessing.structure import preprocess_pdb_monomer_distilation 
from pathlib import Path
import click 
# %%
@click.command()
@click.option(
    "--dataset_cache",
    type=str,
    default="/pscratch/sd/v/vss2134/of3-dev/monomer_distillation_set_datacache.json",
    help="Path to the dataset cache file."
)
@click.option(
    "--output_dir",
    type=str,
    default="/pscratch/sd/v/vss2134/monomer_structures_preparsed",
    help="Path to the output directory."
)
def main(
    dataset_cache: str, 
    output_dir: str
):
    preprocess_pdb_monomer_distilation(
        dataset_cache = Path(dataset_cache), 
        output_dir = Path(output_dir),
        num_workers = 1
    )

if __name__ == "__main__":
    main()
# %%
