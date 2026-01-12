#!/usr/bin/env python3
"""
Create nan-free CIF copies from preprocessed PDB structures listed in a CSV.

Only CLI options:
  --gt-structures-directory
  --output-directory
  --csv-file
"""

import multiprocessing as mp
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm import tqdm

import biotite.structure.io as strucio
from openfold3.core.data.io.structure.cif import parse_target_structure


def _process_pdb_id(args):
    """Worker function to handle one ID."""
    pred_pdb_id, gt_structures_directory, output_directory = args
    try:
        entry_directory = output_directory / pred_pdb_id

        atom_array = parse_target_structure(gt_structures_directory, pred_pdb_id, "npz")

        # set all charges to zero
        atom_array.charge = np.zeros(len(atom_array), dtype=int)

        # filter occupancy != 0
        filtered_array = atom_array[atom_array.occupancy != 0]
        # remove heteroatoms
        filtered_array = filtered_array[~filtered_array.hetero]
        # keep only protein atoms (molecule_type_id == 0)
        filtered_array = filtered_array[filtered_array.molecule_type_id == 0]

        # save output
        outfile = entry_directory / f"{pred_pdb_id}.cif"
        strucio.save_structure(outfile, filtered_array)

        return pred_pdb_id
    except Exception as e:
        # Keep behavior similar to your original script
        print(f"Error processing {pred_pdb_id}: {e}")
        return None


@click.command(context_settings={"show_default": True})
@click.option(
    "--gt-structures-directory",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
)
@click.option(
    "--output-directory",
    type=click.Path(path_type=Path, file_okay=False),
    required=True,
)
@click.option(
    "--csv-file",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--num-workers",
    type=int,
    default=150,
    help="Number of worker processes for parallel processing.",
)
def main(
    gt_structures_directory: Path,
    output_directory: Path,
    csv_file: Path,
    num_workers: int,
) -> None:
    df = pd.read_csv(csv_file)
    pred_pdb_ids = df["entry_id"].tolist()

    # Create output directories
    for pred_pdb_id in tqdm(
        pred_pdb_ids, desc="Creating output directories", total=len(pred_pdb_ids)
    ):
        (output_directory / pred_pdb_id).mkdir(parents=True, exist_ok=True)

    chunksize = 4

    # Build arg tuples so the worker stays picklable + avoids globals
    work_items = [
        (pred_pdb_id, gt_structures_directory, output_directory)
        for pred_pdb_id in pred_pdb_ids
    ]

    with mp.Pool(num_workers) as pool:
        for _ in tqdm(
            pool.imap_unordered(_process_pdb_id, work_items, chunksize=chunksize),
            total=len(pred_pdb_ids),
            desc="Creating nan-free PDB copies",
        ):
            pass

    print("Done.")


if __name__ == "__main__":
    main()
