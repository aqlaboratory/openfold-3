"""
A module for containing writing tools and callbacks for model outputs.
TODO: This will be refactored in the inference pipeline.
"""

import logging
from pathlib import Path

import click
import numpy as np
import torch
from biotite import structure

from openfold3.core.data.io.structure.cif import write_structure


def write_structure_prediction(
    atom_array: structure.AtomArray,
    predicted_coords: np.ndarray,
    pdb_id: str,
    output_dir: Path,
    status: str,
    atom_unresolved_mask: np.ndarray = None,
):
    """Writes predicted coordinates to atom_array and writes mmcif file to disk"""

    for sample in range(predicted_coords.shape[0]):
        output_path = output_dir / f"{pdb_id}_{status}_sample_{sample + 1}.cif"

        # Overwrite coordinates in atom_array
        atom_array.coord = predicted_coords[sample]

        logging.warning(f"Writing predicted structure for {pdb_id} to {output_path}")

        if atom_unresolved_mask is not None:
            atom_array.occupancy = atom_unresolved_mask[sample]
            atom_array_only_resolved = atom_array[atom_array.occupancy != 0]
            write_structure(atom_array_only_resolved, output_path, include_bonds=True)
        else:
            write_structure(atom_array, output_path, include_bonds=True)


def write_gt_pred_structures(batch: dict, outputs: dict, pdb_id_path: Path):
    pdb_id = batch["pdb_id"][0]
    atom_array = batch["atom_array"][0]

    predicted_coords = outputs["atom_positions_predicted"][0].numpy()

    write_structure_prediction(
        atom_array=atom_array,
        predicted_coords=predicted_coords,
        pdb_id=pdb_id,
        output_dir=pdb_id_path,
        status="predicted",
    )
    write_structure_prediction(
        atom_array=atom_array,
        predicted_coords=batch["ground_truth"]["atom_positions"][0].numpy(),
        pdb_id=pdb_id,
        output_dir=pdb_id_path,
        status="ground_truth",
        atom_unresolved_mask=batch["ground_truth"]["atom_resolved_mask"][0].numpy(),
    )


@click.command()
@click.option(
    "--validation_output_dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory with validation outputs",
)
def main(validation_output_dir: Path) -> None:
    output_path = validation_output_dir / "output"

    for pdb_id_path in output_path.iterdir():
        batch = torch.load(pdb_id_path / "batch.pt")
        outputs = torch.load(pdb_id_path / "outputs.pt")

        write_gt_pred_structures(batch=batch, outputs=outputs, pdb_id_path=pdb_id_path)


if __name__ == "__main__":
    main()
