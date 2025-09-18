"""A module for containing writing tools and callbacks for model outputs."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from biotite import structure
from pytorch_lightning.callbacks import BasePredictionWriter

from openfold3.core.data.io.structure.cif import write_structure

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    r"""Custom JSON encoder for handling numpy data types.

    https://gist.github.com/jonathanlurie/1b8d12f938b400e54c1ed8de21269b65
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def _take_batch_dim(x, b: int):
    if isinstance(x, torch.Tensor):
        if len(x.shape) > 1:
            return x[b].cpu().float().numpy()
        else:
            return x
    if isinstance(x, dict):
        return {k: _take_batch_dim(v, b) for k, v in x.items()}
    return x


def _take_sample_dim(x, s: int):
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.shape[0] == 1:
            return x[0]
        return x[s]
    if isinstance(x, dict):
        return {k: _take_sample_dim(v, s) for k, v in x.items()}
    return x


class OF3OutputWriter(BasePredictionWriter):
    """Callback for writing AF3 predicted structure and confidence outputs"""

    def __init__(
        self,
        output_dir: Path,
        pae_enabled: bool = False,
        structure_format: str = "pdb",
        full_confidence_output_format: str = "json",
    ):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.pae_enabled = pae_enabled
        self.structure_format = structure_format
        self.full_confidence_format = full_confidence_output_format

    @staticmethod
    def write_structure_prediction(
        atom_array: structure.AtomArray,
        predicted_coords: np.ndarray,
        plddt: np.ndarray,
        output_file: Path,
    ):
        """Writes predicted coordinates to atom_array and writes mmcif file to disk.

        pLDDT scores are written to the B-factor column of the output file.
        """

        # Set coordinates and plddt scores
        atom_array.coord = predicted_coords
        atom_array.set_annotation("b_factor", plddt)

        # Write the output file
        logger.info(f"Writing predicted structure to {output_file}")
        write_structure(atom_array, output_file, include_bonds=True)

    def get_pae_confidence_scores(self, confidence_scores):
        pae_confidence_scores = {}
        single_value_keys = [
            "iptm",
            "ptm",
            "disorder",
            "has_clash",
            "sample_ranking_score",
        ]

        for key in single_value_keys:
            pae_confidence_scores[key] = confidence_scores[key]

        pae_confidence_scores["ptm_by_asym_id"] = confidence_scores["pTM_by_asym_id"]
        pae_confidence_scores["iptm_by_asym_id_pair"] = {
            str(k): v for k, v in confidence_scores["all_ipTM_scores"]["iptm"].items()
        }
        pae_confidence_scores["bespoke_iptm_by_asym_id_pair"] = {
            str(k): v
            for k, v in confidence_scores["all_ipTM_scores"]["bespoke_iptm"].items()
        }
        return pae_confidence_scores

    def write_confidence_scores(
        self, confidence_scores: dict[str, np.ndarray], output_prefix: Path
    ):
        """Writes confidence scores to disk"""
        plddt = confidence_scores["plddt"]
        pde = confidence_scores["predicted_distance_error"]
        gpde = confidence_scores["global_predicted_distance_error"]
        aggregated_confidence_scores = {"avg_plddt": np.mean(plddt), "gpde": gpde}

        if self.pae_enabled:
            logger.info("Recording PAE confidence outputs")
            aggregated_confidence_scores |= self.get_pae_confidence_scores(
                confidence_scores
            )

        out_file_agg = Path(f"{output_prefix}_confidences_aggregated.json")
        out_file_agg.write_text(
            json.dumps(aggregated_confidence_scores, indent=4, cls=NumpyEncoder)
        )

        # Full confidence scores
        full_confidence_scores = {"plddt": plddt, "pde": pde}
        out_fmt = self.full_confidence_format
        out_file_full = Path(f"{output_prefix}_confidences.{out_fmt}")

        if out_fmt == "json":
            out_file_full.write_text(
                json.dumps(
                    full_confidence_scores,
                    indent=4,
                    cls=NumpyEncoder,
                )
            )
        elif out_fmt == "npz":
            np.savez_compressed(out_file_full, **full_confidence_scores)

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        is_repeated_sample = batch.get("repeated_sample")
        if outputs is None or is_repeated_sample:
            return

        batch, outputs = outputs
        confidence_scores = outputs["confidence_scores"]

        batch_size = len(batch["atom_array"])
        sample_size = outputs["atom_positions_predicted"].shape[1]

        # Iterate over all predictions in the batch
        for b in range(batch_size):
            seed = batch["seed"][b]
            query_id = batch["query_id"][b]

            output_subdir = Path(self.output_dir) / query_id / f"seed_{seed}"

            # Extract attributes for the current batch
            atom_array_batch = batch["atom_array"][b]
            predicted_coords_batch = (
                outputs["atom_positions_predicted"][b].cpu().float().numpy()
            )
            confidence_scores_batch = _take_batch_dim(confidence_scores, b)

            # Iterate over all diffusion samples
            for s in range(sample_size):
                file_prefix = output_subdir / f"{query_id}_seed_{seed}_sample_{s + 1}"
                file_prefix.parent.mkdir(parents=True, exist_ok=True)

                confidence_scores_sample = _take_sample_dim(confidence_scores_batch, s)
                predicted_coords_sample = predicted_coords_batch[s]

                # Save predicted structure
                structure_file = Path(f"{file_prefix}_model.{self.structure_format}")
                self.write_structure_prediction(
                    atom_array=atom_array_batch,
                    predicted_coords=predicted_coords_sample,
                    plddt=confidence_scores_sample["plddt"],
                    output_file=structure_file,
                )

                # Save confidence metrics
                self.write_confidence_scores(
                    confidence_scores=confidence_scores_sample,
                    output_prefix=file_prefix,
                )

        del batch
        del outputs
