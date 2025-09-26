"""A module for containing writing tools and callbacks for model outputs."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from biotite import structure
from pytorch_lightning.callbacks import BasePredictionWriter
from torchmetrics.aggregation import SumMetric

from openfold3.core.data.io.structure.cif import write_structure
from openfold3.core.utils.tensor_utils import tensor_tree_map

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


class OF3OutputWriter(BasePredictionWriter):
    """Callback for writing AF3 predicted structure and confidence outputs"""

    def __init__(
        self,
        output_dir: Path,
        structure_format: str = "pdb",
        full_confidence_output_format: str = "json",
        write_features: bool = False,
        write_latent_outputs: bool = False,
    ):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.structure_format = structure_format
        self.full_confidence_format = full_confidence_output_format
        self.write_features = write_features
        self.write_latent_outputs = write_latent_outputs

        # Track successfully predicted samples
        self.success_count = SumMetric()
        self.failed_count = SumMetric()
        self.total_count = SumMetric()
        self.failed_queries = []

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

    def write_confidence_scores(
        self, confidence_scores: dict[str, np.ndarray], output_prefix: Path
    ):
        """Writes confidence scores to disk"""
        plddt = confidence_scores["plddt"]
        pde = confidence_scores["predicted_distance_error"]
        gpde = confidence_scores["global_predicted_distance_error"]

        # Single-valued aggregated confidence scores
        aggregated_confidence_scores = {
            "avg_plddt": np.mean(plddt),
            "gpde": gpde,
        }
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

    def write_all_outputs(self, batch: dict, outputs: dict, confidence_scores: dict):
        """Writes all outputs for a given batch."""

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
            confidence_scores_batch = {
                key: value[b].cpu().float().numpy() if len(value.shape) > 1 else value
                for key, value in confidence_scores.items()
            }

            # Iterate over all diffusion samples
            for s in range(sample_size):
                file_prefix = output_subdir / f"{query_id}_seed_{seed}_sample_{s + 1}"
                file_prefix.parent.mkdir(parents=True, exist_ok=True)

                confidence_scores_sample = {}
                for key, value in confidence_scores_batch.items():
                    if len(value.shape) < 1:
                        confidence_scores_sample[key] = value.item()
                    elif value.shape[0] == 1:
                        confidence_scores_sample[key] = value[0]
                    else:
                        confidence_scores_sample[key] = value[s]

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

            def fetch_cur_batch(t):
                # Get tensor for current batch dim
                # Remove expanded sample dim if it exists to get original tensor shapes
                if t.ndim < 2:
                    return t

                cur_feats = t[b : b + 1].squeeze(1)  # noqa: B023
                return cur_feats.detach().clone().cpu()

            file_prefix = output_subdir / f"{query_id}_seed_{seed}"
            if self.write_features:
                out_file = Path(f"{file_prefix}_batch.pt")
                cur_batch = tensor_tree_map(fetch_cur_batch, batch, strict_type=False)
                torch.save(cur_batch, out_file)
                del cur_batch

            if self.write_latent_outputs:
                out_file = Path(f"{file_prefix}_latent_output.pt")
                cur_output = tensor_tree_map(
                    fetch_cur_batch, outputs, strict_type=False
                )
                torch.save(cur_output, out_file)
                del cur_output

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):
        # Move metrics to device
        if self.total_count.device != pl_module.device:
            self.total_count.to(pl_module.device)
            self.success_count.to(pl_module.device)
            self.failed_count.to(pl_module.device)

        # Skip repeated samples
        if batch.get("repeated_sample"):
            return

        self.total_count.update(1)

        # Skip and track failed samples
        if outputs is None:
            self.failed_count.update(1)
            self.failed_queries.extend(batch["query_id"])
            return

        batch, outputs = outputs
        confidence_scores = outputs["confidence_scores"]

        # Write predictions and confidence scores
        # Optionally write out input features and latent outputs
        try:
            self.write_all_outputs(
                batch=batch, outputs=outputs, confidence_scores=confidence_scores
            )
            self.success_count.update(1)
        except Exception as e:
            self.failed_count.update(1)
            self.failed_queries.extend(batch["query_id"])
            logger.exception(
                f"Failed to write predictions for query_id(s) "
                f"{', '.join(batch['query_id'])}: {e}"
            )

        del batch, outputs

    def on_predict_end(self, trainer, pl_module):
        """Print summary of inference run."""

        # Gather failed queries from all processes
        if dist.is_available() and dist.is_initialized():
            gathered_lists = [None] * trainer.world_size
            dist.all_gather_object(gathered_lists, self.failed_queries)
        else:
            gathered_lists = [self.failed_queries]

        # Compute the final counts, synced in compute()
        total_queries = self.total_count.compute().int().item()
        success_count = self.success_count.compute().int().item()
        failed_count = self.failed_count.compute().int().item()

        if trainer.is_global_zero:
            # Flatten the list of failed query lists from all processes
            final_failed_list = [item for sublist in gathered_lists for item in sublist]

            summary = [
                "\n" + "=" * 50,
                "    PREDICTION SUMMARY    ",
                "=" * 50,
                f"Total Queries Processed: {total_queries}",
                f"  - Successful Queries:  {success_count}",
                f"  - Failed Queries:      {failed_count}",
            ]

            if final_failed_list:
                failed_str = ", ".join(sorted(list(set(final_failed_list))))
                summary.append(f"\nFailed Queries: {failed_str}")

            summary.append("=" * 50 + "\n")
            print("\n".join(summary))
