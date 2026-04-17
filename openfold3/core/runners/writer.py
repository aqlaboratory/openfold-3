# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A module for containing writing tools and callbacks for model outputs."""

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import re
import torch.distributed as dist
from biotite import structure
from pytorch_lightning.callbacks import BasePredictionWriter

from openfold3.core.data.io.structure.cif import write_structure
from openfold3.core.utils.tensor_utils import tensor_tree_map

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    r"""Custom JSON encoder for handling numpy data types.

    https://gist.github.com/jonathanlurie/1b8d12f938b400e54c1ed8de21269b65
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float16:
                return obj.astype(np.float64).round(3).tolist()
            elif obj.dtype == np.float32:
                return obj.astype(np.float64).round(6).tolist()
            return obj.tolist()
        elif isinstance(obj, np.generic):
            if obj.dtype == np.float16:
                return round(obj.item(), 3)
            elif obj.dtype == np.float32:
                return round(obj.item(), 6)
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
        structure_format: str = "pdb",
        full_confidence_output_format: str = "json",
        full_confidence_output_dtype: Literal["float32", "float16"] = "float16",
        write_features: bool = False,
        write_latent_outputs: bool = False,
        write_full_confidence_scores: bool = True,
    ):
        super().__init__(write_interval="batch")
        self.output_dir = output_dir
        self.structure_format = structure_format
        self.full_confidence_format = full_confidence_output_format
        self.full_confidence_dtype = np.dtype(full_confidence_output_dtype)
        self.write_features = write_features
        self.write_latent_outputs = write_latent_outputs
        self.write_full_confidence_scores = write_full_confidence_scores

        # Track successfully predicted samples
        self.success_count = 0
        self.failed_count = 0
        self.total_count = 0
        self.failed_queries = []

    @staticmethod
    def write_structure_prediction(
        atom_array: structure.AtomArray,
        predicted_coords: np.ndarray,
        plddt: np.ndarray,
        output_file: Path,
        make_ost_compatible: bool = True,
    ):
        """Writes predicted coordinates to atom_array and writes mmcif file to disk.

        pLDDT scores are written to the B-factor column of the output file.
        """

        # Set coordinates and plddt scores
        atom_array.coord = predicted_coords
        atom_array.set_annotation("b_factor", plddt)

        # Write the output file
        logger.info(f"Writing predicted structure to {output_file}")
        write_structure(
            atom_array,
            output_file,
            include_bonds=True,
            make_ost_compatible=make_ost_compatible,
        )

    @staticmethod
    def _renumber_atom_array(
        atom_array: structure.AtomArray,
        custom_residue_ids: list[str]
    ) -> structure.AtomArray:
        """
        Applies custom residue numbering to the Biotite AtomArray using the 
        list of string IDs provided by the user (via the feature dict).

        It handles the splitting of complex PDB IDs (e.g., '102A' -> res_id=102, 
        ins_code='A') and maps them from residue-level to atom-level annotations.
        """
        
        # Get indices where each residue starts in the AtomArray (e.g., [0, 2, 4, 6, 8]).
        residue_start_indices = structure.get_residue_starts(atom_array)
        
        # Initialize new annotation arrays for all atoms.
        arr_len = len(atom_array)
        
        # IMPORTANT: Initializing with zeros is acceptable, as all elements 
        # must be overwritten by the loop for a successful renumbering.
        new_res_id = np.zeros(arr_len, dtype=np.int32) 
        new_ins_code = np.full(arr_len, '', dtype=object) 

        # Pre-calculate end indices for slicing. The end index of residue N is
        # the start index of residue N+1, or the total array length for the last residue.
        # Example: [0, 2, 4, 6, 8] -> [2, 4, 6, 8, 10]
        end_indices = np.append(
            residue_start_indices[1:], 
            arr_len                    
        )
        
        # Iterate over all residue start indices (i is the residue index, not atom index).
        for i in range(len(residue_start_indices)):
            
            if i >= len(custom_residue_ids):
                # Safety break: Should be prevented by featurizer
                break
            
            # Start and End indices for the atom array slice corresponding to this residue.
            start_index = residue_start_indices[i] 
            end_index = end_indices[i]             
            
            res_id_str = custom_residue_ids[i]
            
            # Parse the string ID to separate number and insertion code.
            # (\d+) captures digits (res_id), ([A-Z]?) captures optional char (ins_code)
            match = re.match(r"(\d+)([A-Z]?)", res_id_str)
            if match:
                num = int(match.group(1))
                code = match.group(2) if match.group(2) else ''
            else:
                num, code = 0, ''
            
            # Apply the new res_id and ins_code to all atoms in this residue range
            new_res_id[start_index:end_index] = num
            new_ins_code[start_index:end_index] = code
            
        # Overwrite the AtomArray annotations.
        atom_array.set_annotation("res_id", new_res_id)
        atom_array.set_annotation("ins_code", new_ins_code)
        
        return atom_array
    
    def get_pae_confidence_scores(self, confidence_scores, atom_array):
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

        # Get map from asym id to chain id
        renum_ids = np.unique(atom_array.chain_id, return_inverse=True)[1] + 1
        asym_id_to_chain_id = {
            k: v
            for (k, v) in set(
                [
                    (int(x[0]), str(x[1]))
                    for x in zip(renum_ids, atom_array.chain_id, strict=True)
                ]
            )
        }

        # Asym id -> chain id for chain_ptm
        pae_confidence_scores["chain_ptm"] = {
            asym_id_to_chain_id[int(k)]: v
            for k, v in confidence_scores["chain_ptm"].items()
        }

        # Asym id -> chain id for chain_pair_iptm
        pae_confidence_scores["chain_pair_iptm"] = {}
        for k, v in confidence_scores["chain_pair_iptm"].items():
            # split '(1, 2)' into 1, 2
            k1, k2 = [
                asym_id_to_chain_id[int(i)].strip() for i in k.strip("()").split(",")
            ]
            pae_confidence_scores["chain_pair_iptm"][f"({k1}, {k2})"] = v

        # Asym id -> chain id for bespoke_iptm
        pae_confidence_scores["bespoke_iptm"] = {}
        for k, v in confidence_scores["bespoke_iptm"].items():
            # split '(1, 2)' into 1, 2
            k1, k2 = [
                asym_id_to_chain_id[int(i)].strip() for i in k.strip("()").split(",")
            ]
            pae_confidence_scores["bespoke_iptm"][f"({k1}, {k2})"] = v
        return pae_confidence_scores

    def write_confidence_scores(
        self,
        confidence_scores: dict[str, np.ndarray],
        atom_array: structure.AtomArray,
        output_prefix: Path,
    ):
        """Writes confidence scores to disk"""
        plddt = confidence_scores["plddt"]
        pde = confidence_scores["pde"]
        gpde = confidence_scores["gpde"]
        pae = confidence_scores["pae"]
        aggregated_confidence_scores = {"avg_plddt": np.mean(plddt), "gpde": gpde}

        logger.info("Recording PAE confidence outputs")
        aggregated_confidence_scores |= self.get_pae_confidence_scores(
            confidence_scores, atom_array
        )

        out_file_agg = Path(f"{output_prefix}_confidences_aggregated.json")
        out_file_agg.write_text(
            json.dumps(aggregated_confidence_scores, indent=4, cls=NumpyEncoder)
        )

        # Full confidence scores
        if self.write_full_confidence_scores is True:
            full_confidence_scores = {"plddt": plddt, "pde": pde, "pae": pae}
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
                for key, val in full_confidence_scores.items():
                    if (
                        isinstance(val, np.ndarray)
                        and val.dtype != self.full_confidence_dtype
                    ):
                        full_confidence_scores[key] = val.astype(
                            self.full_confidence_dtype
                        )
                np.savez_compressed(
                    out_file_full, **full_confidence_scores, allow_pickle=False
                )

    def write_all_outputs(self, batch: dict, outputs: dict, confidence_scores: dict):
        """Writes all outputs for a given batch."""

        batch_size = len(batch["atom_array"])
        sample_size = outputs["atom_positions_predicted"].shape[1]

        # Get custom residue IDs for the entire batch
        custom_residue_ids_all = batch.get("custom_residue_ids", None)

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

            # Get the custom IDs list for the current batch entry (list[str] or None)
            custom_residue_ids_sample = (
                custom_residue_ids_all[b]
                if custom_residue_ids_all is not None and custom_residue_ids_all[b] is not None
                else None
            )

            # Apply custom residue numbering as a post-processing step
            if custom_residue_ids_sample is not None:
                # Modify AtomArray with custom residue numbers before writing.
                atom_array_batch = self._renumber_atom_array(
                    atom_array=atom_array_batch,
                    custom_residue_ids=custom_residue_ids_sample
                )

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
                    atom_array=atom_array_batch,
                )

            def fetch_cur_batch(t):
                # Get tensor for current batch dim
                # Remove expanded sample dim if it exists to get original tensor shapes
                if t.ndim < 2:
                    return t

                cur_feats = t[b : b + 1].squeeze(1)  # noqa: B023
                return cur_feats.detach().clone().cpu()

            file_prefix = output_subdir / f"{query_id}_seed_{seed}"

            # Write out input feature dictionary
            if self.write_features:
                out_file = Path(f"{file_prefix}_batch.pt")
                cur_batch = tensor_tree_map(fetch_cur_batch, batch, strict_type=False)
                torch.save(cur_batch, out_file)
                del cur_batch

            # Write out latent reps / raw model outputs
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
        dataloader_idx=0,
    ):
        # Skip repeated samples
        if batch.get("repeated_sample"):
            return

        self.total_count += 1

        # Skip and track failed samples
        if outputs is None:
            self.failed_count += 1
            self.failed_queries.extend(batch["query_id"])
            return

        batch, outputs = outputs
        confidence_scores = outputs["confidence_scores"]

        # Write predictions and confidence scores
        # Optionally write out input features and latent outputs
        try:
            self.write_all_outputs(
                batch=batch,
                outputs=outputs,
                confidence_scores=confidence_scores,
            )
            self.success_count += 1
        except Exception as e:
            self.failed_count += 1
            self.failed_queries.extend(batch["query_id"])
            logger.exception(
                f"Failed to write predictions for query_id(s) "
                f"{', '.join(batch['query_id'])}: {e}"
            )

        del batch, outputs

    def on_predict_end(self, trainer, pl_module):
        """
        Print summary of inference run. Includes a timeout failsafe
        for distributed runs.
        """

        try:
            # Gather summary data from all processes
            final_summary_data = {
                "total": self.total_count,
                "success": self.success_count,
                "failed": self.failed_count,
                "failed_queries": self.failed_queries,
            }

            gathered_data = [final_summary_data]
            if dist.is_available() and dist.is_initialized():
                gathered_data = [None] * trainer.world_size
                dist.all_gather_object(gathered_data, final_summary_data)

            if trainer.is_global_zero:
                # Aggregate the results from all processes on Rank 0
                total_queries = sum(data["total"] for data in gathered_data)
                success_count = sum(data["success"] for data in gathered_data)
                failed_count = sum(data["failed"] for data in gathered_data)
                final_failed_list = [
                    item for data in gathered_data for item in data["failed_queries"]
                ]

                self._write_summary(
                    total_queries=total_queries,
                    success_count=success_count,
                    failed_count=failed_count,
                    failed_list=final_failed_list,
                    global_rank=trainer.global_rank,
                    is_complete=True,
                )

        except RuntimeError as e:
            # TODO: Due to additional sync PL does outside of this callback,
            #  this won't be reached before the timeout error occurs.
            #  Leaving this here for now in case we refactor the prediction
            #  logic to avoid the extra syncs.
            error_str = str(e).lower()
            if "timeout" in error_str or "timed out" in error_str:
                logger.warning(
                    f"[Rank {trainer.global_rank}] Distributed sync timed out! "
                    f"Writing local results to a fallback log."
                )

                self._write_summary(
                    total_queries=self.total_count,
                    success_count=self.success_count,
                    failed_count=self.failed_count,
                    failed_list=self.failed_queries,
                    global_rank=trainer.global_rank,
                    is_complete=False,
                )

            else:
                # Re-raise unexpected runtime errors
                raise e

    def _write_summary(
        self,
        total_queries: int,
        success_count: int,
        failed_count: int,
        failed_list: list,
        global_rank: int = 0,
        is_complete=True,
    ):
        """Helper to format the final summary."""
        if is_complete:
            status = "COMPLETE"
            out_file = self.output_dir / "summary.txt"
        else:
            status = f"INCOMPLETE (Rank {global_rank})"
            out_file = self.output_dir / f"fallback_summary_rank_{global_rank}.txt"

        summary = [
            "\n" + "=" * 50,
            f"    PREDICTION SUMMARY ({status})    ",
            "=" * 50,
            f"Total Queries Processed: {total_queries}",
            f"  - Successful Queries:  {success_count}",
            f"  - Failed Queries:      {failed_count}",
        ]

        if failed_list:
            failed_str = ", ".join(sorted(list(set(failed_list))))
            summary.append(f"\nFailed Queries: {failed_str}")

        summary.append("=" * 50 + "\n")
        summary = "\n".join(summary)

        out_file.write_text(summary)

        if is_complete:
            print(summary)
        else:
            logger.warning(
                f"Fallback summary for Rank {global_rank} saved to: {out_file}"
            )
