import json
import time
from pathlib import Path

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.rank_zero import (
    rank_zero_only,
)


class PredictTimer(pl.Callback):
    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = output_dir

        # For recording runtime per batch
        self.batch_start_time = None

    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0
    ):
        self.batch_start_time = time.perf_counter()

    def _get_runtime(self):
        """Record the runtime for the current batch."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_end_time = time.perf_counter()

        return batch_end_time - self.batch_start_time

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        # Get batch runtime
        runtime = self._get_runtime()

        # Skip repeated samples
        if batch.get("repeated_sample") or outputs is None:
            return

        batch_size = len(batch["atom_array"])

        # Calculate an average runtime for each sample in the batch
        # This is always one sample for now
        runtime_per_sample = runtime / batch_size

        # Iterate over all predictions in the batch
        for b in range(batch_size):
            seed = batch["seed"][b]
            query_id = batch["query_id"][b]

            output_subdir = Path(self.output_dir) / query_id / f"seed_{seed}"

            # Save runtime for the batch
            runtime_file = output_subdir / "timing.json"
            runtime_json = {"runtime_s": runtime_per_sample}
            runtime_file.write_text(json.dumps(runtime_json, indent=4))


class LogInferenceQuerySet(pl.Callback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    @rank_zero_only
    def on_predict_start(self, trainer, pl_module):
        log_path = self.output_dir / "inference_query_set.json"
        with open(log_path, "w") as fp:
            fp.write(
                pl_module.trainer.datamodule.inference_config.query_set.model_dump_json(
                    indent=4
                )
            )
