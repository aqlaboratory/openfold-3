import time
from pathlib import Path

import pytorch_lightning as pl
from lightning_fabric.utilities.rank_zero import (
    rank_zero_only,
)


class PredictTimer(pl.Callback):
    def on_predict_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_predict_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        print(f"Inference runtime: {elapsed} seconds")


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
