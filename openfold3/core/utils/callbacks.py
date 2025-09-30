import time

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class PredictTimer(pl.Callback):
    @rank_zero_only
    def on_predict_start(self, trainer, pl_module):
        self.start_time = time.time()

    @rank_zero_only
    def on_predict_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        print(f"Inference runtime: {elapsed} seconds")
