import time

import pytorch_lightning as pl
from torch.cuda import profiler as profiler


class PredictTimer(pl.Callback):
    def on_predict_start(self, trainer, pl_module):
        self.start_time = time.time()

    def on_predict_end(self, trainer, pl_module):
        elapsed = time.time() - self.start_time
        print(f"Inference runtime: {elapsed} seconds")
