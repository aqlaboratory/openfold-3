# TODO add license

from functools import partial
import json
from typing import Optional

import ml_collections as mlc
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from openfold3.base.utils.tensor_utils import dict_multimap


# TODO yet to refactor
class OpenFoldDataModule(pl.LightningDataModule):
    def __init__(self) -> None:
        super().__init__()

        # input argument self-assignment

        # argument checks - any way to move these earlier?

        # initialization of SingleDatasets - !!! how to handle train/val/test/predict logic?
        if self.stage == "train":
            self.train_datasets = "<List[SingleDataset]>"
        elif self.stage == "valid":
            self.valid_dataset = "<SingleDataset>"
        elif self.stage == "test":
            self.test_dataset = "<SingleDataset>"
        elif self.stage == "predict":
            self.predict_dataset = "<SingleDataset>"
        
        # Wrap training SingleDatasets into MultiDataset

    def train_dataloader(self) -> json.Any:
        return super().train_dataloader()
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return super().val_dataloader()
    
    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return super().test_dataloader()
    
    def predict_dataloader(self) -> TRAIN_DATALOADERS:
        return super().predict_dataloader()
    

class OpenFoldBatchCollator:
    def __call__(self, prots):
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots)

