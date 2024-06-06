# TODO add license

from functools import partial
import json
from typing import Optional, Dict, List, Set, Tuple, Sequence

import ml_collections as mlc
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from openfold3.base.utils.tensor_utils import dict_multimap
from openfold3.base.data.data_structures.singledatasets import DATASET_REGISTRY
from openfold3.base.data.data_structures.stochasticsamplerdataset import (
    OpenFoldStochasticSamplerDataset,
)


class OpenFoldDataModule(pl.LightningDataModule):
    def __init__(self, data_config: List[Sequence[Dict]]) -> None:
        super().__init__()

        # TODO Input argument self-assignment - only assign necessary attributes
        
        # TODO Argument checks

        # Parse data_config
        dataset_classes, dataset_weights, dataset_configs, dataset_types = (
            self.parse_data_config(data_config)
        )

        # Initialize datasets 
        # QUESTION do we want to support validation/testing/prediction on multiple datasets?
        if ("train" in dataset_types) | ("valid" in dataset_types):
            train_datasets = [
                DATASET_REGISTRY[dataset_class](dataset_config)
                for dataset_class, dataset_config in zip(dataset_classes, dataset_configs)
            ]

            generator = torch.Generator().manual_seed(
                "<data_seed>"
            )  # TODO add argument

            self.train_dataset = OpenFoldStochasticSamplerDataset(
                datasets=train_datasets,
                probabilities=dataset_weights,
                virtual_epoch_len="<virtual_epoch_len from data config>",  # TODO add argument
                num_virtual_epochs="<num_virtual_epochs from data config>",  # TODO add argument
                generator=generator,
            )

            self.valid_dataset = "<SingleDataset>"

        elif ("test" in dataset_types):
            self.test_dataset = "<SingleDataset>"

        elif ("predict" in dataset_types):
            self.predict_dataset = "<SingleDataset>"
        else:
            raise ValueError(f"No valid dataset types were found in data_config. Found: {dataset_types}")

    def parse_data_config(
        self, data_config: List[Sequence[Dict]]
    ) -> Tuple[List, List, List, Set]:
        """Parses input data_config into separate lists.

        Args:
            data_config (List[Sequence[Dict]]): Input data configuration list of dataset dictionaries.

        Returns:
            Tuple[List, List, List, Set]: Lists of dataset classes, weights, configurations and unique set of types.
        """
        dataset_classes, dataset_weights, dataset_configs = list(
            zip(
                *[
                    (
                        dataset_entry["dataset_class"],
                        dataset_entry["weight"],
                        dataset_entry["config"],
                    )
                    for dataset_entry in data_config
                ]
            )
        )

        dataset_types = {[dataset_entry["type"] for dataset_entry in data_config]}

        return dataset_classes, dataset_weights, dataset_configs, dataset_types

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
