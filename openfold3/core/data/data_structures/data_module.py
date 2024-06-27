# TODO add license

import warnings
from functools import partial
from typing import Dict, List, Sequence

import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from openfold3.core.data.data_structures.single_datasets import (
    DATASET_REGISTRY,
    OpenFoldSingleDataset,
)
from openfold3.core.data.data_structures.stochastic_sampler_dataset import (
    OpenFoldStochasticSamplerDataset,
)
from openfold3.core.utils.tensor_utils import dict_multimap


class OpenFoldDataModule(pl.LightningDataModule):
    def __init__(self, data_config: List[Sequence[Dict]]) -> None:
        super().__init__()

        # TODO Input argument self-assignment - only assign necessary attributes if any

        # TODO Argument checks
        # train/validation/test/predict datasets exclusivity

        # Parse data_config
        dataset_classes, dataset_weights, dataset_configs, dataset_types = (
            self.parse_data_config(data_config)
        )

        # Initialize datasets
        # QUESTION do we want to support validation/testing/prediction on multiple datasets?
        if ("train" in dataset_types) | ("validation" in dataset_types):
            # Initialize train datasets
            train_datasets = self.init_datasets(
                dataset_classes, dataset_configs, dataset_types, "train"
            )

            generator = torch.Generator().manual_seed(
                "<data_seed>"
            )  # TODO add argument

            # Wrap train datasets in the sampler dataset class
            self.train_dataset = OpenFoldStochasticSamplerDataset(
                datasets=train_datasets,
                probabilities=dataset_weights,
                virtual_epoch_len="<virtual_epoch_len from data config>",  # TODO add argument
                num_virtual_epochs="<num_virtual_epochs from data config>",  # TODO add argument
                generator=generator,
            )

            # Currently only one validation dataset is supported
            self.validation_dataset = self.init_datasets(
                dataset_classes, dataset_configs, dataset_types, "validation"
            )[0]

        elif "test" in dataset_types:
            # Currently only one test dataset is supported
            self.test_dataset = self.init_datasets(
                dataset_classes, dataset_configs, dataset_types, "test"
            )[0]

        elif "predict" in dataset_types:
            # Currently only one predict dataset is supported
            self.predict_dataset = self.init_datasets(
                dataset_classes, dataset_configs, dataset_types, "predict"
            )[0]

        else:
            raise ValueError(
                f"No valid dataset types were found in data_config. Found: {dataset_types}"
            )

    def parse_data_config(
        self, data_config: list[Sequence[dict]]
    ) -> tuple[list, list, list, list]:
        """Parses input data_config into separate lists.

        Args:
            data_config (List[Sequence[Dict]]): Input data configuration list of dataset dictionaries.

        Returns:
            Tuple[List, List, List, Set]: Lists of dataset classes, weights, configurations and unique set of types.
        """
        dataset_classes, dataset_weights, dataset_configs, dataset_types = list(
            zip(
                *[
                    (
                        dataset_entry["dataset_class"],
                        dataset_entry["weight"],
                        dataset_entry["config"],
                        dataset_entry["type"],
                    )
                    for dataset_entry in data_config
                ]
            )
        )

        return dataset_classes, dataset_weights, dataset_configs, dataset_types

    def init_datasets(
        self,
        dataset_classes: list[Sequence[str]],
        dataset_configs: list[Sequence[dict]],
        dataset_types: list[Sequence[str]],
        type_to_init: str,
    ) -> list[Sequence[OpenFoldSingleDataset]]:
        """Initializes datasets.

        Args:
            dataset_classes (list[Sequence[str]]): List of strings matching the specific OpenFoldSingleDataset classes to initialize.
            dataset_configs (list[Sequence[dict]]): List of configs to pass each dataset class.
            dataset_types (list[Sequence[str]]): List of dataset types, elements can be train, validation, test, predict.
            type_to_init (str): One of train, validation, test, predict.

        Returns:
            list[Sequence[OpenFoldSingleDataset]]: List of initialized OpenFoldSingleDataset objects.
        """
        datasets = [
            DATASET_REGISTRY[dataset_class](dataset_config)
            for dataset_class, dataset_config, dataset_type in zip(
                dataset_classes, dataset_configs, dataset_types
            )
            if dataset_type == type_to_init
        ]
        if (type_to_init in ["validation", "test", "predict"]) & (len(datasets) > 1):
            warnings.warn(
                f"{len(datasets)} {type_to_init} datasets were found, using only the first one."
            )
        return datasets

    def train_dataloader(self) -> DataLoader:
        # TODO refactor OpneFoldDataLoader and add arguments
        return DataLoader(self.train_dataset, "<other arguments>")

    def val_dataloader(self) -> DataLoader:
        # TODO refactor OpneFoldDataLoader and add arguments
        return DataLoader(self.validation_dataset, "<other arguments>")

    def test_dataloader(self) -> DataLoader:
        # TODO refactor OpneFoldDataLoader and add arguments
        return DataLoader(self.test_dataset, "<other arguments>")

    def predict_dataloader(self) -> DataLoader:
        # TODO refactor OpneFoldDataLoader and add arguments
        return DataLoader(self.predict_dataset, "<other arguments>")


def openfold_batch_collator(prots):
    stack_fn = partial(torch.stack, dim=0)
    return dict_multimap(stack_fn, prots)
