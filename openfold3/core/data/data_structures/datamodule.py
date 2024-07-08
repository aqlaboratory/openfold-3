# TODO add license

import warnings
from functools import partial
from typing import Dict, List, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from openfold3.core.data.data_structures.singledatasets import (
    DATASET_REGISTRY,
    OpenFoldSingleDataset,
)
from openfold3.core.data.data_structures.stochasticsamplerdataset import (
    OpenFoldStochasticSamplerDataset,
)
from openfold3.core.utils.tensor_utils import dict_multimap


class OpenFoldDataModule(pl.LightningDataModule):
    def __init__(self, data_config: List[Sequence[Dict]]) -> None:
        super().__init__()

        # Input argument self-assignment
        self.batch_size = data_config["batch_size"]
        self.num_workers = data_config["num_workers"]
        self.data_seed = data_config["data_seed"]
        self.virtual_epoch_len = data_config["virtual_epoch_len"]
        self.num_virtual_epochs = data_config["num_virtual_epochs"]

        # Parse data_config
        dataset_classes, dataset_weights, dataset_configs, dataset_types = (
            self.parse_data_config(data_config)
        )

        # Initialize datasets
        # QUESTION do we want to support validation/testing/prediction on 
        # multiple datasets?
        if ("train" in dataset_types) | ("valid" in dataset_types):
            # Initialize train datasets
            train_datasets = self.init_datasets(
                dataset_classes, dataset_configs, dataset_types, "train"
            )

            self.generator = torch.Generator(device="cpu").manual_seed(self.data_seed)

            # Wrap train datasets in the sampler dataset class
            self.train_dataset = OpenFoldStochasticSamplerDataset(
                datasets=train_datasets,
                probabilities=dataset_weights,
                virtual_epoch_len=self.virtual_epoch_len,
                num_virtual_epochs=self.num_virtual_epochs,
                generator=self.generator,
            )

            # Currently only one valid dataset is supported
            self.valid_dataset = self.init_datasets(
                dataset_classes, dataset_configs, dataset_types, "valid"
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
                f"""No valid dataset types were found in data_config. Found: \
                 {dataset_types}"""
            )

    def parse_data_config(
        self, data_config: list[Sequence[dict]]
    ) -> tuple[list, list, list, list]:
        """Parses input data_config into separate lists.

        Args:
            data_config (List[Sequence[Dict]]): 
                Input data configuration list of dataset dictionaries.

        Returns:
            Tuple[List, List, List, Set]: Lists of dataset classes, weights,
            configurations and unique set of types.
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
        dataset_types_unique = set(dataset_types)
        if (len(dataset_types_unique) == 2) & (
            ("train" not in dataset_types) | ("valid" not in dataset_types)
        ):
            raise ValueError(
                f"""An unsupported combination of dataset types was found in \
                 data_config: {dataset_types_unique}. The supported dataset \
                 combinations are: ['train'], ['train', 'valid'], ['test'], \
                 ['predict']."""
            )
        elif (len(dataset_types_unique) == 1) & ("valid" in dataset_types):
            raise ValueError(
                """Validation dataset(s) were provided without any training datasets. \
                The supported dataset combinations are: ['train'], ['train', 'valid'], \
                ['test'], ['predict']."""
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
            dataset_classes (list[Sequence[str]]): 
                List of strings matching the specific OpenFoldSingleDataset classes to
                initialize.
            dataset_configs (list[Sequence[dict]]): 
                List of configs to pass each dataset class.
            dataset_types (list[Sequence[str]]): 
                List of dataset types, elements can be train, valid, test, predict.
            type_to_init (str): 
                One of train, valid, test, predict.

        Returns:
            list[Sequence[OpenFoldSingleDataset]]: List of initialized
            OpenFoldSingleDataset objects.
        """
        datasets = [
            DATASET_REGISTRY[dataset_class](dataset_config)
            for dataset_class, dataset_config, dataset_type in zip(
                dataset_classes, dataset_configs, dataset_types
            )
            if dataset_type == type_to_init
        ]
        if (type_to_init in ["valid", "test", "predict"]) & (len(datasets) > 1):
            warnings.warn(
                f"""{len(datasets)} {type_to_init} datasets were found, using only the \
                 first one.""", 
                stacklevel=2
            )
        return datasets

    def generate_dataloader(self, stage: str):
        """Wrap the appropriate dataset in a DataLoader and return it.

        Args:
            stage (str): 
                Type of DataLoader to return, one of train, valid, test, predict.

        Returns:
            DataLoader: DataLoader object.
        """
        dataset = (
            self.train_dataset
            if stage == "train"
            else self.valid_dataset
            if stage == "valid"
            else self.test_dataset
            if stage == "test"
            else self.predict_dataset
        )

        # TODO refactor OpenFoldDataLoader and add arguments
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=openfold_batch_collator,
            generator=self.generator,
        )

    def train_dataloader(self) -> DataLoader:
        return self.generate_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.generate_dataloader("valid")

    def test_dataloader(self) -> DataLoader:
        return self.generate_dataloader("test")

    def predict_dataloader(self) -> DataLoader:
        return self.generate_dataloader("predict")


def openfold_batch_collator(prots):
    stack_fn = partial(torch.stack, dim=0)
    return dict_multimap(stack_fn, prots)
