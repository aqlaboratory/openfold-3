"""This module contains the DataModule class.

The DataModule is a LightningDataModule class that organizes the
instantiation of Datasets for training, validation, testing and prediction and
wraps Datasets into DataLoaders.

The steps below outline how datapoints get from raw datapoints to the model
and highlight where you currently are in the process:

0. Dataset filtering and cache generation
    raw data -> filtered data
1. PreprocessingPipeline
    filtered data -> preprocessed data
2. SampleProcessingPipeline and FeaturePipeline
    preprocessed data -> parsed/processed data -> FeatureDict
3. SingleDataset
    datapoints -> __getitem__ -> FeatureDict
4. StochasticSamplerDataset (optional)
    Sequence[SingleDataset] -> __getitem__ -> FeatureDict
5. DataLoader
    FeatureDict -> batched data
6. DataModule [YOU ARE HERE]
    SingleDataset/StochasticSamplerDataset -> DataLoader
7. ModelRunner
    batched data -> model
"""

import warnings
from functools import partial
from typing import Dict, List, Sequence

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from openfold3.core.data.framework.single_datasets.abstract_single_dataset import (
    DATASET_REGISTRY,
    SingleDataset,
)
from openfold3.core.data.framework.stochastic_sampler_dataset import (
    StochasticSamplerDataset,
)
from openfold3.core.utils.tensor_utils import dict_multimap


class DataModule(pl.LightningDataModule):
    """A LightningDataModule class for organizing Datasets and DataLoaders."""

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
        if ("train" in dataset_types) | ("validation" in dataset_types):
            # Initialize train datasets
            train_datasets = self.init_datasets(
                dataset_classes, dataset_configs, dataset_types, "train"
            )

            self.generator = torch.Generator(device="cpu").manual_seed(self.data_seed)

            # Wrap train datasets in the sampler dataset class
            self.train_dataset = StochasticSamplerDataset(
                datasets=train_datasets,
                probabilities=dataset_weights,
                virtual_epoch_len=self.virtual_epoch_len,
                num_virtual_epochs=self.num_virtual_epochs,
                generator=self.generator,
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

        elif "prediction" in dataset_types:
            # Currently only one prediction dataset is supported
            self.prediction_dataset = self.init_datasets(
                dataset_classes, dataset_configs, dataset_types, "prediction"
            )[0]

        else:
            raise ValueError(
                f"""No valid dataset types were found in data_config. Found: \
                {dataset_types}."""
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
            ("train" not in dataset_types) | ("validation" not in dataset_types)
        ):
            raise ValueError(
                "An unsupported combination of dataset types was found in"
                f"data_config: {dataset_types_unique}. The supported dataset"
                "combinations are: ['train'], ['train', 'validation'], ['test'],"
                "['prediction']."
            )
        elif (len(dataset_types_unique) == 1) & ("validation" in dataset_types):
            raise ValueError(
                "Validation dataset(s) were provided without any training datasets."
                "The supported dataset combinations are: ['train'], ['train', "
                "'validation'], ['test'], ['prediction']."
            )

        return dataset_classes, dataset_weights, dataset_configs, dataset_types

    def init_datasets(
        self,
        dataset_classes: list[Sequence[str]],
        dataset_configs: list[Sequence[dict]],
        dataset_types: list[Sequence[str]],
        type_to_init: str,
    ) -> list[Sequence[SingleDataset]]:
        """Initializes datasets.

        Args:
            dataset_classes (list[Sequence[str]]):
                List of strings matching the specific SingleDataset classes to
                initialize.
            dataset_configs (list[Sequence[dict]]):
                List of configs to pass each dataset class.
            dataset_types (list[Sequence[str]]):
                List of dataset types, elements can be train, validation, test,
                prediction.
            type_to_init (str):
                One of train, validation, test, prediction.

        Returns:
            list[Sequence[SingleDataset]]: List of initialized SingleDataset objects.
        """
        datasets = [
            DATASET_REGISTRY[dataset_class](dataset_config)
            for dataset_class, dataset_config, dataset_type in zip(
                dataset_classes, dataset_configs, dataset_types
            )
            if dataset_type == type_to_init
        ]
        if (type_to_init in ["validation", "test", "prediction"]) & (len(datasets) > 1):
            warnings.warn(
                f"""{len(datasets)} {type_to_init} datasets were found, using only the \
                first one.""",
                stacklevel=2,
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
            else self.validation_dataset
            if stage == "validation"
            else self.test_dataset
            if stage == "test"
            else self.prediction_dataset
        )

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
        return self.generate_dataloader("validation")

    def test_dataloader(self) -> DataLoader:
        return self.generate_dataloader("test")

    def predict_dataloader(self) -> DataLoader:
        return self.generate_dataloader("prediction")


def openfold_batch_collator(prots):
    stack_fn = partial(torch.stack, dim=0)
    return dict_multimap(stack_fn, prots)
