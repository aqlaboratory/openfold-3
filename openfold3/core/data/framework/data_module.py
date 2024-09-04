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

import dataclasses
import enum
import random
import warnings
from functools import partial
from ml_collections import ConfigDict
from typing import Any, Optional, Union

import pytorch_lightning as pl
import torch
import torch.utils
import torch.utils.data
from lightning_fabric.utilities.rank_zero import (
    rank_zero_only,
)
from lightning_utilities.core.imports import RequirementCache
from torch.utils.data import DataLoader

from openfold3.core.data.framework.lightning_utils import _generate_seed_sequence
from openfold3.core.data.framework.single_datasets.abstract_single_dataset import (
    DATASET_REGISTRY,
    SingleDataset,
)
from openfold3.core.data.framework.stochastic_sampler_dataset import (
    StochasticSamplerDataset,
)
from openfold3.core.utils.tensor_utils import dict_multimap

_NUMPY_AVAILABLE = RequirementCache("numpy")


@dataclasses.dataclass
class MultiDatasetConfig:
    """Dataclass for storing dataset configurations.

    Attributes:
        classes: list[str]
            List of dataset class names used as keys into the dataset registry.
        types: list[str]
            List of dataset types, elements can be train, validation, test, prediction.
        configs: list[Union[dict[str, Any], None]]
            List of dictionaries containing SingleDataset init input paths (!!!) and
            other config arguments if needed/available.
        weights: list[float]
            List of weights used for sampling from each dataset.
    """

    classes: list[str]
    types: list[str]
    configs: list[Union[dict[str, Any], None]]
    weights: list[float]

    def __len__(self):
        return len(self.classes)

    def get_subset(self, index: bool) -> "MultiDatasetConfig":
        """Returns a subset of the MultiDatasetConfig.

        Args:
            index (bool):
                Index of the subset.

        Returns:
            MultiDatasetConfig:
                Subset of the MultiDatasetConfig.
        """

        def apply_bool(value, index):
            return [v for v, i in zip(value, index) if i]

        return MultiDatasetConfig(
            classes=apply_bool(self.classes, index),
            types=apply_bool(self.types, index),
            configs=apply_bool(self.configs, index),
            weights=apply_bool(self.weights, index),
        )


@dataclasses.dataclass
class DataModuleConfig:
    batch_size: int
    num_workers: int
    data_seed: int
    epoch_len: int
    num_epochs: int
    datasets: list[ConfigDict]


class DatasetType(enum.Enum):
    """Enum for dataset types."""

    train = enum.auto()
    validation = enum.auto()
    test = enum.auto()
    prediction = enum.auto()


class DataModule(pl.LightningDataModule):
    """A LightningDataModule class for organizing Datasets and DataLoaders."""

    def __init__(self, data_config: DataModuleConfig) -> None:
        super().__init__()

        # Possibly remove this block in favor of initializing directly from DataModuleConfig
        self.batch_size = data_config.batch_size
        self.num_workers = data_config.num_workers
        self.data_seed = data_config.data_seed
        self.epoch_len = data_config.epoch_len
        self.num_epochs = data_config.num_epochs

        # Parse datasets
        multi_dataset_config = self.parse_data_config(data_config.datasets)

        # Custom worker init function with manual data seed
        def worker_init_function_with_data_seed(
            worker_id: int, rank: Optional[int] = None
        ) -> None:
            """Modified default Lightning worker_init_fn with manual data seed.

            This worker_init_fn enables decoupling stochastic processes in the data
            pipeline from those in the model. Taken from Pytorch Lightning 2.4.1 source
            code: https://github.com/Lightning-AI/pytorch-lightning/blob/f3f10d460338ca8b2901d5cd43456992131767ec/src/lightning/fabric/utilities/seed.py#L85

            Args:
                worker_id (int):
                    Worker id.
                rank (Optional[int], optional):
                    Worker process rank. Defaults to None.
            """
            # implementation notes: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
            global_rank = rank if rank is not None else rank_zero_only.rank
            process_seed = self.data_seed
            # back out the base seed so we can use all the bits
            base_seed = process_seed - worker_id
            seed_sequence = _generate_seed_sequence(
                base_seed, worker_id, global_rank, count=4
            )
            torch.manual_seed(seed_sequence[0])  # torch takes a 64-bit seed
            random.seed(
                (seed_sequence[1] << 32) | seed_sequence[2]
            )  # combine two 64-bit seeds
            if _NUMPY_AVAILABLE:
                import numpy as np

                np.random.seed(
                    seed_sequence[3] & 0xFFFFFFFF
                )  # numpy takes 32-bit seed only

        self.worker_init_function_with_data_seed = worker_init_function_with_data_seed

        # Initialize datasets
        if DatasetType.train in multi_dataset_config.types:
            # Initialize train datasets
            train_datasets = self.init_datasets(multi_dataset_config, DatasetType.train)
            multi_dataset_config_train = multi_dataset_config.get_subset(
                [type == DatasetType.train for type in multi_dataset_config.types]
            )
            self.generator = torch.Generator(device="cpu").manual_seed(self.data_seed)

            # Wrap train datasets in the sampler dataset class
            self.train_dataset = StochasticSamplerDataset(
                datasets=train_datasets,
                dataset_probabilities=multi_dataset_config_train.weights,
                epoch_len=self.epoch_len,
                num_epochs=self.num_epochs,
                generator=self.generator,
            )

        if DatasetType.validation in multi_dataset_config.types:
            multi_dataset_config_validation = multi_dataset_config.get_subset(
                [type == DatasetType.validation for type in multi_dataset_config.types]
            )
            self.validation_dataset = self.init_datasets(
                multi_dataset_config_validation, DatasetType.validation
            )[0]

        if DatasetType.test in multi_dataset_config.types:
            multi_dataset_config_test = multi_dataset_config.get_subset(
                [type == DatasetType.test for type in multi_dataset_config.types]
            )
            self.test_dataset = self.init_datasets(
                multi_dataset_config_test, DatasetType.test
            )[0]

        if DatasetType.prediction in multi_dataset_config.types:
            multi_dataset_config_prediction = multi_dataset_config.get_subset(
                [type == DatasetType.prediction for type in multi_dataset_config.types]
            )
            self.prediction_dataset = self.init_datasets(
                multi_dataset_config_prediction, DatasetType.prediction
            )[0]

    def parse_data_config(self, data_config: list[ConfigDict]) -> MultiDatasetConfig:
        """Parses input data_config into separate lists.

        Args:
            data_config (list[dict]):
                Input data configuration list of dataset dictionaries.

        Returns:
            MultiDatasetConfig:
                Lists of dataset classes, weights, configurations and unique set of
                types.
        """

        def get_cast(
            dictionary: dict, key: Union[str, int], cast_type: type, default: Any = None
        ) -> Any:
            """Simultanously try to get and try to cast a value from a dictionary.

            Args:
                dictionary (dict):
                    Dictionary to get the value from.
                key (Union[str, int]):
                    Key to get the value from.
                cast_type (type):
                    Type to cast the value to.
                default (Any, optional):
                    Default value to return if key not available. Defaults to None.

            Raises:
                ValueError:
                    If the value cannot be cast to the specified type.

            Returns:
                Any:
                    Cast value or default.
            """
            value = dictionary.get(key, default)
            try:
                return cast_type(value) if value is not None else default
            except ValueError as exc:
                raise ValueError(f"Could not cast {key} to {cast_type}.") from exc

        classes, types, configs, weights = list(
            zip(
                *[
                    (
                        get_cast(dataset_entry, "class", str),
                        DatasetType[get_cast(dataset_entry, "type", str)],
                        dataset_entry.get("config", None),
                        get_cast(dataset_entry, "weight", float),
                    )
                    for dataset_entry in data_config
                ]
            )
        )
        multi_dataset_config = MultiDatasetConfig(
            classes=classes,
            types=types,
            configs=configs,
            weights=weights,
        )

        # Check dataset configuration
        self.run_checks(multi_dataset_config)

        return multi_dataset_config

    def run_checks(self, multi_dataset_config: MultiDatasetConfig) -> None:
        """Runs checks on the provided crop weights and types.

        Checks for valid combinations of SingleDataset types and normalizes weights and
        cropping weights if available and they do not sum to 1. Updates
        multi_dataset_config in place.

        Args:
            multi_dataset_config: DatasetConfig:
                Parsed dataset config.

        Returns:
            None.
        """

        # Check if provided weights sum to 1
        train_dataset_config = multi_dataset_config.get_subset(
            [type == DatasetType.train for type in multi_dataset_config.types]
        )
        if sum(train_dataset_config.weights) != 1:
            warnings.warn(
                "Dataset weights do not sum to 1. Normalizing weights.",
                stacklevel=2,
            )
            train_dataset_config.weights = [
                weight / sum(train_dataset_config.weights)
                for weight in train_dataset_config.weights
            ]

        # Check if provided crop weights sum to 1
        for idx, config_i in enumerate(train_dataset_config.configs):
            if sum(config_i["crop_weights"].values()) != 1:
                warnings.warn(
                    f"Dataset {train_dataset_config.classes[idx]} crop weights do not "
                    "sum to 1. Normalizing weights.",
                    stacklevel=2,
                )
                train_dataset_config.configs[idx]["crop_weights"] = {
                    key: value / sum(config_i["crop_weights"].values())
                    for key, value in config_i["crop_weights"].items()
                }

        # Check if provided dataset type combination is valid
        types = multi_dataset_config.types
        types_unique = set(types)
        supported_types = {
            DatasetType.train,
            DatasetType.validation,
            DatasetType.test,
            DatasetType.prediction,
        }
        supported_combinations = [
            {DatasetType.train},
            {DatasetType.train, DatasetType.validation},
            {DatasetType.test},
            {DatasetType.prediction},
        ]

        if types_unique not in supported_combinations:
            raise ValueError(
                "An unsupported combination of dataset types was found in"
                f"data_config: {types_unique}. The supported dataset"
                f"combinations are: {supported_combinations}."
            )
        if types_unique == {DatasetType.validation}:
            raise ValueError(
                "Validation dataset(s) were provided without any training datasets."
                f"The supported dataset combinations are: {supported_combinations}."
            )
        elif any([type_ not in supported_types for type_ in types_unique]):
            raise ValueError(
                f"An unsupported dataset type was found in data_config: {types_unique}."
                " Supported types are: train, validation, test, prediction."
            )
        # REMOVE THIS WITH ENUM
        elif (len(types_unique) == 1) & (
            all(
                [
                    DatasetType.train not in types,
                    DatasetType.validation not in types,
                    DatasetType.test not in types,
                    DatasetType.prediction not in types,
                ]
            )
        ):
            raise ValueError(
                "An unsupported combination of dataset types was found in"
                f"data_config: {types_unique}. The supported dataset"
                f"combinations are: {supported_combinations}."
            )

    def init_datasets(
        self,
        multi_dataset_config: MultiDatasetConfig,
        type_to_init: DatasetType,
    ) -> list[SingleDataset]:
        """Initializes datasets.

        Args:
            multi_dataset_config (MultiDatasetConfig):
                Parsed config of all input datasets.
            type_to_init (DatasetType):
                One of train, validation, test, prediction.

        Returns:
            list[Sequence[SingleDataset]]: List of initialized SingleDataset objects.
        """
        # Note that the dataset config already contains the paths!
        datasets = [
            DATASET_REGISTRY[dataset_class](dataset_config)
            for dataset_class, dataset_config, dataset_type in zip(
                multi_dataset_config.classes,
                multi_dataset_config.configs,
                multi_dataset_config.types,
            )
            if dataset_type == type_to_init
        ]

        if (
            type_to_init
            in [DatasetType.validation, DatasetType.test, DatasetType.prediction]
        ) & (len(datasets) > 1):
            datasets = datasets[:1]
            warnings.warn(
                f"Currently only one {type_to_init} dataset is supported, but "
                f"{len(datasets)} datasets were found. Using only the "
                "first one.",
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
            if stage == DatasetType.train
            else self.validation_dataset
            if stage == DatasetType.validation
            else self.test_dataset
            if stage == DatasetType.test
            else self.prediction_dataset
        )

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=openfold_batch_collator,
            generator=self.generator,
            worker_init_fn=self.worker_init_function_with_data_seed,
        )

    def train_dataloader(self) -> DataLoader:
        """Creates training dataloader.

        Returns:
            DataLoader: training dataloader.
        """
        return self.generate_dataloader(DatasetType.train)

    def val_dataloader(self) -> DataLoader:
        """Creates validation dataloader.

        Returns:
            DataLoader: validation dataloader.
        """
        return self.generate_dataloader(DatasetType.validation)

    def test_dataloader(self) -> DataLoader:
        """Creates test dataloader.

        Returns:
            DataLoader: test dataloader.
        """
        return self.generate_dataloader(DatasetType.test)

    def predict_dataloader(self) -> DataLoader:
        """Creates prediction dataloader.

        Returns:
            DataLoader: prediction dataloader.
        """
        return self.generate_dataloader(DatasetType.prediction)


def openfold_batch_collator(prots):
    stack_fn = partial(torch.stack, dim=0)
    return dict_multimap(stack_fn, prots)
