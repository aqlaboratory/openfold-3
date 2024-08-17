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
import warnings
from functools import partial
from typing import Any, Union

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from openfold3.core.data.framework.single_datasets.abstract_single_dataset import (
    DATASET_REGISTRY,
    SingleDataset,
)
from openfold3.core.data.framework.stochastic_sampler_dataset import (
    StochasticSamplerDataset,
)
from openfold3.core.utils.tensor_utils import dict_multimap


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


class DatasetType(enum.Enum):
    """Enum for dataset types."""

    TRAIN = enum.auto()
    VALIDATION = enum.auto()
    TEST = enum.auto()
    PREDICTION = enum.auto()


class DataModule(pl.LightningDataModule):
    """A LightningDataModule class for organizing Datasets and DataLoaders."""

    def __init__(self, data_config: list[dict]) -> None:
        super().__init__()

        # Input argument self-assignment
        self.batch_size = data_config["batch_size"]
        self.num_workers = data_config["num_workers"]
        self.data_seed = data_config["data_seed"]
        self.virtual_epoch_len = data_config["virtual_epoch_len"]
        self.num_virtual_epochs = data_config["num_virtual_epochs"]

        # Parse data_config
        multi_dataset_config = self.parse_data_config(data_config)

        # Set up worker_init_fn as a closure
        def worker_init_fn(worker_id: int) -> None:
            """Worker initialization function for setting random seeds.

            Args:
                worker_id (int):
                    Worker ID.
            """
            # Get the process rank (in DDP, each process is assigned a rank)
            rank = dist.get_rank() if dist.is_initialized() else 0

            # Calculate a unique seed based on the global seed, rank, and worker ID
            # Use dataset seed here and ensure it's within 32-bit range
            seed = self.data_seed % 2**32
            # Modify seed with rank and worker ID
            seed += rank * 1000 + worker_id

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            # For multi-GPU scenarios
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.worker_init_fn = worker_init_fn

        # Initialize datasets
        if DatasetType.train in multi_dataset_config.types:
            # Initialize train datasets
            train_datasets = self.init_datasets(multi_dataset_config, DatasetType.train)

            self.generator = torch.Generator(device="cpu").manual_seed(self.data_seed)

            # Wrap train datasets in the sampler dataset class
            self.train_dataset = StochasticSamplerDataset(
                datasets=train_datasets,
                probabilities=multi_dataset_config.weights,
                virtual_epoch_len=self.virtual_epoch_len,
                num_virtual_epochs=self.num_virtual_epochs,
                generator=self.generator,
            )

        if "validation" in multi_dataset_config.types:
            self.validation_dataset = self.init_datasets(
                multi_dataset_config, "validation"
            )[0]

        if "test" in multi_dataset_config.types:
            self.test_dataset = self.init_datasets(multi_dataset_config, "test")[0]

        if "prediction" in multi_dataset_config.types:
            self.prediction_dataset = self.init_datasets(
                multi_dataset_config, "prediction"
            )[0]

    def parse_data_config(self, data_config: list[dict]) -> MultiDatasetConfig:
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
                        get_cast(dataset_entry, "type", str),
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
        if sum(multi_dataset_config.weights) != 1:
            warnings.warn(
                "Dataset weights do not sum to 1. Normalizing weights.",
                stacklevel=2,
            )
            multi_dataset_config.weights = [
                weight / sum(multi_dataset_config.weights)
                for weight in multi_dataset_config.weights
            ]

        # Check if provided crop weights sum to 1
        for idx, config_i in enumerate(multi_dataset_config.configs):
            if sum(config_i["crop_weights"].values()) != 1:
                warnings.warn(
                    f"Dataset {multi_dataset_config.classes[idx]} crop weights do not "
                    "sum to 1. Normalizing weights.",
                    stacklevel=2,
                )
                multi_dataset_config.configs[idx]["crop_weights"] = {
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
        if types_unique == {"validation"}:
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
        type_to_init: str,
    ) -> list[SingleDataset]:
        """Initializes datasets.

        Args:
            multi_dataset_config (MultiDatasetConfig):
                Parsed config of all input datasets.
            type_to_init (str):
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

        if (type_to_init in ["validation", "test", "prediction"]) & (len(datasets) > 1):
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
            worker_init_fn=self.worker_init_fn,
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
