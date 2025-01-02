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
from pathlib import Path
from typing import Any, Optional, Union

import pytorch_lightning as pl
import torch
import torch.utils
import torch.utils.data
from lightning_fabric.utilities.rank_zero import (
    rank_zero_only,
)
from lightning_utilities.core.imports import RequirementCache
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from openfold3.core.data.framework.lightning_utils import _generate_seed_sequence
from openfold3.core.data.framework.single_datasets.abstract_single import (
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
        modes: list[str]
            List of dataset modes, elements can be train, validation, test, prediction.
        configs: list[Union[dict[str, Any], None]]
            List of dictionaries containing SingleDataset init input paths (!!!) and
            other config arguments if needed/available.
        weights: list[float]
            List of weights used for sampling from each dataset.
    """

    classes: list[str]
    modes: list[str]
    configs: list[Union[dict[str, Any], None]]
    weights: list[float]

    def __len__(self):
        return len(self.classes)

    def get_subset(self, index: list[bool]) -> "MultiDatasetConfig":
        """Returns a subset of the MultiDatasetConfig.

        Args:
            index (bool):
                Index of the subset.

        Returns:
            MultiDatasetConfig:
                Subset of the MultiDatasetConfig.
        """

        def apply_bool(value, idx):
            return [v for v, i in zip(value, idx) if i]

        return MultiDatasetConfig(
            classes=apply_bool(self.classes, index),
            modes=apply_bool(self.modes, index),
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

    def to_dict(self):
        _dict = self.__dict__.copy()
        datasets = []
        for d in _dict["datasets"]:
            d.config.dataset_paths = {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in d.config.dataset_paths.items()
            }
            datasets.append(d.to_dict())
        _dict["datasets"] = datasets
        return _dict


class DatasetMode(enum.Enum):
    """Enum for dataset modes."""

    train = enum.auto()
    validation = enum.auto()
    test = enum.auto()
    prediction = enum.auto()


class DataModule(pl.LightningDataModule):
    """A LightningDataModule class for organizing Datasets and DataLoaders."""

    def __init__(self, data_config: DataModuleConfig) -> None:
        super().__init__()

        # Possibly initialize directly from DataModuleConfig
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
        if DatasetMode.train in multi_dataset_config.modes:
            # Initialize train datasets
            train_datasets = self.init_datasets(multi_dataset_config, DatasetMode.train)
            multi_dataset_config_train = multi_dataset_config.get_subset(
                [mode == DatasetMode.train for mode in multi_dataset_config.modes]
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

        if DatasetMode.validation in multi_dataset_config.modes:
            multi_dataset_config_validation = multi_dataset_config.get_subset(
                [mode == DatasetMode.validation for mode in multi_dataset_config.modes]
            )
            self.validation_dataset = self.init_datasets(
                multi_dataset_config_validation, DatasetMode.validation
            )[0]
        # Dummy is needed as PLightning will still try to access the validation dataset.
        else:
            self.validation_dataset = []

        if DatasetMode.test in multi_dataset_config.modes:
            multi_dataset_config_test = multi_dataset_config.get_subset(
                [mode == DatasetMode.test for mode in multi_dataset_config.modes]
            )
            self.test_dataset = self.init_datasets(
                multi_dataset_config_test, DatasetMode.test
            )[0]

        if DatasetMode.prediction in multi_dataset_config.modes:
            multi_dataset_config_prediction = multi_dataset_config.get_subset(
                [mode == DatasetMode.prediction for mode in multi_dataset_config.modes]
            )
            self.prediction_dataset = self.init_datasets(
                multi_dataset_config_prediction, DatasetMode.prediction
            )[0]

    @classmethod
    def parse_data_config(cls, data_config: list[ConfigDict]) -> MultiDatasetConfig:
        """Parses input data_config into separate lists.

        Args:
            data_config (list[dict]):
                Input data configuration list of dataset dictionaries.

        Returns:
            MultiDatasetConfig:
                Lists of dataset classes, weights, configurations and unique set of
                modes.
        """

        def get_cast(
            dictionary: Union[dict, ConfigDict],
            key: Union[str, int],
            cast_type: type,
            default: Any = None,
        ) -> Any:
            """Simultaneously try to get and try to cast a value from a dictionary.

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

        classes, modes, configs, weights = list(
            zip(
                *[
                    (
                        get_cast(dataset_entry, "class", str),
                        DatasetMode[get_cast(dataset_entry, "mode", str)],
                        dataset_entry.get("config", None),
                        get_cast(dataset_entry, "weight", float),
                    )
                    for dataset_entry in data_config
                ]
            )
        )
        multi_dataset_config = MultiDatasetConfig(
            classes=classes,
            modes=modes,
            configs=configs,
            weights=weights,
        )

        # Check dataset configuration
        cls.run_checks(multi_dataset_config)

        return multi_dataset_config

    @staticmethod
    def run_checks(multi_dataset_config: MultiDatasetConfig) -> None:
        """Runs checks on the provided crop weights and modes.

        Checks for valid combinations of SingleDataset modes and normalizes weights and
        cropping weights if available, and they do not sum to 1. Updates
        multi_dataset_config in place.

        Args:
            multi_dataset_config: DatasetConfig:
                Parsed dataset config.

        Returns:
            None.
        """

        # Check if provided weights sum to 1
        train_dataset_config = multi_dataset_config.get_subset(
            [mode == DatasetMode.train for mode in multi_dataset_config.modes]
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

        # Check if provided dataset mode combination is valid
        modes = multi_dataset_config.modes
        modes_unique = set(modes)
        supported_types = {
            DatasetMode.train,
            DatasetMode.validation,
            DatasetMode.test,
            DatasetMode.prediction,
        }
        supported_combinations = [
            {DatasetMode.train},
            {DatasetMode.train, DatasetMode.validation},
            {DatasetMode.test},
            {DatasetMode.prediction},
        ]

        if modes_unique not in supported_combinations:
            raise ValueError(
                "An unsupported combination of dataset modes was found in"
                f"data_config: {modes_unique}. The supported dataset"
                f"combinations are: {supported_combinations}."
            )
        if modes_unique == {DatasetMode.validation}:
            raise ValueError(
                "Validation dataset(s) were provided without any training datasets."
                f"The supported dataset combinations are: {supported_combinations}."
            )
        elif any([type_ not in supported_types for type_ in modes_unique]):
            raise ValueError(
                f"An unsupported dataset mode was found in data_config: {modes_unique}."
                " Supported modes are: train, validation, test, prediction."
            )
        # REMOVE THIS WITH ENUM
        elif (len(modes_unique) == 1) & (
            all(
                [
                    DatasetMode.train not in modes,
                    DatasetMode.validation not in modes,
                    DatasetMode.test not in modes,
                    DatasetMode.prediction not in modes,
                ]
            )
        ):
            raise ValueError(
                "An unsupported combination of dataset modes was found in"
                f"data_config: {modes_unique}. The supported dataset"
                f"combinations are: {supported_combinations}."
            )

    @staticmethod
    def init_datasets(
        multi_dataset_config: MultiDatasetConfig,
        type_to_init: DatasetMode,
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
                multi_dataset_config.modes,
            )
            if dataset_type == type_to_init
        ]

        if (
            type_to_init
            in [DatasetMode.validation, DatasetMode.test, DatasetMode.prediction]
        ) & (len(datasets) > 1):
            datasets = datasets[:1]
            warnings.warn(
                f"Currently only one {type_to_init} dataset is supported, but "
                f"{len(datasets)} datasets were found. Using only the "
                "first one.",
                stacklevel=2,
            )
        return datasets

    def generate_dataloader(self, stage: DatasetMode):
        """Wrap the appropriate dataset in a DataLoader and return it.

        Args:
            stage (str):
                Mode of DataLoader to return, one of train, valid, test, predict.

        Returns:
            DataLoader: DataLoader object.
        """
        dataset = (
            self.train_dataset
            if stage == DatasetMode.train
            else self.validation_dataset
            if stage == DatasetMode.validation
            else self.test_dataset
            if stage == DatasetMode.test
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
        return self.generate_dataloader(DatasetMode.train)

    def val_dataloader(self) -> DataLoader:
        """Creates validation dataloader.

        Returns:
            DataLoader: validation dataloader.
        """
        return self.generate_dataloader(DatasetMode.validation)

    def test_dataloader(self) -> DataLoader:
        """Creates test dataloader.

        Returns:
            DataLoader: test dataloader.
        """
        return self.generate_dataloader(DatasetMode.test)

    def predict_dataloader(self) -> DataLoader:
        """Creates prediction dataloader.

        Returns:
            DataLoader: prediction dataloader.
        """
        return self.generate_dataloader(DatasetMode.prediction)


# TODO: Remove debug logic
def openfold_batch_collator(samples: list[dict[str, torch.Tensor]]):
    """Collates a list of samples into a batch."""

    pdb_ids = [s.pop("pdb_id") for s in samples]

    def pad_feat_fn(values: list[torch.Tensor]) -> torch.Tensor:
        """
        Pad the tensors to the same length. Remove the extra dimension if stacking
        a 1D tensor (i.e. loss weights).
        """
        values = torch.nn.utils.rnn.pad_sequence(
            values, batch_first=True, padding_value=0
        )
        return values.squeeze(-1)

    # The ligand permutation mappings are a special feature and need to be handled
    # separately
    ref_space_uid_to_perm_dicts = []
    for sample in samples:
        ref_space_uid_to_perm_dicts.append(sample.pop("ref_space_uid_to_perm"))

    samples = dict_multimap(pad_feat_fn, samples)

    # Add the ref_space_uid_to_perm back to the samples
    samples["ref_space_uid_to_perm"] = ref_space_uid_to_perm_dicts

    samples["pdb_id"] = ", ".join(pdb_ids)

    return samples
