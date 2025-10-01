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
4. SamplerDataset (optional)
    Sequence[SingleDataset] -> __getitem__ -> FeatureDict
5. DataLoader
    FeatureDict -> batched data
6. DataModule [YOU ARE HERE]
    SingleDataset/SamplerDataset -> DataLoader
7. ModelRunner
    batched data -> model
"""

import dataclasses
import enum
import random
import warnings
from typing import Any, Optional, Union

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.utils
import torch.utils.data
from lightning_fabric.utilities.rank_zero import (
    rank_zero_only,
)
from lightning_utilities.core.imports import RequirementCache
from pydantic import BaseModel, SerializeAsAny
from torch.utils.data import DataLoader

from openfold3.core.data.framework.lightning_utils import _generate_seed_sequence
from openfold3.core.data.framework.single_datasets.abstract_single import (
    DATASET_REGISTRY,
    SingleDataset,
)
from openfold3.core.data.framework.stochastic_sampler_dataset import (
    SamplerDataset,
)
from openfold3.core.data.pipelines.preprocessing.template import TemplatePreprocessor
from openfold3.core.data.tools.colabfold_msa_server import (
    MsaComputationSettings,
    preprocess_colabfold_msas,
)
from openfold3.core.utils.tensor_utils import dict_multimap

_NUMPY_AVAILABLE = RequirementCache("numpy")


class DatasetMode(enum.Enum):
    """Enum for dataset modes."""

    train = "train"
    validation = "validation"
    test = "test"
    prediction = "prediction"


class DatasetSpec(BaseModel):
    """Dataset specification provided to initialize datasets in the DataModule.

    The DataModule accepts list of these configurations to create
    `torch.Datasets` for pl.Trainer.
    """

    name: str
    dataset_class: str
    mode: DatasetMode
    weight: Optional[float] = None
    config: SerializeAsAny[BaseModel] = SerializeAsAny()


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

    def get_config_for_mode(self, mode: DatasetMode) -> "MultiDatasetConfig":
        datasets_stage_mask = [m == mode for m in self.modes]
        return self.get_subset(datasets_stage_mask)


class DataModuleConfig(BaseModel):
    datasets: list[SerializeAsAny[BaseModel]]
    batch_size: int = 1
    num_workers: int = 0
    num_workers_validation: int = 0
    data_seed: int = 42
    epoch_len: int = 1
    num_epochs: int = 1000  # PL default


class DataModule(pl.LightningDataModule):
    """A LightningDataModule class for organizing Datasets and DataLoaders."""

    def __init__(
        self, data_module_config: DataModuleConfig, world_size: Optional[int] = None
    ) -> None:
        super().__init__()

        # Possibly initialize directly from DataModuleConfig
        self.batch_size = data_module_config.batch_size
        self.num_workers = data_module_config.num_workers
        self.num_workers_validation = data_module_config.num_workers_validation
        self.data_seed = data_module_config.data_seed
        self.epoch_len = data_module_config.epoch_len
        self.num_epochs = data_module_config.num_epochs
        self.world_size = world_size

        # Parse datasets
        self.multi_dataset_config = self.parse_data_config(data_module_config.datasets)
        self._initialize_next_dataset_indices()

    def _initialize_next_dataset_indices(self):
        train_configs = self.multi_dataset_config.get_config_for_mode(DatasetMode.train)
        self.next_dataset_indices = dict()
        for cfg in train_configs.configs:
            if cfg.sample_in_order:
                self.next_dataset_indices[cfg.name] = 0

    def setup(self, stage=None):
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
        self.generator = torch.Generator(device="cpu").manual_seed(self.data_seed)

        self.datasets_by_mode = {k: [] for k in DatasetMode}
        # Initialize datasets
        if DatasetMode.train in self.multi_dataset_config.modes:
            multi_dataset_config_train = self.multi_dataset_config.get_config_for_mode(
                DatasetMode.train
            )
            # Initialize train datasets
            all_train_datasets = self.init_datasets(multi_dataset_config_train)

            # Wrap train datasets in the sampler dataset class
            train_dataset = SamplerDataset(
                datasets=all_train_datasets,
                dataset_probabilities=multi_dataset_config_train.weights,
                epoch_len=self.epoch_len,
                num_epochs=self.num_epochs,
                generator=self.generator,
                next_dataset_indices=self.next_dataset_indices,
            )
            self.datasets_by_mode[DatasetMode.train] = train_dataset

        for dataset_mode in [
            DatasetMode.validation,
            DatasetMode.test,
            DatasetMode.prediction,
        ]:
            multi_dataset_config_mode = self.multi_dataset_config.get_config_for_mode(
                dataset_mode
            )
            mode_datasets = self.init_datasets(
                multi_dataset_config_mode, set_world_size=True
            )

            if len(mode_datasets) > 1:
                warnings.warn(
                    f"Currently only one {dataset_mode} dataset is supported, but "
                    f"{len(mode_datasets)} datasets were found. Using only the "
                    "first one.",
                    stacklevel=2,
                )

            self.datasets_by_mode[dataset_mode] = (
                mode_datasets[0] if mode_datasets else []
            )

    @classmethod
    def parse_data_config(cls, data_config: list[dict]) -> MultiDatasetConfig:
        """Parses input data_config into separate lists.

        Args:
            data_config (list[dict]):
                Input data configuration list of dataset dictionaries.

        Returns:
            MultiDatasetConfig:
                Lists of dataset classes, weights, configurations and unique set of
                modes.
        """

        classes, modes, configs, weights = [], [], [], []
        for dataset_entry in data_config:
            classes.append(dataset_entry.dataset_class)
            modes.append(dataset_entry.mode)
            weights.append(dataset_entry.weight)
            configs.append(dataset_entry.config)

        multi_dataset_config = MultiDatasetConfig(
            classes=classes,
            modes=modes,
            configs=configs,
            weights=weights,
        )

        # Check dataset configuration
        cls.run_checks(multi_dataset_config)

        return multi_dataset_config

    @classmethod
    def run_training_dataset_checks(
        cls,
        train_dataset_config: MultiDatasetConfig,
    ) -> None:
        """Check that dataset weights and crop weights are normalized"""
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
            config_i_crop_weights = config_i.crop.crop_weights.model_dump()
            if sum(config_i_crop_weights.values()) != 1:
                warnings.warn(
                    f"Dataset {train_dataset_config.classes[idx]} crop weights do not "
                    "sum to 1. Normalizing weights.",
                    stacklevel=2,
                )
                train_dataset_config.configs[idx].crop.crop_weights = {
                    key: value / sum(config_i_crop_weights.values())
                    for key, value in config_i_crop_weights.items()
                }
                print(f"{train_dataset_config.configs[idx].crop.crop_weights=}")

    @classmethod
    def run_checks(cls, multi_dataset_config: MultiDatasetConfig) -> None:
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
        if len(train_dataset_config.classes):
            cls.run_training_dataset_checks(train_dataset_config)

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
            {DatasetMode.validation},
            {DatasetMode.test},
            {DatasetMode.prediction},
        ]

        if modes_unique not in supported_combinations:
            raise ValueError(
                "An unsupported combination of dataset modes was found in"
                f"data_config: {modes_unique}. The supported dataset"
                f"combinations are: {supported_combinations}."
            )
        elif any([type_ not in supported_types for type_ in modes_unique]):
            raise ValueError(
                f"An unsupported dataset mode was found in data_config: {modes_unique}."
                " Supported modes are: train, validation, test, prediction."
            )

    def init_datasets(
        self, multi_dataset_config: MultiDatasetConfig, set_world_size: bool = False
    ) -> list[SingleDataset]:
        """Initializes datasets.

        Args:
            multi_dataset_config (MultiDatasetConfig):
                Parsed config of all input datasets.
            set_world_size: Whether to set the world size in the dataset initialization

        Returns:
            list[Sequence[SingleDataset]]: List of initialized SingleDataset objects.
        """
        # Note that the dataset config already contains the paths!
        datasets = []
        for dataset_class, dataset_config in zip(
            multi_dataset_config.classes,
            multi_dataset_config.configs,
        ):
            if set_world_size:
                dataset = DATASET_REGISTRY[dataset_class](
                    dataset_config, self.world_size
                )
            else:
                dataset = DATASET_REGISTRY[dataset_class](dataset_config)
            datasets.append(dataset)
        return datasets

    def generate_dataloader(self, mode: DatasetMode):
        """Wrap the appropriate dataset in a DataLoader and return it.

        Args:
            mode (DatasetMode):
                Mode of DataLoader to return, one of train, valid, test, predict.

        Returns:
            DataLoader: DataLoader object.
        """

        # TODO: Val does not need this many workers. Due to memory leak issue,
        #  reduce workers here to run with more workers overall in training
        #  as temporary quick fix.
        if (
            mode == DatasetMode.validation
            and DatasetMode.train in self.multi_dataset_config.modes
        ):
            num_workers = self.num_workers_validation
        else:
            num_workers = self.num_workers

        return DataLoader(
            dataset=self.datasets_by_mode[mode],
            batch_size=self.batch_size,
            num_workers=num_workers,
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

    def state_dict(self):
        state = {"next_dataset_indices": self.next_dataset_indices}
        return state

    def load_state_dict(self, state_dict: dict[str, Any]):
        if not self.next_dataset_indices:
            return

        loaded_index_keys = state_dict["next_dataset_indices"].keys()
        current_index_keys = self.next_dataset_indices.keys()
        if set(loaded_index_keys) != set(current_index_keys):
            raise ValueError(
                "Datasets selected for in-order sampling do not match in"
                "current configuration and checkpoint."
                f"Current {current_index_keys} Checkpoint {loaded_index_keys}"
            )
        self.next_dataset_indices = state_dict["next_dataset_indices"]


class InferenceDataModule(DataModule):
    """LightnigngDataModule that contains a prepare_data hook for inference."""

    def __init__(
        self,
        data_module_config: DataModuleConfig,
        world_size: int | None = None,
        use_msa_server: bool = False,
        use_templates: bool = False,
        msa_computation_settings: MsaComputationSettings | None = None,
    ):
        # get information about msas from the experiment runner
        # probably should add to the config
        super().__init__(data_module_config, world_size)
        self.use_msa_server = use_msa_server
        self.use_templates = use_templates
        self.msa_computation_settings = msa_computation_settings
        _configs = self.multi_dataset_config.get_config_for_mode(DatasetMode.prediction)
        self.inference_config = _configs.configs[0]

    def prepare_data(self) -> None:
        # Colabfold msa preparation
        if self.use_msa_server:
            self.inference_config.query_set = preprocess_colabfold_msas(
                inference_query_set=self.inference_config.query_set,
                compute_settings=self.msa_computation_settings,
            )

        if self.use_templates:
            template_preprocessor = TemplatePreprocessor(
                input_set=self.inference_config.query_set,
                config=self.inference_config.template_preprocessor,
            )
            template_preprocessor()

    def setup(self, stage=None):
        """Broadcast updated query set to all ranks if multiple GPUs are used."""
        if self.world_size and self.world_size > 1:
            if dist.get_rank() == 0:
                placeholder = [self.inference_config.query_set]
            else:
                placeholder = [None]
            dist.broadcast_object_list(placeholder, src=0)
            self.inference_config.query_set = placeholder[0]
        super().setup()


# TODO: Remove debug logic and improve handlingi of training only features
def openfold_batch_collator(samples: list[dict[str, torch.Tensor]]):
    """Collates a list of samples into a batch."""

    has_pdb_id = "pdb_id" in samples[0]

    if has_pdb_id:
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
    has_ref_space_uid_to_perm = "ref_space_uid_to_perm" in samples[0]

    if has_ref_space_uid_to_perm:
        ref_space_uid_to_perm_dicts = []
        for sample in samples:
            ref_space_uid_to_perm_dicts.append(sample.pop("ref_space_uid_to_perm"))

    samples = dict_multimap(pad_feat_fn, samples)

    # Add the ref_space_uid_to_perm back to the samples
    if has_ref_space_uid_to_perm:
        samples["ref_space_uid_to_perm"] = ref_space_uid_to_perm_dicts

    if has_pdb_id:
        samples["pdb_id"] = pdb_ids

    return samples
