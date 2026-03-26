# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import random
import warnings
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic import BaseModel, field_validator, model_validator
from pydantic import ConfigDict as PydanticConfigDict

from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from openfold3.core.data.tools.colabfold_msa_server import MsaComputationSettings
from openfold3.entry_points.parameters import (
    DEFAULT_CACHE_PATH,
    DEFAULT_CHECKPOINT_NAME,
    OPENFOLD_MODEL_CHECKPOINT_REGISTRY,
    download_model_parameters,
    get_default_checkpoint_dir,
)
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    InferenceDatasetConfigKwargs,
    TrainingDatasetPaths,
)
from openfold3.projects.of3_all_atom.project_entry import ModelUpdate

logger = logging.getLogger(__name__)

ValidModeType = Literal["train", "predict", "eval", "test"]


class CheckpointConfig(BaseModel):
    """Settings for training checkpoint writing."""

    monitor: str | None = None
    mode: str | None = None
    every_n_epochs: int = 1
    auto_insert_metric_name: bool = False
    filename: str | None = None
    enable_version_counter: bool = True
    save_last: bool = True
    save_top_k: int = -1
    every_n_train_steps: int | None = None
    save_on_train_epoch_end: bool | None = None

    every_n_train_steps: int | None = None
    train_time_interval: Any | None = None

    @model_validator(mode="after")
    def validate_checkpoint_settings(self):
        if self.every_n_train_steps not in (None, 0):
            raise ValueError(
                "Mid-epoch checkpointing is not allowed: set "
                "checkpoint_config.every_n_train_steps to null. "
                "Use every_n_epochs in the checkpoint callback "
                "config for epoch-boundary checkpointing."
            )
        if self.train_time_interval not in (None, 0, "", False):
            raise ValueError(
                "Mid-epoch checkpointing is not allowed: set "
                "checkpoint_config.train_time_interval to null. "
                "Use every_n_epochs in the checkpoint callback "
                "config for epoch-boundary checkpointing."
            )
        return self


class WandbConfig(BaseModel):
    """Configuration for Weights and Biases experiment result logging."""

    project: str | None = None
    experiment_name: str | None = None
    entity: str | None = None
    group: str | None = None
    id: str | None = None
    offline: bool = False


class LoggingConfig(BaseModel):
    """Settings for training logging."""

    log_lr: bool = True
    log_grads: bool = False
    log_level: Literal["debug", "info", "warning", "error"] | None = None
    wandb_config: WandbConfig | None = None


class DataModuleArgs(BaseModel):
    """Settings for openfold3.core.data.framework.data_module"""

    model_config = PydanticConfigDict(extra="forbid")
    batch_size: int = 1
    data_seed: int | None = None
    num_workers: int = 10
    num_workers_validation: int = 4
    epoch_len: int = 4


class PlTrainerArgs(BaseModel):
    """Arguments to configure pl.Trainer, including settings for number of devices."""

    model_config = PydanticConfigDict(extra="allow")
    max_epochs: int = 1000  # pl_trainer default
    accelerator: str = "gpu"
    precision: int | str = "32-true"
    num_nodes: int = 1
    devices: int = 1  # number of GPUs per node
    profiler: str | None = None
    log_every_n_steps: int = 1
    enable_checkpointing: bool = True
    enable_model_summary: bool = False
    accumulate_grad_batches: int = 1
    gradient_clip_val: int | float | None = None
    gradient_clip_algorithm: str | None = None

    # Extra arguments that are not passed directly to pl.Trainer
    deepspeed_config_path: Path | None = None
    distributed_timeout: timedelta | None = default_pg_timeout
    mpi_plugin: bool = False

    use_distributed_sampler: bool = True

    @model_validator(mode="after")
    def validate_distributed_sampler_settings(self):
        if self.use_distributed_sampler is False:
            warnings.warn(
                "pl_trainer_args.use_distributed_sampler is set to False. "
                "Note that this arg is currently being ignored as we always use "
                "the OF3DistributedSampler for training.",
                stacklevel=2,
            )
        return self


class OutputWritingSettings(BaseModel):
    """File formats to use for writing inference prediction results.

    Used by OF3OutputWriter in openfold3.core.runners.writer
    """

    structure_format: Literal["pdb", "cif"] = "cif"
    full_confidence_output_format: Literal["json", "npz"] = "json"
    write_features: bool = False
    write_latent_outputs: bool = False


class ExperimentSettings(BaseModel):
    """General settings for all experiments"""

    mode: ValidModeType
    output_dir: Path = Path("./")
    log_dir: Path | None = None

    @field_validator("output_dir", mode="after")
    def create_output_dir(cls, value: Path):
        if not value.exists():
            value.mkdir(parents=True, exist_ok=True)
        return value


class CheckpointLoadingSettings(BaseModel):
    """
    Provides more granular control over checkpoint loading.
    While the standard PL process restores the entire training state,
    these settings allow for selective loading of specific components.
    """

    manual_checkpoint_loading: bool = False
    init_from_ema_weights: bool = False
    restore_lr_scheduler: bool = False
    restore_time_step: bool = False
    strict_loading: bool = True


class TrainingExperimentSettings(ExperimentSettings):
    """General settings specific for training experiments"""

    mode: ValidModeType = "train"
    seed: int = 42
    restart_checkpoint_path: str | None = None
    preemption_safe_resume: bool = False
    ckpt_load_settings: CheckpointLoadingSettings = CheckpointLoadingSettings()

    @field_validator("restart_checkpoint_path", mode="before")
    def validate_checkpoint_path(cls, value: Any) -> str | None:
        """
        Validates the restart_checkpoint_path.

        The path can be one of the following:
        - None (if no checkpoint is provided).
        - A special string: "last", "hpc", "registry" accepted by PL.
        - A string representing a valid path to a file.
        - A string representing a valid path to a directory (for deepspeed checkpoints).
        """
        # PL accepted strings
        allowed_strings = ["last", "hpc", "registry"]
        allowed_values = allowed_strings + [None]

        if value not in allowed_values and not Path(value).exists():
            raise ValueError(
                f'"{value}" is not a valid file, directory, or accepted keyword '
                f"({', '.join(allowed_strings)})"
            )
        return value

    @model_validator(mode="after")
    def validate_ckpt_load_settings(self):
        manual_settings_enabled = any(
            [
                self.ckpt_load_settings.init_from_ema_weights,
                self.ckpt_load_settings.restore_lr_scheduler,
                self.ckpt_load_settings.restore_time_step,
            ]
        )
        if (
            not self.ckpt_load_settings.manual_checkpoint_loading
            and manual_settings_enabled
        ):
            raise ValueError(
                "If any manual checkpoint loading settings are enabled, "
                "manual_checkpoint_loading must be set to True."
            )
        if (
            self.restart_checkpoint_path is None
            and self.ckpt_load_settings.manual_checkpoint_loading
        ):
            raise ValueError(
                "If manual_checkpoint_loading is set to True, "
                "restart_checkpoint_path must be provided."
            )

        return self


def generate_seeds(start_seed, num_seeds):
    """Helper function for generating random seeds."""
    random.seed(start_seed)
    return [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]


class InferenceExperimentSettings(ExperimentSettings):
    """General settings specific for inference experiments"""

    mode: ValidModeType = "predict"
    seeds: int | list[int] = [42]
    num_seeds: int | None = None
    use_msa_server: bool = False
    use_templates: bool = False
    skip_existing: bool = False

    @model_validator(mode="after")
    def generate_seeds(self):
        """Creates a list of seeds if a list of seeds is not provided."""
        if isinstance(self.seeds, list):
            pass
        elif isinstance(self.seeds, int):
            if self.num_seeds is None:
                raise ValueError(
                    "Attempted to generate seeds using starting"
                    f" seed {self.seeds} but num_seeds was not provided."
                    "Please either provide `num_seeds` or a list of seeds."
                )
            self.seeds = generate_seeds(self.seeds, self.num_seeds)
        elif self.seeds is None:
            raise ValueError("seeds must be provided (either int or list[int])")

        return self


class ExperimentConfig(BaseModel):
    """Base set of arguments expected for all experiments"""

    experiment_settings: ExperimentSettings
    pl_trainer_args: PlTrainerArgs = PlTrainerArgs()
    model_update: ModelUpdate


class TrainingExperimentConfig(ExperimentConfig):
    """Training experiment config"""

    # pydantic model setting to prevent extra fields in main experiment config
    model_config = PydanticConfigDict(extra="forbid")
    # required arguments for training experiment
    dataset_paths: dict[str, TrainingDatasetPaths]
    dataset_configs: dict[str, Any]

    experiment_settings: TrainingExperimentSettings = TrainingExperimentSettings()
    logging_config: LoggingConfig = LoggingConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    model_update: ModelUpdate = ModelUpdate(presets=["train"])
    data_module_args: DataModuleArgs = DataModuleArgs()

    @model_validator(mode="after")
    def synchronize_seeds(self):
        """
        Ensures data_seed in DataModuleArgs is set. If it isn't, it will
        default to the model seed.
        """
        model_seed = self.experiment_settings.seed
        data_seed = self.data_module_args.data_seed
        world_size = self.pl_trainer_args.devices * self.pl_trainer_args.num_nodes

        # TODO: Currently this will never be true because 42 is the default seed.
        #  Revisit after removing the default seed value for training and inference
        if model_seed is None and world_size > 1:
            raise ValueError("For distributed training, seed must be specified")

        if data_seed is None:
            self.data_module_args.data_seed = model_seed

        return self

    @model_validator(mode="after")
    def check_preemption_safe(self):
        """
        Checks whether preemption_safe_resume settings are valid.
        Currently, this only supports jobs that use wandb logging
        with a set id.

        It will have the following effects if set:
        1. When restarted, the run will resume from the last locally
           saved checkpoint for a given wandb id.
        2. ckpt_load_settings will be disabled if the run
           already exists and has existing checkpoints.
        3. restart_checkpoint_path will be set to "last" if the run
           already exists and has existing checkpoints.
        """
        if not self.experiment_settings.preemption_safe_resume:
            return self

        wandb_config = self.logging_config.wandb_config
        if wandb_config is None:
            raise ValueError(
                "The `preemption_safe_resume` setting currently only supports jobs "
                "run with wandb. Please provide a wandb_config."
            )
        if wandb_config.id is None:
            raise ValueError(
                "The `preemption_safe_resume` setting requires wandb_config.id to "
                "be set. This ensures that if a job is preempted, the new job resumes "
                "from the same id."
            )

        return self


class InferenceExperimentConfig(ExperimentConfig):
    """Inference experiment config"""

    # pydantic model setting to prevent extra fields in main experiment config
    model_config = PydanticConfigDict(extra="forbid")
    inference_ckpt_path: Path | None = None
    inference_ckpt_name: str | None = None

    # default location to look for parameters if no ckpt_path is given
    cache_path: Path | None = None

    experiment_settings: InferenceExperimentSettings = InferenceExperimentSettings()
    model_update: ModelUpdate = ModelUpdate(presets=["predict", "pae_enabled"])
    data_module_args: DataModuleArgs = DataModuleArgs()
    dataset_config_kwargs: InferenceDatasetConfigKwargs = InferenceDatasetConfigKwargs()
    output_writer_settings: OutputWritingSettings = OutputWritingSettings()
    msa_computation_settings: MsaComputationSettings = MsaComputationSettings()
    template_preprocessor_settings: TemplatePreprocessorSettings = (
        TemplatePreprocessorSettings(mode="predict")
    )

    @model_validator(mode="before")
    @classmethod
    def set_default_cache_path(cls, data):
        """Set default cache_path if not provided"""
        if data.get("cache_path") is None:
            cache_path = os.environ.get("OPENFOLD_CACHE") or DEFAULT_CACHE_PATH
            Path(cache_path).mkdir(parents=True, exist_ok=True)
            data["cache_path"] = cache_path
        return data

    @model_validator(mode="after")
    def validate_ckpt_settings(self):
        """Validates inference_ckpt_path and inference_ckpt name settings."""
        # Prioritize using checkpoint path when set
        if isinstance(self.inference_ckpt_path, Path):
            if self.inference_ckpt_path.exists():
                return self
            raise ValueError(
                f"Provided checkpoint path {self.inference_ckpt_path} does not exist"
            )

        elif self.inference_ckpt_name is not None:
            # validate checkpoint name is in registry
            if self.inference_ckpt_name not in OPENFOLD_MODEL_CHECKPOINT_REGISTRY:
                raise ValueError(
                    f"inference_ckpt_name {self.inference_ckpt_name} not found in "
                    "checkpoint registry. Please select from "
                    f"{list(OPENFOLD_MODEL_CHECKPOINT_REGISTRY.keys())}."
                )

            # validate checkpoint name is compatible with current version
            current_openfold3_version = Version(version("openfold3"))
            allowed_versions = SpecifierSet(
                OPENFOLD_MODEL_CHECKPOINT_REGISTRY[
                    self.inference_ckpt_name
                ].version_compatibility
            )
            if current_openfold3_version not in allowed_versions:
                raise ValueError(
                    f"Selected checkpoint {self.inference_ckpt_name} is not compatible"
                    "with the currently installed OpenFold3 version"
                    f"{current_openfold3_version}. Allowed versions for this "
                    f"checkpoint are {allowed_versions}."
                )
        else:
            logger.info(
                "No inference_ckpt_path or inference_ckpt_name provided, "
                "selecting default checkpoint."
            )
            self.inference_ckpt_name = DEFAULT_CHECKPOINT_NAME
        return self

    @model_validator(mode="after")
    def _try_default_ckpt_path(self):
        """Attempt to use and/or download default checkpoint.

        This function will:
        1) Attempt to find the checkpoints in the path specified by
           `cache_path` / `CHECKPOINT_ROOT_FILENAME`,
        2) If not found, attempt to download the specified checkpoint name
        (self.inference_ckpt_name to `cache_path` and write the checkpoint root file.
        3) Set the inference_ckpt_path to the found or downloaded checkpoint path.
        """
        # Skip ckpt selection if ckpt is previously specified
        if self.inference_ckpt_path is not None:
            return self

        param_dir = get_default_checkpoint_dir(cache_path=self.cache_path)
        path_to_ckpt = (
            param_dir
            / OPENFOLD_MODEL_CHECKPOINT_REGISTRY[self.inference_ckpt_name].file_name
        )

        if not path_to_ckpt.exists():
            download_model_parameters(param_dir, self.inference_ckpt_name)

        self.inference_ckpt_path = path_to_ckpt

        return self

    @model_validator(mode="after")
    def synchronize_seeds(self):
        """
        Ensures data_seed in DataModuleArgs is set. If it isn't, it will
        default to the first model seed in the provided list.
        """
        model_seeds = self.experiment_settings.seeds
        data_seed = self.data_module_args.data_seed

        if data_seed is None:
            self.data_module_args.data_seed = model_seeds[0]

        return self

    @model_validator(mode="after")
    def copy_ccd_file_path(self):
        """Copies ccd_file_path dataset_config_kwargs>template_preprocessor_settings."""
        if self.dataset_config_kwargs.ccd_file_path is not None:
            if self.template_preprocessor_settings.ccd_file_path is not None:
                warnings.warn(
                    "Overwriting ccd_file_path in template_preprocessor_settings with "
                    "dataset_config_kwargs.ccd_file_path. We recommend specifying"
                    "ccd_file_path only in dataset_config_kwargs.",
                    stacklevel=2,
                )
            self.template_preprocessor_settings.ccd_file_path = (
                self.dataset_config_kwargs.ccd_file_path
            )

        return self
