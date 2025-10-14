import random
import warnings
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from pydantic import BaseModel, field_validator, model_validator
from pydantic import ConfigDict as PydanticConfigDict

from openfold3.core.data.pipelines.preprocessing.template import (
    TemplatePreprocessorSettings,
)
from openfold3.core.data.tools.colabfold_msa_server import MsaComputationSettings
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    InferenceDatasetConfigKwargs,
    TrainingDatasetPaths,
)
from openfold3.projects.of3_all_atom.project_entry import ModelUpdate

ValidModeType = Literal["train", "predict", "eval", "test"]


class CheckpointConfig(BaseModel):
    """Settings for training checkpoint writing."""

    every_n_epochs: int = 1
    auto_insert_metric_name: bool = False
    save_last: bool = True
    save_top_k: int = -1


class WandbConfig(BaseModel):
    """Configuration for Weights and Biases experiment result logging."""

    project: str
    experiment_name: str
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

    # Extra arguments that are not passed directly to pl.Trainer
    deepspeed_config_path: Path | None = None
    distributed_timeout: timedelta | None = default_pg_timeout
    mpi_plugin: bool = False


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
    def validate_ckpt_load_settings(cls, model):
        manual_settings_enabled = any(
            [
                model.ckpt_load_settings.init_from_ema_weights,
                model.ckpt_load_settings.restore_lr_scheduler,
                model.ckpt_load_settings.restore_time_step,
            ]
        )
        if (
            not model.ckpt_load_settings.manual_checkpoint_loading
            and manual_settings_enabled
        ):
            raise ValueError(
                "If any manual checkpoint loading settings are enabled, "
                "manual_checkpoint_loading must be set to True."
            )
        if (
            model.restart_checkpoint_path is None
            and model.ckpt_load_settings.manual_checkpoint_loading
        ):
            raise ValueError(
                "If manual_checkpoint_loading is set to True, "
                "restart_checkpoint_path must be provided."
            )

        return model


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

        if data_seed is None:
            self.data_module_args.data_seed = model_seed

        return self


class InferenceExperimentConfig(ExperimentConfig):
    """Inference experiment config"""

    # pydantic model setting to prevent extra fields in main experiment config
    model_config = PydanticConfigDict(extra="forbid")
    # Required inputs for performing inference
    inference_ckpt_path: Path

    experiment_settings: InferenceExperimentSettings = InferenceExperimentSettings()
    model_update: ModelUpdate = ModelUpdate(presets=["predict", "pae_enabled"])
    data_module_args: DataModuleArgs = DataModuleArgs()
    dataset_config_kwargs: InferenceDatasetConfigKwargs = InferenceDatasetConfigKwargs()
    output_writer_settings: OutputWritingSettings = OutputWritingSettings()
    msa_computation_settings: MsaComputationSettings = MsaComputationSettings()
    template_preprocessor: TemplatePreprocessorSettings = TemplatePreprocessorSettings(
        mode="predict"
    )

    @model_validator(mode="after")
    def synchronize_seeds(cls, model):
        """
        Ensures data_seed in DataModuleArgs is set. If it isn't, it will
        default to the first model seed in the provided list.
        """
        model_seeds = model.experiment_settings.seeds
        data_seed = model.data_module_args.data_seed

        if data_seed is None:
            model.data_module_args.data_seed = model_seeds[0]

        return model

    @model_validator(mode="after")
    def copy_ccd_file_path(self):
        """Copies ccd_file_path dataset_config_kwargs>template_preprocessor_settings."""
        if self.ccd_file_path is not None:
            if self.template_preprocessor_settings.ccd_file_path is not None:
                warnings.warn(
                    "Overwriting ccd_file_path in template_preprocessor_settings with "
                    "dataset_config_kwargs.ccd_file_path. We recommend specifying"
                    "ccd_file_path only in dataset_config_kwargs.",
                    stacklevel=2,
                )
            self.template_preprocessor_settings.ccd_file_path = self.ccd_file_path

        return self
