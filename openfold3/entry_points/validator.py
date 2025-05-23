from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel
from pydantic import ConfigDict as PydanticConfigDict

from openfold3.core.config.path_definitions import FilePathOrNone
from openfold3.projects.af3_all_atom.config.dataset_configs import (
    InferenceDatasetConfigKwargs,
    TrainingDatasetPaths,
)
from openfold3.projects.af3_all_atom.project_entry import ModelUpdate


class CheckpointConfig(BaseModel):
    every_n_epochs: int = 1
    auto_insert_metric_name: bool = False
    save_last: bool = True
    save_top_k: int = -1


class WandbConfig(BaseModel):
    project: str = "my project"
    experiment_name: str = "expt_name"
    entity: Optional[str] = None
    group: Optional[str] = None
    id: Optional[str] = None
    offline: bool = False


class LoggingConfig(BaseModel):
    log_lr: bool = True
    log_level: str | None = None
    wandb_config: WandbConfig | None = None


class DataModuleArgs(BaseModel):
    """Settings for openfold3.core.data.framework.data_module"""

    model_config = PydanticConfigDict(extra="forbid")
    batch_size: int = 1
    data_seed: int = 1234
    num_workers: int = 10
    epoch_len: int = 4
    num_epochs: int = 1000


class PlTrainerArgs(BaseModel):
    """Arguments to configure pl.Trainer"""

    max_epochs: int = 1000  # pl_trainer default
    accelerator: str = "gpu"
    precision: str = "bf16-mixed"
    num_nodes: int = 1
    devices: int = 1  # number of GPUs per node
    profiler: Optional[str] = None
    log_every_n_steps: int = 1
    enable_checkpointing: bool = True
    enable_model_summary: bool = False

    # Extra arguments that are not passed directly to pl.Trainer
    deepspeed_config_path: Path | None = None
    mpi_plugin: bool = False


class ExperimentSettings(BaseModel):
    """General settings for all experiments"""

    mode: str  # must be train or predict
    output_dir: Path


class TrainingExperimentSettings(ExperimentSettings):
    """General settings specific for training experiments"""

    seed: int = 42
    restart_checkpoint_path: FilePathOrNone = None


class InferenceExperimentSettings(ExperimentSettings):
    """General settings specific for training experiments"""

    query_json: Path
    seeds: list[int] = [42]
    inference_ckpt_path: Path


class ExperimentConfig(BaseModel):
    """Base set of arguments expected for all experiments"""

    experiment_settings: ExperimentSettings
    pl_trainer_args: PlTrainerArgs
    model_update: ModelUpdate


class TrainingExperimentConfig(ExperimentConfig):
    """Training experiment config"""

    experiment_settings: TrainingExperimentSettings
    logging_config: LoggingConfig = LoggingConfig()
    checkpoint_config: CheckpointConfig = CheckpointConfig()
    model_update: ModelUpdate = ModelUpdate(presets=["train"])
    dataset_paths: dict[str, TrainingDatasetPaths]
    dataset_configs: dict[str, Any]
    data_module_args: DataModuleArgs = DataModuleArgs()


class InferenceExperimentConfig(ExperimentConfig):
    """Inference experiment config"""

    # TODO: Add MSA configuration settings
    experiment_settings: InferenceExperimentSettings
    model_update: ModelUpdate = ModelUpdate(presets=["predict"])
    data_module_args: DataModuleArgs = DataModuleArgs()
    dataset_config_kwargs: InferenceDatasetConfigKwargs = InferenceDatasetConfigKwargs()
