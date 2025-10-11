import logging
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

import boto3
import botocore
from botocore.config import Config as botocoreConfig
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from pydantic import BaseModel, field_validator, model_validator
from pydantic import ConfigDict as PydanticConfigDict

from openfold3.core.config.config_utils import FilePathOrNone
from openfold3.core.data.tools.colabfold_msa_server import MsaComputationSettings
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    InferenceDatasetConfigKwargs,
    TrainingDatasetPaths,
)
from openfold3.projects.of3_all_atom.project_entry import ModelUpdate

# ruff: noqa: I001
from packaging import version  # noqa: E402
import gemmi  # noqa: E402

if version.parse(gemmi.__version__) >= version.parse("0.7.3"):
    gemmi.set_leak_warnings(False)


logger = logging.getLogger(__name__)

ValidModeType = Literal["train", "predict", "eval", "test"]
DEFAULT_CACHE_PATH = Path("~/.openfold3/").expanduser()
CHECKPOINT_NAME = "of3_ft3_v1.pt"


def get_openfold_cache_dir() -> Path:
    """Identifies the default cache directory.
    Prioritizes $OPENFOLD3_CACHE_DIR, then ~/.openfold3."""
    cache_path = os.environ.get("OPENFOLD3_CACHE")
    if cache_path is None:
        cache_path = DEFAULT_CACHE_PATH
    return Path(cache_path)


def _maybe_download_parameters(target_path: Path):
    """Checks if the openfold3 model parametrs """
    openfold_bucket = "openfold"
    checkpoint_path = f"openfold3_params/{CHECKPOINT_NAME}"

    if target_path.exists():

        return

    s3 = boto3.client('s3', config=botocoreConfig(signature_version=botocore.UNSIGNED))

    try:
        # Get file size
        response = s3.head_object(Bucket=openfold_bucket, Key=checkpoint_path)
        size_bytes = response['ContentLength']
        size_gb = size_bytes / (1024 ** 3)
        
        # Ask for confirmation with file size
        confirm = input(
            f"Download {checkpoint_path} ({size_gb:.2f} GB) "
            f"from s3://{openfold_bucket}? (yes/no): "
        )
        
        if confirm.lower() in ['yes', 'y']:
            logger.info(f"Downloading to {target_path}...")
            s3.download_file(openfold_bucket, checkpoint_path, target_path)
            logger.info("Download complete!")
        else:
            logger.warning("Download cancelled")
            
    except Exception as e:
        print(f"Error: {e}")


class CheckpointConfig(BaseModel):
    """Settings for training checkpoint writing."""

    every_n_epochs: int = 1
    auto_insert_metric_name: bool = False
    save_last: bool = True
    save_top_k: int = -1


class WandbConfig(BaseModel):
    """Configuration for Weights and Biases experiment result logging."""

    project: str = "my project"
    experiment_name: str = "expt_name"
    entity: str | None = None
    group: str | None = None
    id: str | None = None
    offline: bool = False


class LoggingConfig(BaseModel):
    """Settings for training logging."""

    log_lr: bool = True
    log_level: Literal["debug", "info", "warning", "error"] | None = None
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


class TrainingExperimentSettings(ExperimentSettings):
    """General settings specific for training experiments"""

    mode: ValidModeType = "train"
    seed: int = 42
    restart_checkpoint_path: FilePathOrNone = None


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
                    "num_seeds must be provided when seeds is a single int"
                )
            generate_seeds(self.seeds, self.num_seeds)
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


class InferenceExperimentConfig(ExperimentConfig):
    """Inference experiment config"""

    # pydantic model setting to prevent extra fields in main experiment config
    model_config = PydanticConfigDict(extra="forbid")
    # Required inputs for performing inference
    inference_ckpt_path: Path | None = None

    experiment_settings: InferenceExperimentSettings = InferenceExperimentSettings()
    model_update: ModelUpdate = ModelUpdate(presets=["predict", "pae_enabled"])
    data_module_args: DataModuleArgs = DataModuleArgs()
    dataset_config_kwargs: InferenceDatasetConfigKwargs = InferenceDatasetConfigKwargs()
    output_writer_settings: OutputWritingSettings = OutputWritingSettings()
    msa_computation_settings: MsaComputationSettings = MsaComputationSettings()

    @field_validator("inference_ckpt_path", mode="before")
    def _try_default_ckpt_path(cls, value):
        if value is None:
            value = get_openfold_cache_dir() / CHECKPOINT_NAME 
            _maybe_download_parameters(value)
        return value
