from typing import Any, Optional

from pydantic import BaseModel


class CheckpointConfig(BaseModel):
    every_n_epochs: int
    auto_insert_metric_name: bool
    save_last: bool
    save_top_k: int


class WandbConfig(BaseModel):
    project: str
    entity: Optional[str] = None
    group: Optional[str] = None
    id: Optional[str] = None
    experiment_name: str
    offline: bool = False


class TrainableExperimentConfig(BaseModel):
    mode: str
    project_type: str
    presets: list[str]
    mpi_plugin: bool
    seed: int
    batch_size: int
    num_gpus: int
    output_dir: str
    log_lr: bool
    compile: bool
    num_workers: int
    epoch_len: int
    restart_checkpoint_path: str
    deepspeed_config_path: str | None = None
    checkpoint: CheckpointConfig | None = None
    wandb: WandbConfig | None = None

    pl_trainer: dict[str, Any]
    dataset_paths: dict[str, Any]
    config_update: dict[str, Any]
    dataset_configs: dict[str, Any]
