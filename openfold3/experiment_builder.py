import json
import logging
import os
import sys
from functools import cached_property
from pathlib import Path

import pytorch_lightning as pl
import wandb
from ml_collections import ConfigDict
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from openfold3.core.config import config_utils, registry_base
from openfold3.core.data.framework.data_module import DataModule, DataModuleConfig
from openfold3.projects import registry
from openfold3.projects.af3_all_atom.config.runner_file_checks import (
    _check_data_module_config,
)

logger = logging.getLogger(__name__)


class ExperimentBuilder:
    def __init__(self, runner_yaml: str | Path) -> None:
        self.runner_args = ConfigDict(config_utils.load_yaml(runner_yaml))

    def setup(self) -> None:
        self._setup_logger()
        self._set_random_seed()
        if self.use_wandb:
            self._wandb_setup()

    def _wandb_setup(self) -> None:
        self.wandb = WandbHandler(
            self.runner_args.wandb,
            self.is_mpi_rank_zero,
            self.output_dir,
        )
        self.wandb.store_configs(
            self.runner_args,
            self.data_module_config,
            self.model_config,
        )

    def _setup_logger(self) -> None:
        log_level = self.runner_args.get("log_level")
        if log_level is None:
            return

        VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if not isinstance(log_level, str) or log_level.upper() not in VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {VALID_LOG_LEVELS}")

        log_level = log_level.upper()
        log_filepath = self.output_dir / "console_logs.log"
        logging.basicConfig(filename=log_filepath, level=log_level, filemode="w")

    def _set_random_seed(self) -> None:
        seed = self.runner_args.get("seed")
        if seed is None and self.is_distributed:
            raise ValueError("For distributed training, seed must be specified")

        if not isinstance(seed, int):
            raise ValueError(
                f"seed={seed} must be an integer. Please provide a valid seed."
            )

        logging.info(f"Running with seed: {seed}")
        pl.seed_everything(seed, workers=True)

    def run(self):
        mode = self.runner_args.mode
        # Run process appropriate process
        logging.info(f"Running {mode} mode.")
        # Training + validation
        if mode == "train":
            target_method = self.trainer.fit
        elif mode == "profile":
            raise NotImplementedError("Profiling mode not yet implemented.")
        elif mode == "eval":
            target_method = self.trainer.validate
        elif mode == "test":
            target_method = self.trainer.test
        elif mode == "predict":
            target_method = self.trainer.predict
        else:
            raise ValueError(
                f"""Invalid mode argument: {mode}. Choose one of "
                "'train', 'test', 'predict', 'profile'."""
            )

        target_method(
            model=self.lightning_module,
            datamodule=self.lightning_data_module,
            ckpt_path=self.ckpt_path,
        )

    @property
    def use_wandb(self):
        return self.runner_args.wandb and (not self.is_mpi or self.is_mpi_rank_zero)

    @cached_property
    def output_dir(self) -> Path:
        output_dir = self.runner_args.get("output_dir")
        if output_dir is None:
            raise ValueError("output_dir must be specified in runner yaml")

        if not isinstance(output_dir, str):
            raise ValueError(f"output_dir={output_dir} must be a string")
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    @property
    def world_size(self) -> int:
        return self.runner_args.num_gpus * self.runner_args.pl_trainer.num_nodes

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def project_entry(self) -> registry_base.ProjectEntry:
        return registry.get_project_entry(self.runner_args.project_type)

    @cached_property
    def project_config(self) -> ConfigDict:
        presets = self.runner_args.presets
        config_update = self.runner_args.get("config_update")

        project_config = registry.make_config_with_presets(self.project_entry, presets)
        if config_update is not None:
            project_config.update(config_update)

        return project_config

    @property
    def ckpt_path(self) -> str | None:
        _ckpt_path = self.runner_args.get("restart_checkpoint_path")
        if _ckpt_path is not None and not isinstance(_ckpt_path, str):
            raise ValueError("restart_checkpoint_path must be a string.")

        return _ckpt_path

    @property
    def model_config(self) -> ConfigDict:
        return self.project_config.model

    @cached_property
    def lightning_module(self) -> pl.LightningModule:
        return self.project_entry.model_runner(
            self.model_config, _compile=self.runner_args.compile
        )

    @cached_property
    def data_module_config(self) -> DataModuleConfig:
        dataset_config_builder = self.project_entry.dataset_config_builder
        return registry.make_dataset_module_config(
            self.runner_args,
            dataset_config_builder,
            self.project_config,
        )

    @cached_property
    def lightning_data_module(self) -> DataModule:
        _check_data_module_config(self.data_module_config)
        return DataModule(self.data_module_config, world_size=self.world_size)

    @property
    def is_mpi(self) -> bool:
        return self.runner_args.get("mpi_plugin")

    @property
    def cluster_environment(self) -> MPIEnvironment | None:
        return MPIEnvironment() if self.is_mpi else None

    @cached_property
    def use_deepspeed_adam(self) -> bool:
        return self.model_config.settings.optimizer.use_deepspeed_adam

    @cached_property
    def strategy(self) -> DDPStrategy | DeepSpeedStrategy | None:
        deepspeed_config_path = self.runner_args.get("deepspeed_config_path")
        if deepspeed_config_path is not None:
            _strategy = DeepSpeedStrategy(
                config=deepspeed_config_path,
                cluster_environment=self.cluster_environment,
            )
            if not self.use_deepspeed_adam:
                _strategy.config["zero_force_ds_cpu_optimizer"] = False

            return _strategy

        if self.is_distributed:
            return DDPStrategy(
                find_unused_parameters=False,
                cluster_environment=self.cluster_environment,
            )

        return None

    @property
    def is_mpi_rank_zero(self) -> bool:
        return self.is_mpi and self.cluster_environment.global_rank() == 0

    @cached_property
    def loggers(self):
        _loggers = []
        if self.use_wandb:
            _loggers.append(self.wandb.logger)
        return _loggers

    @cached_property
    def callbacks(self):
        _callbacks = []

        _checkpoint = self.runner_args.get("checkpoint")
        if _checkpoint is not None:
            _callbacks.append(ModelCheckpoint(**_checkpoint.to_dict()))

        _log_lr = self.runner_args.get("log_lr")
        if _log_lr is not None and self.use_wandb:
            _callbacks.append(LearningRateMonitor(logging_interval="step"))

        return _callbacks

    @property
    def gradient_clip_val(self) -> float:
        return self.model_config.settings.gradient_clipping

    @cached_property
    def trainer(self) -> pl.Trainer:
        trainer_args = self.runner_args.pl_trainer.to_dict()
        trainer_args.update(
            {
                "default_root_dir": self.output_dir,
                "strategy": self.strategy,
                "callbacks": self.callbacks,
                "logger": self.loggers,
                "devices": self.runner_args.num_gpus,
                # If DeepSpeed is enabled, these values will be passed to the DS config
                "gradient_clip_val": self.gradient_clip_val,
                "gradient_clip_algorithm": "norm",
            }
        )

        return pl.Trainer(**trainer_args)


class WandbHandler:
    def __init__(
        self,
        wandb_args: None | ConfigDict,
        is_mpi_rank_zero: bool,
        output_dir: Path,
    ):
        self.wandb_args = wandb_args
        self.output_dir = output_dir
        self.is_mpi_rank_zero = is_mpi_rank_zero

    def _init_logger(self) -> None:
        """Configures wandb and wandb logger."""
        if self.wandb_args is None:
            raise ValueError("wandb_args must be provided to use wandb logger")

        wandb_id = self.wandb_args.id if hasattr(self.wandb_args, "id") else None

        wandb_init_dict = dict(
            project=self.wandb_args.project,
            entity=self.wandb_args.entity,
            group=self.wandb_args.group,
            name=self.wandb_args.experiment_name,
            dir=self.output_dir,
            resume="allow",
            reinit=True,
            id=wandb_id,
        )

        # Only initialize wandb for rank zero worker (MPI env), or else
        # each worker will generate a different id
        if self.is_mpi_rank_zero:
            wandb.run = wandb.init(**wandb_init_dict)

        run_offline = self.wandb_args.get("offline", False)
        self._logger = WandbLogger(
            **wandb_init_dict,
            save_dir=self.output_dir,
            log_model=False,
            offline=run_offline,
        )

    @property
    def logger(self) -> WandbLogger:
        if not hasattr(self, "_logger"):
            self._init_logger()
        return self._logger

    def store_configs(
        self,
        runner_args: ConfigDict,
        data_module_config: DataModuleConfig,
        model_config: ConfigDict,
    ) -> None:
        wandb_experiment = self.logger.experiment
        # Save pip environment to wandb
        freeze_path = os.path.join(wandb_experiment.dir, "package_versions.txt")
        os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
        wandb_experiment.save(f"{freeze_path}")

        # user given runner yaml
        runner_yaml_path = os.path.join(wandb_experiment.dir, "runner.json")
        with open(runner_yaml_path, "w") as fp:
            json.dump(runner_args.to_dict(), fp, indent=4)
        wandb_experiment.save(runner_yaml_path)

        # save the deepspeed config if it exists
        if runner_args.get("deepspeed_config_path"):
            wandb_experiment.save(runner_args.deepspeed_config_path)

        # Save data module config
        data_config_path = os.path.join(wandb_experiment.dir, "data_config.json")
        with open(data_config_path, "w") as fp:
            json.dump(data_module_config.to_dict(), fp, indent=4)
        wandb_experiment.save(data_config_path)

        # Save model config
        model_config_path = os.path.join(wandb_experiment.dir, "model_config.json")
        with open(model_config_path, "w") as fp:
            json.dump(model_config.to_dict(), fp, indent=4)
        wandb_experiment.save(model_config_path)
