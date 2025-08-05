import json
import logging
import os
import shutil
import sys
import time
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path

import ml_collections as mlc
import pytorch_lightning as pl
import wandb
from lightning_fabric.utilities.rank_zero import _get_rank
from pydantic import BaseModel
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from openfold3.core.data.framework.data_module import DataModule, DataModuleConfig
from openfold3.core.runners.writer import OF3OutputWriter
from openfold3.core.utils.precision_utils import OF3DeepSpeedPrecision
from openfold3.core.utils.script_utils import set_ulimits
from openfold3.entry_points.validator import (
    ExperimentConfig,
    TrainingExperimentConfig,
)
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    InferenceDatasetSpec,
    InferenceJobConfig,
    TrainingDatasetSpec,
)
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

logger = logging.getLogger(__name__)


class ExperimentRunner(ABC):
    """Abstract class for experiments"""

    def __init__(self, experiment_config: ExperimentConfig):
        self.experiment_config = experiment_config

        self.mode = experiment_config.experiment_settings.mode
        self.pl_trainer_args = experiment_config.pl_trainer_args
        self.deepspeed_config_path = self.pl_trainer_args.deepspeed_config_path

        # typical model update config
        self.model_update = experiment_config.model_update
        self.compile = self.model_update.compile

    def setup(self) -> None:
        """Set up the experiment environment.

        This includes configuring logging, setting the random seed,
        and initializing WandB if enabled.
        """

        # Set resource limits
        set_ulimits()

    ###############
    # Model and dataset setup
    ###############
    @property
    def project_entry(self) -> OF3ProjectEntry:
        """Get the project entry from the registry."""
        return OF3ProjectEntry()

    @property
    def model_config(self) -> mlc.ConfigDict:
        """Retrieve the model configuration."""
        return self.project_entry.get_model_config_with_update(self.model_update)

    @cached_property
    def lightning_module(self) -> pl.LightningModule:
        """Instantiate and return the model."""
        return self.project_entry.runner(self.model_config, _compile=self.compile)

    @cached_property
    def output_dir(self) -> Path:
        """Get or create the output directory."""
        _out_dir = self.experiment_config.experiment_settings.output_dir
        _out_dir.mkdir(exist_ok=True, parents=True)
        return _out_dir

    @cached_property
    @abstractmethod
    def ckpt_path(self) -> str | None:
        """Get the checkpoint path for the model."""
        pass

    @property
    @abstractmethod
    def data_module_config(self) -> DataModuleConfig:
        """Construct arguments for the data_module."""
        pass

    @cached_property
    def lightning_data_module(self):
        return DataModule(
            self.data_module_config,
            world_size=self.world_size,
        )

    ###############
    # Distributed properties
    ###############
    @cached_property
    def num_gpus(self) -> int:
        """Retrieves the number of nodes available for training."""
        return self.pl_trainer_args.devices

    @cached_property
    def num_nodes(self) -> int:
        """Retrieves the number of nodes available for training."""
        return self.pl_trainer_args.num_nodes

    @property
    def world_size(self) -> int:
        """Compute the world size based on GPUs and nodes."""
        return self.num_gpus * self.num_nodes

    @property
    def is_distributed(self) -> bool:
        """Check if the training is distributed using the world size."""
        return self.world_size > 1

    @property
    def is_mpi(self) -> bool:
        """Check if MPI plugin is enabled."""
        return self.pl_trainer_args.mpi_plugin

    @property
    def is_rank_zero(self) -> bool:
        """Check if the current process is rank zero in an MPI environment."""
        if self.is_mpi:
            return self.cluster_environment.global_rank() == 0
        else:
            return _get_rank() == 0

    @property
    def cluster_environment(self) -> MPIEnvironment | None:
        """Return the MPI cluster environment if enabled."""
        return MPIEnvironment() if self.is_mpi else None

    @cached_property
    def strategy(self) -> DDPStrategy | DeepSpeedStrategy | str:
        """Determine and return the training strategy."""
        if self.deepspeed_config_path is not None:
            _strategy = DeepSpeedStrategy(
                config=self.deepspeed_config_path,
                cluster_environment=self.cluster_environment,
                precision_plugin=OF3DeepSpeedPrecision(
                    precision=self.pl_trainer_args.precision
                ),
            )

            _use_deepspeed_adam = (
                self.model_config.settings.optimizer.use_deepspeed_adam
            )
            if not _use_deepspeed_adam:
                _strategy.config["zero_force_ds_cpu_optimizer"] = False

            return _strategy

        if self.is_distributed:
            return DDPStrategy(
                find_unused_parameters=False,
                cluster_environment=self.cluster_environment,
            )

        return "auto"

    ###############
    # Logging and Callbacks
    ###############

    @cached_property
    def callbacks(self):
        """Set up and return the list of training callbacks."""
        _callbacks = []
        return _callbacks

    @cached_property
    def loggers(self):
        """Retrieve the list of loggers to be used in the experiment."""
        _loggers = []
        return _loggers

    ###############
    # pl.Trainer class and run command
    ###############

    @cached_property
    def trainer(self) -> pl.Trainer:
        """Create and return the trainer instance."""
        trainer_args = self.pl_trainer_args.model_dump(
            exclude={"deepspeed_config_path", "mpi_plugin"}
        )
        trainer_args.update(
            {
                "default_root_dir": self.output_dir,
                "strategy": self.strategy,
                "callbacks": self.callbacks,
                "logger": self.loggers,
                # If DeepSpeed is enabled, these values will be passed to the DS config
                "gradient_clip_val": self.model_config.settings.gradient_clipping,
                "gradient_clip_algorithm": "norm",
            }
        )

        return pl.Trainer(**trainer_args)

    def run(self):
        """Run the experiment in the specified mode.

        Depending on the mode (train, eval, test, predict), the corresponding
        PyTorch Lightning method is invoked.
        """
        # Run process appropriate process
        logger.info(f"Running {self.mode} mode.")
        # Training + validation
        if self.mode == "train":
            target_method = self.trainer.fit
        elif self.mode == "profile":
            raise NotImplementedError("Profiling mode not yet implemented.")
        elif self.mode == "eval":
            target_method = self.trainer.validate
        elif self.mode == "test":
            target_method = self.trainer.test
        elif self.mode == "predict":
            target_method = self.trainer.predict
        else:
            raise ValueError(
                f"""Invalid mode argument: {self.mode}. Choose one of "
                "'train', 'test', 'predict', 'profile'."""
            )

        target_method(
            model=self.lightning_module,
            datamodule=self.lightning_data_module,
            ckpt_path=self.ckpt_path,
        )


class TrainingExperimentRunner(ExperimentRunner):
    """Training experiment builder."""

    def __init__(self, experiment_config: TrainingExperimentConfig):
        super().__init__(experiment_config)

        self.seed = experiment_config.experiment_settings.seed
        self.restart_checkpoint_path = (
            experiment_config.experiment_settings.restart_checkpoint_path
        )
        self.dataset_paths = experiment_config.dataset_paths
        self.dataset_configs = experiment_config.dataset_configs
        self.data_module_args = experiment_config.data_module_args
        self.logging_config = experiment_config.logging_config
        self.checkpoint_config = experiment_config.checkpoint_config

    def setup(self) -> None:
        """Set up the experiment environment.

        This includes configuring logging, setting the random seed,
        and initializing WandB if enabled.
        """
        super().setup()
        self._setup_logger()
        self._set_random_seed()
        if self.use_wandb:
            self._wandb_setup()

    @cached_property
    def data_module_config(self) -> DataModuleConfig:
        """Make a DataModuleConfig from self.dataset_paths and self.dataset_configs."""
        cfgs = []
        for mode, ds_specs in self.dataset_configs.items():
            for name, spec in ds_specs.items():
                spec["name"] = name
                spec["mode"] = mode
                spec["config"]["dataset_paths"] = self.dataset_paths[name]

                cfgs.append(TrainingDatasetSpec.model_validate(spec))

        return DataModuleConfig(datasets=cfgs, **self.data_module_args.model_dump())

    @cached_property
    def ckpt_path(self) -> Path | None:
        return self.restart_checkpoint_path

    @property
    def use_wandb(self):
        """Determine if WandB should be used.

        Returns:
            True if WandB configuration is provided and is rank zero
        """
        return self.logging_config.wandb_config and self.is_rank_zero

    def _wandb_setup(self) -> None:
        """Initialize WandB logging and store configuration files."""
        self.wandb = WandbHandler(
            self.logging_config.wandb_config,
            self.is_rank_zero,
            self.output_dir,
        )
        self.wandb.store_configs(
            self.experiment_config,
            self.data_module_config,
            self.model_config,
        )

    def _setup_logger(self) -> None:
        """Configure the logging settings.

        Sets the log level and log file path based on runner arguments.
        """
        log_level = self.logging_config.log_level
        if log_level is None:
            return

        log_level = log_level.upper()
        log_filepath = self.output_dir / "console_logs.log"
        logging.basicConfig(filename=log_filepath, level=log_level, filemode="w")

    def _set_random_seed(self) -> None:
        """Set the random seed for reproducibility."""

        seed = self.seed
        if seed is None and self.is_distributed:
            raise ValueError("For distributed training, seed must be specified")

        if not isinstance(seed, int):
            raise ValueError(
                f"seed={seed} must be an integer. Please provide a valid seed."
            )

        logger.info(f"Running with seed: {seed}")
        pl.seed_everything(seed, workers=True)

    @cached_property
    def loggers(self):
        """Retrieve the list of loggers to be used in the experiment."""
        _loggers = []
        if self.use_wandb:
            _loggers.append(self.wandb.logger)
        return _loggers

    @cached_property
    def callbacks(self):
        """Set up and return the list of training callbacks."""
        _callbacks = []

        _checkpoint = self.checkpoint_config
        if _checkpoint is not None:
            _callbacks.append(ModelCheckpoint(**_checkpoint.model_dump()))

        _log_lr = self.logging_config.log_lr
        if _log_lr and self.use_wandb:
            _callbacks.append(LearningRateMonitor(logging_interval="step"))

        return _callbacks


class InferenceExperimentRunner(ExperimentRunner):
    """Training experiment builder."""

    def __init__(self, experiment_config):
        super().__init__(experiment_config)

        self.experiment_config = experiment_config

        self.dataset_config_kwargs = experiment_config.dataset_config_kwargs
        self.inference_ckpt_path = experiment_config.inference_ckpt_path
        self.data_module_args = experiment_config.data_module_args
        self.seeds = experiment_config.experiment_settings.seeds
        self.output_writer_settings = experiment_config.output_writer_settings
        self.timer = ExperimentTimer()

    def run(self, inference_query_set) -> None:
        """Set up the experiment environment."""
        self.timer.start("Inference")
        self.inference_query_set = inference_query_set
        self._log_inference_query_set()
        super().run()
        self.timer.stop()
        print(f"Inference Runtime: {self.timer.get('Inference')}")

    @cached_property
    def callbacks(self):
        """Set up prediction writer callback."""
        _callbacks = [
            OF3OutputWriter(self.output_dir, **self.output_writer_settings.model_dump())
        ]
        return _callbacks

    @cached_property
    def data_module_config(self):
        inference_config = InferenceJobConfig(
            query_set=self.inference_query_set,
            seeds=self.seeds,
            ccd_file_path=self.dataset_config_kwargs.ccd_file_path,
            msa=self.dataset_config_kwargs.msa,
            template=self.dataset_config_kwargs.template,
            template_preprocessor=self.experiment_config.template_preprocessor_settings,
        )
        inference_spec = InferenceDatasetSpec(config=inference_config)
        return DataModuleConfig(
            datasets=[inference_spec], **self.data_module_args.model_dump()
        )

    @cached_property
    def ckpt_path(self):
        """Get the checkpoint path for the model."""
        return self.inference_ckpt_path

    def _log_inference_query_set(self):
        """Record the inference query set used for prediction"""
        log_path = self.output_dir / "inference_query_set.json"
        with open(log_path, "w") as fp:
            fp.write(self.inference_query_set.model_dump_json(indent=4))

    def cleanup(self):
        if self.experiment_config.msa_computation_settings.cleanup_msa_dir:
            output_dir = (
                self.experiment_config.msa_computation_settings.msa_output_directory
            )
            logger.info(f"Removing MSA output directory: {output_dir}")
            shutil.rmtree(output_dir)


class WandbHandler:
    """Handles WandB logger initialization and configuration storage.

    This class is responsible for setting up the WandB logger and saving
    the experiment configurations to WandB.
    """

    def __init__(
        self,
        wandb_args: BaseModel | None,
        is_rank_zero: bool,
        output_dir: Path,
    ):
        """Initialize the WandbHandler.

        Args:
            wandb_args: The WandB related configuration.
            is_rank_zero: True if the current process is rank zero.
            output_dir: The directory to store WandB files.
        """
        self.wandb_args = wandb_args
        self.output_dir = output_dir
        self.is_rank_zero = is_rank_zero
        self._logger = None

    def _init_logger(self) -> None:
        """Initialize the wandb environment and create the WandbLogger."""
        if self.wandb_args is None:
            raise ValueError("wandb_args must be provided to use wandb logger")

        wandb_init_dict = dict(
            project=self.wandb_args.project,
            entity=self.wandb_args.entity,
            group=self.wandb_args.group,
            name=self.wandb_args.experiment_name,
            dir=self.output_dir,
            resume="allow",
            reinit=True,
            id=self.wandb_args.id,
        )

        # Only initialize wandb for rank zero worker
        # each worker will generate a different id
        if self.is_rank_zero:
            wandb.run = wandb.init(**wandb_init_dict)

        self._logger = WandbLogger(
            **wandb_init_dict,
            save_dir=self.output_dir,
            log_model=False,
        )

    @property
    def logger(self) -> WandbLogger:
        """Return the WandB logger instance. The logger is initialized
        on first access."""
        if self._logger is None:
            self._init_logger()
        assert self._logger is not None
        return self._logger

    def store_configs(
        self,
        runner_args: TrainingExperimentConfig,
        data_module_config: DataModuleConfig,
        model_config: mlc.ConfigDict,
    ) -> None:
        """Store experiment configuration files to the WandB run directory.

        This method saves the pip freeze output, runner configuration,
        data module configuration, and model configuration as files in
        the WandB run.

        Args:
            runner_args: The runner configuration.
            data_module_config: The configuration for the data module.
            model_config: The configuration for the model.
        """

        wandb_experiment = self.logger.experiment
        # Save pip environment to wandb

        freeze_path = os.path.join(wandb_experiment.dir, "package_versions.txt")
        os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
        wandb_experiment.save(f"{freeze_path}")

        # user given runner yaml
        runner_yaml_path = os.path.join(wandb_experiment.dir, "runner.json")
        with open(runner_yaml_path, "w") as fp:
            fp.write(runner_args.model_dump_json(indent=4))
        wandb_experiment.save(runner_yaml_path)

        # save the deepspeed config if it exists
        if runner_args.pl_trainer_args.deepspeed_config_path:
            wandb_experiment.save(runner_args.pl_trainer_args.deepspeed_config_path)

        # Save data module config
        data_config_path = os.path.join(wandb_experiment.dir, "data_config.json")
        with open(data_config_path, "w") as fp:
            fp.write(data_module_config.model_dump_json(indent=4))
        wandb_experiment.save(data_config_path)

        # Save model config
        model_config_path = os.path.join(wandb_experiment.dir, "model_config.json")
        with open(model_config_path, "w") as fp:
            json.dump(model_config.to_dict(), fp, indent=4)
        wandb_experiment.save(model_config_path)


class ExperimentTimer:
    """Timer class that can be used to time different parts of the experiment."""

    def __init__(self):
        self.start_time = None
        self.elapsed = {}

    def start(self, label: str = "total"):
        self.start_time = time.time()
        self._label = label

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        duration = time.time() - self.start_time
        self.elapsed[self._label] = self.elapsed.get(self._label, 0.0) + duration
        self.start_time = None
        return duration

    def get(self, label: str = "total"):
        return self.elapsed.get(label, 0.0)

    def summary(self):
        return {k: round(v, 2) for k, v in self.elapsed.items()}
