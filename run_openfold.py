# args TODO add license

import argparse
import logging
import os
import sys

import pytorch_lightning as pl
from ml_collections import ConfigDict
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

import wandb
from openfold3.core.config import config_utils
from openfold3.core.data.framework.data_module import DataModule
from openfold3.projects import registry


def _configure_wandb_logger(
    wandb_args: ConfigDict, is_rank_zero: bool, output_dir: str
) -> WandbLogger:
    """Configures wandb and wandb logger."""

    def _maybe_get_wandb_id():
        if is_rank_zero:
            if hasattr(wandb_args, "id"):
                return wandb_args.id
            else:
                return wandb.util.generate_id()
        return None

    wandb_id = _maybe_get_wandb_id()

    wandb_init_dict = dict(
        project=wandb_args.project,
        entity=wandb_args.entity,
        group=wandb_args.group,
        name=wandb_args.experiment_name,
        dir=output_dir,
        save_dir=output_dir,
        resume="allow",
        reinit=True,
        id=wandb_id,
    )

    # Only initialize wandb for rank zero worker, or else
    # each worker will generate a different id
    if is_rank_zero:
        wandb.init(**wandb_init_dict)

    wandb_logger = WandbLogger(**wandb_init_dict, log_model=False)
    return wandb_logger


def main(args):
    runner_args = ConfigDict(config_utils.load_yaml(args.runner_yaml))

    # Set seed
    if runner_args.get("seed"):
        pl.seed_everything(runner_args.seed, workers=True)

    project_entry = registry.get_project_entry(runner_args.project_type)

    project_config = registry.make_config_with_presets(
        project_entry, runner_args.presets
    )
    if runner_args.get("config_update"):
        project_config.update(runner_args.config_update)

    # TODO: Implement checkpoint reloading logic
    # Be sure to call runner.resume_last_lr_step to set the lr_step
    if runner_args.get("restart_checkpoint_path"):
        raise ValueError("Restarting from checkpoints is currently not supported")

    model_config = project_config.model
    lightning_module = project_entry.model_runner(
        model_config, _compile=runner_args.compile
    )

    dataset_config_builder = project_entry.dataset_config_builder
    data_module_config = registry.make_dataset_module_config(
        runner_args,
        dataset_config_builder,
        project_config,
    )
    lightning_data_module = DataModule(data_module_config)

    # Set up trainer arguments and callbacks
    callbacks = []

    if runner_args.get("checkpoint_every_epoch"):
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=1,
                auto_insert_metric_name=False,
                save_top_k=-1,
            )
        )

    if runner_args.get("log_lr"):
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    loggers = []
    if runner_args.get("mpi_plugin") and os.environ.get("PMI_RANK") is None:
        raise ValueError(
            "PMI_RANK is not set as an environment variable, find another way to specify rank."
        )
    IS_RANK_ZERO = runner_args.get("mpi_plugin") and (
        int(os.environ.get("PMI_RANK")) == 0
    )

    if runner_args.get("wandb"):
        wandb_logger = _configure_wandb_logger(
            runner_args.wandb, IS_RANK_ZERO, runner_args.output_dir
        )
        loggers.append(wandb_logger)
        if IS_RANK_ZERO:
            # Save pip environment to wandb
            freeze_path = f"{wandb_logger.experiment.dir}/package_versions.txt"
            os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
            wandb_logger.experiment.save(f"{freeze_path}")

        # Save data module config:
        wandb_logger.experiment.save(data_module_config)

    if runner_args.get("mpi_plugin"):
        cluster_environment = MPIEnvironment()
    else:
        cluster_environment = None

    # Select optimziation strategy
    IS_MULTIGPU = runner_args.get("num_gpus", 0) > 1
    IS_MULTINODE = runner_args.get("num_nodes", 1) > 1
    if runner_args.get("deepspeed_config_path"):
        deepspeed_config = config_utils.load_json(runner_args.deepspeed_config_path)
        # Set gradient clipping to what is specified by model_config
        deepspeed_config["gradient_clipping"] = model_config.settings.gradient_clipping
        strategy = DeepSpeedStrategy(
            config=runner_args.deepspeed_config_path,
            cluster_environment=cluster_environment,
        )
        if (runner_args.get("wandb") is not None) & IS_RANK_ZERO:
            wandb_logger.experiment.save(deepspeed_config)
            wandb_logger.experiment.save("openfold/config.py")
    elif IS_MULTIGPU or IS_MULTINODE:
        strategy = DDPStrategy(
            find_unused_parameters=False, cluster_environment=cluster_environment
        )
    else:
        strategy = None

    trainer_args = runner_args.pl_trainer.to_dict()
    trainer_args.update(
        {
            "default_root_dir": runner_args.output_dir,
            "strategy": strategy,
            "callbacks": callbacks,
            "logger": loggers,
            "devices": runner_args.num_gpus,
        }
    )

    trainer = pl.Trainer(**trainer_args)

    # Run process appropriate process
    logging.info(f"Running {runner_args.mode} mode.")
    # Training + validation / profiling
    if (runner_args.mode == "train") | (runner_args.mode == "profile"):
        if runner_args.mode == "profile":  # TODO Implement profiling
            raise NotImplementedError("Profiling mode not yet implemented.")
        else:
            trainer.fit(lightning_module, lightning_data_module, ckpt_path)
    # Testing
    elif runner_args.mode == "test":
        trainer.test(lightning_module, lightning_data_module)
    # Prediction == inference
    elif runner_args.mode == "predict":
        trainer.predict(lightning_module, lightning_data_module)
    else:
        raise ValueError(
            f"""Invalid mode argument: {runner_args.mode}. Choose one of "
            "'train', 'test', 'predict', 'profile'."""
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runner_yaml",
        type=str,
        help=(
            "Yaml that specifies mdoel and dataset parameters,"
            "see examples/runner.yml"
        ),
    )

    args = parser.parse_args()

    # Argument compatibility checks

    main(args)
