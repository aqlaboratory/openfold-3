# TODO add license

import argparse
import logging
import os
import sys

import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import MPIEnvironment
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from openfold3.core.config import config_utils
from openfold3.core.data.framework.data_module import DataModule
from openfold3.core.utils.callbacks import (
    EarlyStoppingVerbose,
    PerformanceLoggingCallback,
)
from openfold3.core.utils.torchscript import script_preset_
from openfold3.model_implementations import registry


def main(args):
    # Parse arguments/update input config
    # IS_LOW_PRECISION = args.precision in [
    #     "bf16-mixed",
    #     "16",
    #     "bf16",
    #     "16-true",
    #     "16-mixed",
    #     "bf16-mixed",
    # ]
    IS_RANK_ZERO = args.mpi_plugin and (int(os.environ.get("PMI_RANK")) == 0)
    # TODO <Set up model and data configs>

    # Set seed
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # Initialize model wrapper
    runner_yaml_config = config_utils.load_yaml(args.runner_yaml)
    # Update model config with section from yaml with model update
    model_config = registry.make_model_config(
        args.model_name, runner_yaml_config.model_update_yaml
    )
    # Initialize lightning module with desired config
    lightning_module = registry.get_lightning_module(model_config)
    # TODO <checkpoint resume logic goes here>

    # Script model
    if args.script_model:
        script_preset_(lightning_module)  # TODO check if this works

    # Initialize data wrapper
    lightning_data_module = DataModule(args.data_config)

    # Set up trainer arguments and callbacks
    callbacks = []

    if args.checkpoint_every_epoch:
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=1,
                auto_insert_metric_name=False,
                save_top_k=-1,
            )
        )

    if args.early_stopping:
        # TODO check if works/necessary
        callbacks.append(
            EarlyStoppingVerbose(
                monitor="val/lddt_ca",
                min_delta=args.min_delta,
                patience=args.patience,
                verbose=False,
                mode="max",
                check_finite=True,
                strict=True,
            )
        )

    if args.log_performance:
        global_batch_size = (
            args.batch_size
            * args.gpus
            * args.num_nodes
            * args.gradient_accumulation_steps
        )
        # TODO check if works/necessary
        callbacks.append(
            PerformanceLoggingCallback(
                log_file=os.path.join(args.output_dir, "performance_log.json"),
                global_batch_size=global_batch_size,
            )
        )

    if args.log_lr:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    loggers = []
    if args.wandb:
        # Create W&B run ID if necessary
        if (args.checkpoint_path is None) & IS_RANK_ZERO:
            wandb_run_id = wandb.util.generate_id()
        elif (args.checkpoint_path is not None) & IS_RANK_ZERO:
            wandb_run_id = args.wandb_id
        else:
            wandb_run_id = None
        # Initialize W&B for zero-rank process only, ontherwise each process will log
        # separately onto W&B
        if args.mpi_plugin & IS_RANK_ZERO:
            wandb_init_dict = dict(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=args.experiment_name,
                dir=args.output_dir,
                resume="allow",
                reinit=True,
                id=wandb_run_id,
            )
            wandb.init(**wandb_init_dict)
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.experiment_name,
            dir=args.output_dir,
            resume="allow",
            reinit=True,
            log_model=False,
            id=wandb_run_id,
        )
        loggers.append(wandb_logger)
        # Still don't know what this does @Christina? :D
        if IS_RANK_ZERO:
            freeze_path = f"{wandb_logger.experiment.dir}/package_versions.txt"
            os.system(f"{sys.executable} -m pip freeze > {freeze_path}")
            wandb_logger.experiment.save(f"{freeze_path}")

    if args.mpi_plugin:
        cluster_environment = MPIEnvironment()
    else:
        cluster_environment = None

    if args.deepsepped_config_path is not None:
        strategy = DeepSpeedStrategy(
            config=args.deepspeed_config_path, cluster_environment=cluster_environment
        )
        if args.wandb & IS_RANK_ZERO:
            wandb_logger.experiment.save(args.deepspeed_config_path)
            wandb_logger.experiment.save("openfold/config.py")
    elif ((args.gpus is not None) & (args.gpus > 1)) | (args.num_nodes > 1):
        strategy = DDPStrategy(
            find_unused_parameters=False, cluster_environment=cluster_environment
        )
    else:
        strategy = None

    trainer_args = {
        "num_nodes": args.num_nodes,
        "precision": args.precision,
        "max_epochs": args.max_epochs,
        "log_every_n_steps": args.log_every_n_steps,
        "flush_logs_ever_n_steps": args.flush_logs_ever_n_steps,
        "num_sanity_val_steps": args.num_sanity_val_steps,
        "reload_dataloaders_every_n_epochs": args.reload_dataloaders_every_n_epochs,
        "default_root_dir": args.output_dir,
        "strategy": strategy,
        "callbacks": callbacks,
        "logger": loggers,
    }

    trainer = pl.Trainer(**trainer_args)

    # Run process appropriate process
    logging.info(f"Running {args.mode} mode.")
    # Training + validation / profiling
    if (args.mode == "train") | (args.mode == "profile"):
        ckpt_path = None if args.resume_model_weights_only else args.resume_from_ckpt
        if args.mode == "profile":  # TODO can remove when profiling is implemented
            raise NotImplementedError("Profiling mode not yet implemented.")
        else:
            trainer.fit(lightning_module, lightning_data_module, ckpt_path)
    # Testing
    elif args.mode == "test":
        trainer.test(lightning_module, lightning_data_module)
    # Prediction == inference
    elif args.mode == "predict":
        trainer.predict(lightning_module, lightning_data_module)
    else:
        raise ValueError(
            f"""Invalid mode argument: {args.mode}. Choose one of 'train', 'test', \
             'predict', 'profile'."""
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # TODO add necessary arguments

    args = parser.parse_args()

    # Argument compatibility checks

    main(args)
