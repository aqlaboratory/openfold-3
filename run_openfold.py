# args TODO add license

import logging
from pathlib import Path

import click
import torch

from openfold3.core.config import config_utils
from openfold3.entry_points.experiment_runner import TrainingExperimentRunner
from openfold3.entry_points.validator import (
    TrainableExperimentConfig,
)

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if torch_major_version > 1 or (torch_major_version == 1 and torch_minor_version >= 12):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)

@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Yaml that specifies model and dataset parameters, see examples/runner.yml",
)
@click.option("--seed", type=int, help="Initial seed for all processes")
@click.option(
    "--data_seed",
    type=int,
    help="Initial seed for data pipeline. Defaults to seed if not specified.",
)
def train(runner_yaml: Path, seed: int | None = None, data_seed: int | None = None):
    """Perform a training experiment with a preprepared dataset cache."""
    expt_config = TrainableExperimentConfig.model_validate(
        config_utils.load_yaml(runner_yaml)
    )
    
    # overwrite seed defaults if provided:
    expt_config.seed = seed if seed else expt_config.seed
    expt_config.data_seed = data_seed if data_seed else expt_config.data_seed

    expt_runner = TrainingExperimentRunner(expt_config)
    expt_runner.setup()
    expt_runner.run()

@click.command()
@click.option(
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Json containing the queries for prediction.", 
)
@click.option(
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="Yaml that specifies model and dataset parameters, see examples/runner.yml",
)
@click.option("--use_msa_server",type=bool, default=True, help="Use ColabFold MSA server to perform alignments.")
def predict(query_json: Path, runner_yaml: Path | None = None, use_msa_server: bool = True):
    raise NotImplementedError("Prediction is not implemented yet.") 
    # query_set = InferenceQuerySet.from_json(runner_args.inference_query_path)

    # Msa processing logic here:

    # expt_config = TrainableExperimentConfig.model_validate(
    #     config_utils.load_yaml(runner_yaml)
    # )
    

if __name__ == "__main__":
    cli()
