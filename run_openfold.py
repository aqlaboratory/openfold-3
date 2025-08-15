# args TODO add license
r"""

# training
python run_openfold.py train --runner_yaml=examples/training_new.yml \
    --seed=42 \
    --data_seed=1234

# inference
python run_openfold.py predict --runner_yaml=examples/inference_new.yml

"""

import logging
from pathlib import Path

import click
import torch

from openfold3.core.config import config_utils
from openfold3.core.data.pipelines.preprocessing.template import TemplatePreprocessor
from openfold3.core.data.tools.colabfold_msa_server import preprocess_colabfold_msas
from openfold3.entry_points.experiment_runner import (
    InferenceExperimentRunner,
    TrainingExperimentRunner,
)
from openfold3.entry_points.validator import (
    InferenceExperimentConfig,
    TrainingExperimentConfig,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    colabfold_msa_settings,
)
from openfold3.projects.of3_all_atom.config.inference_query_format import (
    InferenceQuerySet,
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
    help="Yaml that specifies model and dataset parameters,"
    " see examples/training_new.yml",
)
@click.option("--seed", type=int, help="Initial seed for all processes")
@click.option(
    "--data_seed",
    type=int,
    help="Initial seed for data pipeline. Defaults to seed if not specified.",
)
def train(runner_yaml: Path, seed: int | None = None, data_seed: int | None = None):
    """Perform a training experiment with a preprepared dataset cache."""
    expt_config = TrainingExperimentConfig.model_validate(
        config_utils.load_yaml(runner_yaml)
    )

    # overwrite seed defaults if provided:
    expt_config.experiment_settings.seed = (
        seed if seed else expt_config.experiment_settings.seed
    )
    expt_config.data_module_args.data_seed = (
        data_seed if data_seed else expt_config.data_module_args.data_seed
    )

    expt_runner = TrainingExperimentRunner(expt_config)
    expt_runner.setup()
    expt_runner.run()


@cli.command()
@click.option(
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Json containing the queries for prediction.",
)
@click.option(
    "--inference_ckpt_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    required=True,
    help="Path for model checkpoint to be used for inference",
)
@click.option(
    "--num_diffusion_samples",
    type=int,
    default=None,
    required=False,
    help="Number of diffusion samples to generate for each query.",
)
@click.option(
    "--num_model_seeds",
    type=int,
    default=None,
    required=False,
    help="Number of model seeds to use for each query.",
)
@click.option(
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="Yaml that specifies model and dataset parameters, see examples/runner.yml",
)
@click.option(
    "--use_msa_server",
    type=bool,
    default=True,
    help="Use ColabFold MSA server to perform alignments.",
)
@click.option(
    "--use_templates",
    type=bool,
    default=False,
    help="Use ColabFold MSA server to perform template alignments.",
)
@click.option(
    "--output_dir",
    type=click.Path(exists=False, file_okay=True, dir_okay=True, path_type=Path),
    required=False,
    help="Output directory for writing results",
)
def predict(
    query_json: Path,
    inference_ckpt_path: Path,
    num_diffusion_samples: int | None = None,
    num_model_seeds: int | None = None,
    runner_yaml: Path | None = None,
    use_msa_server: bool = True,
    use_templates: bool = False,
    output_dir: Path | None = None,
):
    """Perform inference on a set of queries defined in the query_json."""
    logging.basicConfig(level=logging.INFO)
    runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else dict()

    expt_config = InferenceExperimentConfig(
        inference_ckpt_path=inference_ckpt_path, **runner_args
    )
    expt_runner = InferenceExperimentRunner(
        expt_config,
        num_diffusion_samples,
        num_model_seeds,
        use_msa_server,
        use_templates,
        output_dir,
    )

    # Load inference query set
    query_set = InferenceQuerySet.from_json(query_json)

    # Perform MSA computation if selected
    #  update query_set with MSA paths
    if expt_runner.use_msa_server:
        logger.info("Using ColabFold MSA server for alignments.")
        query_set = preprocess_colabfold_msas(
            inference_query_set=query_set,
            compute_settings=expt_config.msa_computation_settings,
        )

        # Update the msa dataset config settings
        updated_dataset_config_kwargs = expt_config.dataset_config_kwargs.model_copy(
            update={"msa": colabfold_msa_settings}
        )
        expt_config = expt_config.model_copy(
            update={"dataset_config_kwargs": updated_dataset_config_kwargs}
        )

    # Preprocess template alignments and optionally template structures
    if expt_runner.use_templates:
        logger.info("Using templates for inference.")
        template_preprocessor = TemplatePreprocessor(
            input_set=query_set, config=expt_config.template_preprocessor_settings
        )
        template_preprocessor()
    else:
        logger.info("Not using templates for inference.")

    # Run the forward pass
    expt_runner.setup()
    expt_runner.run(query_set)
    expt_runner.cleanup()

    # TODO add post-process relaxation with openmm


@cli.command()
def align_msa_server(inference_query_set):
    raise NotImplementedError("Alignment is not implemented yet.")
    # run_msa_server(inference_query_set)
    # inference_query_set.dump()


if __name__ == "__main__":
    cli()
