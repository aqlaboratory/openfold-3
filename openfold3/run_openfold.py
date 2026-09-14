# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""

Main run script for OpenFold3. Please see the README for usage details.

"""
# ruff: noqa: F821

import json
import logging
from pathlib import Path

import click

from openfold3.core.config import config_utils
from openfold3.entry_points.import_utils import _torch_gpu_setup

logger = logging.getLogger(__name__)


@click.group()
def cli():
    pass


def _summarize_query_set(query_set) -> dict:
    """Return a summary of an inference query set."""
    molecule_type_counts = {}
    ligand_representation_counts = {}
    chain_declaration_count = 0
    chain_instance_count = 0
    polymer_residue_count = 0
    ligand_chain_instance_count = 0
    main_msa_chain_count = 0
    paired_msa_chain_count = 0
    template_alignment_chain_count = 0
    direct_cif_template_chain_count = 0
    covalent_bond_count = 0

    for query in query_set.queries.values():
        covalent_bond_count += len(query.covalent_bonds or [])
        for chain in query.chains:
            chain_declaration_count += 1
            instance_count = len(chain.chain_ids)
            chain_instance_count += instance_count
            molecule_type = chain.molecule_type.name.lower()
            molecule_type_counts[molecule_type] = (
                molecule_type_counts.get(molecule_type, 0) + instance_count
            )

            if molecule_type == "ligand":
                ligand_chain_instance_count += instance_count
                if chain.smiles is not None:
                    representation = "smiles"
                elif chain.ccd_codes is not None:
                    representation = "ccd_codes"
                elif chain.sdf_file_path is not None:
                    representation = "sdf"
                else:
                    representation = "unspecified"
                ligand_representation_counts[representation] = (
                    ligand_representation_counts.get(representation, 0) + instance_count
                )
            elif chain.sequence is not None:
                polymer_residue_count += len(chain.sequence) * instance_count

            main_msa_chain_count += chain.main_msa_file_paths is not None
            paired_msa_chain_count += chain.paired_msa_file_paths is not None
            template_alignment_chain_count += (
                chain.template_alignment_file_path is not None
            )
            direct_cif_template_chain_count += chain.template_cif_paths is not None

    return {
        "query_count": len(query_set.queries),
        "chain_declaration_count": chain_declaration_count,
        "chain_instance_count": chain_instance_count,
        "molecule_type_counts": dict(sorted(molecule_type_counts.items())),
        "polymer_residue_count": polymer_residue_count,
        "ligand_chain_instance_count": ligand_chain_instance_count,
        "ligand_representation_counts": dict(
            sorted(ligand_representation_counts.items())
        ),
        "main_msa_chain_count": main_msa_chain_count,
        "paired_msa_chain_count": paired_msa_chain_count,
        "template_alignment_chain_count": template_alignment_chain_count,
        "direct_cif_template_chain_count": direct_cif_template_chain_count,
        "covalent_bond_count": covalent_bond_count,
    }


@cli.command("check-query")
@click.option(
    "--query-json",
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSON file containing queries to check.",
)
@click.option(
    "output_format",
    "--format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format for the query summary.",
)
def check_query(query_json: Path, output_format: str):
    """Validate and summarize a query JSON without running inference."""
    from pydantic import ValidationError

    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    try:
        query_set = InferenceQuerySet.from_json(query_json)
    except (OSError, UnicodeError, ValidationError) as exc:
        raise click.ClickException(f"Query validation failed:\n{exc}") from exc

    summary = _summarize_query_set(query_set)
    if output_format.lower() == "json":
        click.echo(json.dumps({"valid": True, "summary": summary}, indent=2))
        return

    click.echo("Query file is valid.")
    click.echo(f"Queries: {summary['query_count']}")
    click.echo(f"Chain declarations: {summary['chain_declaration_count']}")
    click.echo(f"Chain instances: {summary['chain_instance_count']}")
    click.echo(f"Molecule types: {summary['molecule_type_counts']}")
    click.echo(f"Polymer residues: {summary['polymer_residue_count']}")
    click.echo(f"Ligand chain instances: {summary['ligand_chain_instance_count']}")
    click.echo(f"Ligand representations: {summary['ligand_representation_counts']}")
    click.echo(f"Chains with main MSA paths: {summary['main_msa_chain_count']}")
    click.echo(f"Chains with paired MSA paths: {summary['paired_msa_chain_count']}")
    click.echo(
        "Chains with template alignment paths: "
        f"{summary['template_alignment_chain_count']}"
    )
    click.echo(
        "Chains with direct-CIF templates: "
        f"{summary['direct_cif_template_chain_count']}"
    )
    click.echo(f"Covalent bonds: {summary['covalent_bond_count']}")


@cli.command()
@click.option(
    "--runner-yaml",
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Yaml that specifies model and dataset parameters,"
    " see examples/training_new.yml",
)
@click.option("--seed", type=int, help="Initial seed for all processes")
@click.option(
    "--data-seed",
    "--data_seed",
    type=int,
    help="Initial seed for data pipeline. Defaults to seed if not specified.",
)
def train(runner_yaml: Path, seed: int | None = None, data_seed: int | None = None):
    """Perform a training experiment with a preprepared dataset cache."""
    _torch_gpu_setup()
    from openfold3.entry_points.experiment_runner import (
        TrainingExperimentRunner,
    )
    from openfold3.entry_points.validator import (
        TrainingExperimentConfig,
    )

    runner_dict = config_utils.load_yaml(runner_yaml)

    # overwrite seed defaults if provided:
    if seed is not None:
        runner_dict["experiment_settings"]["seed"] = seed

    if data_seed is not None:
        runner_dict["data_module_args"]["data_seed"] = data_seed

    expt_config = TrainingExperimentConfig.model_validate(runner_dict)

    expt_runner = TrainingExperimentRunner(expt_config)
    expt_runner.setup()
    expt_runner.run()


@cli.command()
@click.option(
    "--query-json",
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Json containing the queries for prediction.",
)
@click.option(
    "--inference-ckpt-path",
    "--inference_ckpt_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
    required=False,
    help="Path for model checkpoint to be used for inference. "
    "If not specified, will attempt to find or download parameters to "
    "$OPENFOLD_CACHE [default: ~/.openfold3/]",
)
@click.option(
    "--inference-ckpt-name",
    "--inference_ckpt_name",
    type=str,
    required=False,
    help="Name of the checkpoint to be used for inference."
    " Only used if `inference_ckpt_path` is not specified.",
)
@click.option(
    "--num-diffusion-samples",
    "--num_diffusion_samples",
    type=int,
    default=None,
    required=False,
    help="Number of diffusion samples to generate for each query.",
)
@click.option(
    "--num-model-seeds",
    "--num_model_seeds",
    type=int,
    default=None,
    required=False,
    help="Number of model seeds to use for each query.",
)
@click.option(
    "--runner-yaml",
    "--runner_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="Yaml that specifies model and dataset parameters, see examples/runner.yml",
)
@click.option(
    "--use-msa-server",
    "--use_msa_server",
    type=bool,
    default=None,
    help=(
        "Use ColabFold MSA server to perform alignments. If unset, the value from"
        " the runner yaml (or config default) is used."
    ),
)
@click.option(
    "--use-templates",
    "--use_templates",
    type=bool,
    default=None,
    help=(
        "Whether to use templates for prediction. If unset, the value from the"
        " runner yaml (or config default) is used."
    ),
)
@click.option(
    "--output-dir",
    "--output_dir",
    type=click.Path(exists=False, file_okay=True, dir_okay=True, path_type=Path),
    required=False,
    help="Output directory for writing results",
)
def predict(
    query_json: Path,
    inference_ckpt_path: Path | None = None,
    inference_ckpt_name: str | None = None,
    num_diffusion_samples: int | None = None,
    num_model_seeds: int | None = None,
    runner_yaml: Path | None = None,
    use_msa_server: bool | None = None,
    use_templates: bool | None = None,
    output_dir: Path | None = None,
):
    """Perform inference on a set of queries defined in the query_json."""
    _torch_gpu_setup()

    from openfold3.entry_points.experiment_runner import (
        InferenceExperimentRunner,
    )
    from openfold3.entry_points.validator import (
        InferenceExperimentConfig,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    logging.basicConfig(level=logging.INFO)
    runner_args = config_utils.load_yaml(runner_yaml) if runner_yaml else dict()

    expt_config = InferenceExperimentConfig(
        inference_ckpt_path=inference_ckpt_path,
        inference_ckpt_name=inference_ckpt_name,
        **runner_args,
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

    # Run the forward pass
    expt_runner.setup()
    expt_runner.run(query_set)
    expt_runner.cleanup()


@cli.command()
@click.option(
    "--query-json",
    "--query_json",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Json containing the queries for prediction.",
)
@click.option(
    "--output-dir",
    "--output_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Output directory for writing alignments",
)
@click.option(
    "--msa-computation-settings-yaml",
    "--msa_computation_settings_yaml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    help="Yaml file to customize Colabfold MSA settings,"
    " see MsaComputationSettings for options.",
)
def align_msa_server(
    query_json: Path,
    output_dir: Path,
    msa_computation_settings_yaml: Path | None = None,
):
    """Run MSA server alignment only with ColabFold MSA server.
    
    Example command:
    python run_openfold.py align-msa-server \
        --query_json query_example.json \
        --output_dir output/msa_server_test \
    
    More settings can be specified using the `msa_computation_settings_yaml` flag
    An example yaml file is provided in `examples/msa_server.yml`
    """
    _torch_gpu_setup()
    from openfold3.core.data.tools.colabfold_msa_server import (
        MsaComputationSettings,
        preprocess_colabfold_msas,
    )
    from openfold3.projects.of3_all_atom.config.inference_query_format import (
        InferenceQuerySet,
    )

    query_set = InferenceQuerySet.from_json(query_json)

    msa_settings = MsaComputationSettings.from_config_with_cli_override(
        output_dir, msa_computation_settings_yaml
    )
    query_set = preprocess_colabfold_msas(
        inference_query_set=query_set,
        compute_settings=msa_settings,
    )

    with open(output_dir / "query_msa.json", "w") as fp:
        fp.write(query_set.model_dump_json(indent=4))


if __name__ == "__main__":
    cli()
