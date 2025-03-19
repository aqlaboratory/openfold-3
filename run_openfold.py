# args TODO add license

import argparse
import logging

import torch

from openfold3.experiment_builder import ExperimentBuilder

torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if torch_major_version > 1 or (torch_major_version == 1 and torch_minor_version >= 12):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


def main(args):
    args = parser.parse_args()
    builder = ExperimentBuilder(args.runner_yaml)
    builder.setup()
    builder.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runner_yaml",
        type=str,
        help=(
            "Yaml that specifies model and dataset parameters, see examples/runner.yml"
        ),
    )
    args = parser.parse_args()
    main(args)
