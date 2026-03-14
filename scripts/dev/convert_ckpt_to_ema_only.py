"""
Converts a full checkpoint file to a checkpoint that only contains EMA weights
for inference.

Usage:

python scripts/dev/convert_ckpt_to_ema_only.py /path/to/full_checkpoint
/path/to/output_ema_only_checkpoint
"""

import argparse
import operator
from pathlib import Path, PosixPath

import ml_collections as mlc
import torch

from openfold3.core.utils.checkpoint_loading_utils import load_checkpoint
from openfold3.projects.of3_all_atom.model import MODEL_VERSION, OpenFold3

# # Add OpenFold3 model to safe models to load
torch.serialization.add_safe_globals(
    [
        OpenFold3,
        mlc.ConfigDict,
        mlc.FieldReference,
        int,
        bool,
        float,
        operator.add,
        mlc.config_dict._Op,
        PosixPath,
    ]
)


def convert_checkpoint_to_ema_only(input_ckpt_path, output_ckpt_path):
    full_ckpt = load_checkpoint(Path(input_ckpt_path))

    ema_parameters = full_ckpt["ema"]["params"]
    torch.save(ema_parameters, output_ckpt_path)


def add_version_tensor_to_checkpoint(input_ckpt, output_ckpt):
    # Should only be required for checkpoint release before versioning was added
    model = torch.load(input_ckpt, map_location="cpu")
    model["version_tensor"] = MODEL_VERSION
    torch.save(model, output_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ckpt_path", type=str)
    parser.add_argument("output_ckpt_path", type=str)
    parser.add_argument(
        "--skip-ema-conversion",
        action="store_true",
        help="If set, will skip the conversion to EMA-only checkpoint and "
        "just add the version tensor to the input checkpoint.",
    )

    args = parser.parse_args()
    tmp_file = args.output_ckpt_path + ".tmp"

    if args.skip_ema_conversion:
        add_version_tensor_to_checkpoint(args.input_ckpt_path, args.output_ckpt_path)
    else:
        convert_checkpoint_to_ema_only(args.input_ckpt_path, tmp_file)
        add_version_tensor_to_checkpoint(tmp_file, args.output_ckpt_path)
