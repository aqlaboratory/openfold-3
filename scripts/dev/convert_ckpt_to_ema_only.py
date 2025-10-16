"""
Converts a full checkpoint file to a checkpoint that only contains EMA weights
for inference.

Usage:

python scripts/dev/convert_ckpt_to_ema_only.py /path/to/full_checkpoint
/path/to/output_ema_only_checkpoint
"""

import argparse
from pathlib import Path

import torch

from openfold3.core.utils.checkpoint_loading_utils import load_checkpoint


def convert_checkpoint_to_ema_only(args):
    full_ckpt = load_checkpoint(Path(args.input_ckpt_path))
    output_path = args.output_ckpt_path

    ema_parameters = full_ckpt["ema"]["params"]
    torch.save(ema_parameters, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ckpt_path", type=str)
    parser.add_argument("output_ckpt_path", type=str)

    args = parser.parse_args()

    convert_checkpoint_to_ema_only(args)
