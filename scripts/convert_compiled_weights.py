"""Script for converting Deepspeed training checkpoints into a single file.

Given an example directory:
    epoch=5-step=10999.ckpt
    ├── checkpoint
    │   ├── zero_pp_rank_0_mp_rank_00_model_states.pt
    │   ├── zero_pp_rank_0_mp_rank_00_optim_states.pt
    │   ├── zero_pp_rank_1_mp_rank_00_model_states.pt
    │   └── zero_pp_rank_1_mp_rank_00_optim_states.pt
    ├── latest
    └── zero_to_fp32.py

Example usage:

python scripts/convert_compiled_weights.py /path/epoch=5-step=10999.ckpt/
/path/epoch=5-step=10999.ckpt/converted.ckpt.pt

NB: for inference with pytorch-lightning + deepspeed, the converted checkpoint file is
expected to be in a deepseed checkpoint directory.
"""

import argparse

from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)


def convert_compiled_weights(args):
    """Converts state dict and ema parameters from a compiled model, but not optimizer
    states.
    Note: Not compatible on cpu, need to run conversion on gpu
    """
    input_path = args.input_ckpt_path
    output_path = args.output_ckpt_path

    convert_zero_checkpoint_to_fp32_state_dict(input_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_ckpt_path", type=str)
    parser.add_argument("output_ckpt_path", type=str)

    args = parser.parse_args()

    convert_compiled_weights(args)
