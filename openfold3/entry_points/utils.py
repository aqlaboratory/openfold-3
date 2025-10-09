import importlib
import logging
from pathlib import Path

import torch

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    from deepspeed.utils import zero_to_fp32

logger = logging.getLogger(__name__)


def load_model_state_dict_from_ds_checkpoint(checkpoint_dir: Path) -> dict:
    latest_path = checkpoint_dir / "latest"
    if latest_path.is_file():
        with open(latest_path) as fd:
            tag = fd.read().strip()
    else:
        raise ValueError(f"Unable to find 'latest' file at {latest_path}")

    ds_checkpoint_dir = checkpoint_dir / tag
    _DS_CHECKPOINT_VERSION = 2  # based on manual parsing of checkpoint files
    state_file = zero_to_fp32.get_model_state_file(
        str(ds_checkpoint_dir), _DS_CHECKPOINT_VERSION
    )
    return torch.load(state_file)


def load_checkpoint(ckpt_path: Path) -> dict:
    if ckpt_path.is_dir():
        return load_model_state_dict_from_ds_checkpoint(ckpt_path)

    if ckpt_path.is_file():
        return torch.load(ckpt_path)

    raise ValueError(f"Checkpoint path {ckpt_path} is not a valid file or directory.")


def get_state_dict_from_checkpoint(ckpt: dict, init_from_ema_weights: bool) -> dict:
    is_pretrained_model = "module" not in ckpt or "state_dict" not in ckpt

    # Loading from pre-trained model
    if is_pretrained_model:
        return {"model." + k: v for k, v in ckpt.items()}

    # Loading from EMA weights
    if init_from_ema_weights:
        logger.info("Loading model from ema weights")
        return {"model." + k: v for k, v in ckpt["ema"]["params"].items()}

    # Loading from DeepSpeed checkpoint
    if "module" in ckpt:
        return ckpt["module"]

    # Loading from PL checkpoint
    return ckpt["state_dict"]
