import importlib.util

import torch


def is_cuequivariance_available() -> bool:
    """
    Check if cuequivariance_torch is installed and CUDA is available.
    Even if cuequivariance_torch is installed, it only works with CUDA.
    """
    return (
        importlib.util.find_spec("cuequivariance_torch") is not None
        and torch.cuda.is_available()
    )
