import torch

from openfold3.core.model.primitives.initialization import lecun_normal_init_


def initialize_model_weights(model):
    """Re-initialize model weights with non-zero values.

    Otherwise tests that check for agreement between different implementations
    will effectively test nothing because the zero-initialized linear weights
    will turn everything to zeros.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                lecun_normal_init_(module.weight)
