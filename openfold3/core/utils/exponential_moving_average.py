import warnings
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn

from openfold3.core.utils.tensor_utils import tensor_tree_map


class ExponentialMovingAverage:
    """
    Maintains moving averages of parameters with exponential decay

    At each step, the stored copy `copy` of each parameter `param` is
    updated as follows:

        `copy = decay * copy + (1 - decay) * param`

    where `decay` is an attribute of the ExponentialMovingAverage object.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float,
        submodules_to_update: Optional[list] = None,
    ):
        """
        Args:
            model:
                A torch.nn.Module whose parameters are to be tracked
            decay:
                A value (usually close to 1.) by which updates are
                weighted as part of the above formula
            submodules_to_update:
                A list of submodules whose EMA weights will be updated.
                If not specified, all weights are updated.
        """
        super().__init__()

        def clone_param(t):
            return t.clone().detach()

        self.params = tensor_tree_map(clone_param, model.state_dict())
        self.decay = decay
        self.submodules_to_update = submodules_to_update
        self.device = next(model.parameters()).device

    def to(self, device):
        self.params = tensor_tree_map(lambda t: t.to(device), self.params)
        self.device = device
        return self

    def _update_state_dict_(self, update, state_dict):
        with torch.no_grad():
            for k, v in update.items():
                stored = state_dict[k]
                if not isinstance(v, torch.Tensor):
                    self._update_state_dict_(v, stored)
                else:
                    diff = stored - v
                    diff *= 1 - self.decay
                    stored -= diff

    def update(self, model: torch.nn.Module) -> None:
        """
        Updates the stored parameters using the state dict of the provided
        module. The module should have the same structure as that used to
        initialize the ExponentialMovingAverage object.
        """
        if self.submodules_to_update is None:
            # If no subset is specified, update all parameters.
            self._update_state_dict_(model.state_dict(), self.params)
            return

        # If a subset is specified, filter the state_dict to only include
        # parameters from the enabled submodules.
        update_dict = OrderedDict()
        model_state_dict = model.state_dict()

        for key, value in model_state_dict.items():
            is_enabled = any(
                key == prefix or key.startswith(f"{prefix}.")
                for prefix in self.submodules_to_update
            )
            if is_enabled:
                update_dict[key] = value

        if not update_dict:
            warnings.warn(
                "ExponentialMovingAverage: No parameters found for the specified "
                f"submodules_to_update: {self.submodules_to_update}.",
                stacklevel=2,
            )
            return

        self._update_state_dict_(update_dict, self.params)

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        for k in state_dict["params"]:
            self.params[k] = state_dict["params"][k].clone()
        self.decay = state_dict["decay"]

    def state_dict(self) -> OrderedDict:
        return OrderedDict(
            {
                "params": self.params,
                "decay": self.decay,
            }
        )
