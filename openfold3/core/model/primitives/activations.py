import torch
from torch import nn

from .linear import Linear


class ReLULayer(nn.Module):
    def __init__(self, c_in: int, c_out: int, bias: bool = True, init="relu"):
        """

        Args:
            c_in:
            c_out:
            bias:
            init:
        """
        super(ReLULayer, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear = Linear(self.c_in, self.c_out, bias=bias, init=init)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x:

        Returns:

        """
        return self.relu(self.linear(x))


class SwiGLU(nn.Module):
    def __init__(self, c_in: int, c_out: int, bias: bool = False, init="he_normal"):
        """

        Args:
            c_in:
            c_out:
            bias:
            init:
        """
        super(SwiGLU, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.linear_a = Linear(self.c_in, self.c_out, bias=bias, init=init)
        self.linear_b = Linear(self.c_in, self.c_out, bias=bias, init=init)
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x:

        Returns:

        """
        return self.swish(self.linear_a(x)) * self.linear_b(x)
