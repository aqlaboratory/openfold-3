# Copyright 2026 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
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

"""Initialization functions for network parameters."""

import math
from functools import lru_cache

import torch
from scipy.stats import truncnorm
from torch import nn


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


@lru_cache
def _cached_truncnorm_std(a, b, loc, scale):
    return truncnorm.std(a, b, loc, scale)


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    f = _calculate_fan(weights.shape, fan)

    scale = scale / max(1, f)
    # truncnorm.std is always 0.8796256610342398
    std = float(math.sqrt(scale) / _cached_truncnorm_std(a=-2, b=2, loc=0, scale=1))

    with torch.no_grad():
        nn.init.trunc_normal_(
            weights,
            mean=0.0,
            std=std,
            a=-2.0 * std,
            b=2.0 * std,
        )


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def kaiming_normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")
