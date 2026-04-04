# Copyright 2026 AlQuraishi Laboratory
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

import re

import torch

from openfold3.core.model.layers.triangular_multiplicative_update import (
    FusedTriangleMultiplicationOutgoing,
    TriangleMultiplicationOutgoing,
)
from openfold3.tests.config import consts

# Updates pair representation z[i,j] by projecting to two gated vectors (a, b),
# contracting along a shared dimension (outgoing vs incoming), then projecting
# back. "Outgoing" contracts over the starting node, "Incoming" over the ending
# node. Shape-preserving: [*, N, N, C_z] -> [*, N, N, C_z].


def _make_module(c_z, c):
    """Pick fused vs non-fused variant based on model preset."""
    # Multimer v3 uses a fused variant (single projection split into a, b)
    # vs separate projections for each
    if re.fullmatch("^model_[1-5]_multimer_v3$", consts.model_preset):
        return FusedTriangleMultiplicationOutgoing(c_z, c)
    return TriangleMultiplicationOutgoing(c_z, c)


def test_shape(device, seeded_rng, ndarrays_regression):
    # c_z: pair representation channel dim (128 in production)
    c_z = consts.c_z
    # c: hidden projection dim (production uses ~128; smaller here for speed)
    c = 11

    tm = _make_module(c_z, c).to(device)
    # Reinitialize all params to non-trivial values (some layers may be
    # zero-initialized by default for residual identity at init)
    for p in tm.parameters():
        torch.nn.init.normal_(p, std=0.01)
    tm.eval()

    n_res = consts.n_res
    batch_size = consts.batch_size

    # Pair representation: [batch, N_residues, N_residues, C_z]
    x = torch.rand((batch_size, n_res, n_res, c_z), device=device)
    # Binary mask: which residue pairs are valid
    mask = torch.randint(0, 2, size=(batch_size, n_res, n_res), device=device)
    shape_before = x.shape
    with torch.no_grad():
        x = tm(x, mask)
    shape_after = x.shape

    # Shape must be preserved for the residual addition z = z + tri_mul(z)
    assert shape_before == shape_after

    # Guard against trivial all-zero output (e.g. from zero-initialized weights)
    assert x.abs().max().item() > 0, (
        "Output is all zeros — snapshot would be meaningless"
    )

    # Snapshot regression: output must be numerically identical across runs.
    # CUDA tolerances are looser to accommodate hardware-level differences.
    # Regenerate with: pytest --force-regen
    tolerances = dict(atol=1e-6, rtol=1e-5)
    ndarrays_regression.check(
        {"output": x.cpu().numpy()},
        default_tolerance=tolerances,
    )
