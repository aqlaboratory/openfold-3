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

import pytest
import torch

from openfold3.core.model.layers.triangular_attention import TriangleAttention
from openfold3.tests.config import consts


# starting=True -> "starting node" variant: rows attend to rows,
# biased by z[i, k]. False would transpose internally for the
# "ending node" variant (columns attend to columns).
@pytest.mark.parametrize("starting", [True, False])
def test_shape(starting, device, seeded_rng, ndarrays_regression):
    # c_z: pair representation channel dim (128 in production)
    c_z = consts.c_z
    # c: attention hidden dim (production uses 32; smaller here for speed)
    c = 12
    no_heads = 4

    tan = TriangleAttention(
        c_z,
        c,
        no_heads,
        starting=starting,
    ).to(device)
    # AlphaFold initializes the output projection to zero (so residual blocks
    # start as identity). Reinitialize all params so the test exercises the
    # actual computation and produces non-trivial output.
    for p in tan.parameters():
        torch.nn.init.normal_(p, std=0.01)
    tan.eval()

    batch_size = consts.batch_size
    n_res = consts.n_res

    # Pair representation: [batch, N_residues, N_residues, C_z]
    x = torch.rand((batch_size, n_res, n_res, c_z), device=device)
    shape_before = x.shape
    # chunk_size=None -> no memory-saving chunking, full attention in one pass
    with torch.no_grad():
        x = tan(x, chunk_size=None)
    shape_after = x.shape

    # Shape must be preserved for the residual addition z = z + tri_att(z)
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
