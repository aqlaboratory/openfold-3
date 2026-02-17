# Copyright 2025 AlQuraishi Laboratory
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
def test_shape(starting, ndarrays_regression):
    # NOTE: seeding may need further work — torch.manual_seed controls both
    # the random input and the module's weight init. If init changes upstream,
    # regenerate snapshots with: pytest --force-regen
    torch.manual_seed(42)

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
    )
    tan.eval()

    batch_size = consts.batch_size
    n_res = consts.n_res

    # Pair representation: [batch, N_residues, N_residues, C_z]
    x = torch.rand((batch_size, n_res, n_res, c_z))
    shape_before = x.shape
    # chunk_size=None -> no memory-saving chunking, full attention in one pass
    with torch.no_grad():
        x = tan(x, chunk_size=None)
    shape_after = x.shape

    # Shape must be preserved for the residual addition z = z + tri_att(z)
    assert shape_before == shape_after

    # Snapshot regression: output must be numerically identical across runs.
    # Regenerate with: pytest --force-regen
    ndarrays_regression.check(
        {"output": x.cpu().numpy()},
        default_tolerance=dict(atol=1e-6, rtol=1e-5),
    )
