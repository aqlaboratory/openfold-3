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

import math
import unittest

import torch

from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
from openfold3.core.model.layers.transition import ConditionedTransitionBlock
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts
from openfold3.tests.utils.data_utils import randomize_parameters


class TestDiffusionTransformer(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_res = consts.n_res
        c_a = 768
        c_s = consts.c_s
        c_z = consts.c_z
        c_hidden = 16
        no_heads = 3
        no_blocks = 2

        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()

        diff_transformer_config = (
            config.architecture.diffusion_module.diffusion_transformer
        )
        diff_transformer_config.update(
            {
                "c_a": c_a,
                "c_s": c_s,
                "c_z": c_z,
                "c_hidden": c_hidden,
                "no_heads": no_heads,
                "no_blocks": no_blocks,
            }
        )

        dt = DiffusionTransformer(**diff_transformer_config).eval()

        a = torch.rand((batch_size, n_res, c_a))
        s = torch.rand((batch_size, n_res, c_s))
        z = torch.rand((batch_size, n_res, n_res, c_z))
        single_mask = torch.randint(0, 2, size=(batch_size, n_res))

        shape_a_before = a.shape

        a = dt(a, s, z, mask=single_mask)

        self.assertTrue(a.shape == shape_a_before)


class TestDiffusionTransformerMaskBias(unittest.TestCase):
    """
    `get_mask_bias` lets the token/atom key-mask bias be computed once per diffusion
    rollout instead of at every block of every step. Passing it to `forward` as
    `mask_bias` must give the same result as deriving the bias from `mask` inside
    `forward`, both for the plain (token-level) and the cross-attention (blocked
    atom-level) transformer.
    """

    def _make_transformer(self, n_query, n_key, c_a, c_z):
        dt = DiffusionTransformer(
            c_a=c_a,
            c_s=c_a,
            c_z=c_z,
            c_hidden=8,
            no_heads=2,
            no_blocks=2,
            n_transition=2,
            n_query=n_query,
            n_key=n_key,
            inf=1e9,
        ).eval()
        return randomize_parameters(dt)

    def test_token_level(self):
        batch_size = consts.batch_size
        n_sample = 3
        n_token = consts.n_res
        c_a = 32
        c_z = 8

        dt = self._make_transformer(n_query=None, n_key=None, c_a=c_a, c_z=c_z)

        a = torch.randn((batch_size, n_sample, n_token, c_a))
        s = torch.randn((batch_size, 1, n_token, c_a))
        z = torch.randn((batch_size, 1, n_token, n_token, c_z))
        mask = torch.ones((batch_size, 1, n_token))
        mask[..., -4:] = 0

        with torch.no_grad():
            expected = dt(a, s, z, mask=mask)
            actual = dt(a, s, z, mask=mask, mask_bias=dt.get_mask_bias(mask))

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_atom_level_blocks(self):
        batch_size = consts.batch_size
        n_sample = 3
        n_atom = 150
        n_query = 32
        n_key = 128
        num_blocks = math.ceil(n_atom / n_query)
        c_a = 32
        c_z = 8

        dt = self._make_transformer(n_query=n_query, n_key=n_key, c_a=c_a, c_z=c_z)

        a = torch.randn((batch_size, n_sample, n_atom, c_a))
        s = torch.randn((batch_size, 1, n_atom, c_a))
        z = torch.randn((batch_size, 1, num_blocks, n_query, n_key, c_z))
        mask = torch.ones((batch_size, 1, n_atom))
        mask[..., -20:] = 0

        with torch.no_grad():
            expected = dt(a, s, z, mask=mask) # Compute the output using the mask directly
            actual = dt(a, s, z, mask=mask, mask_bias=dt.get_mask_bias(mask)) # Use the precomputed mask

        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_atom_level_mask_bias_shape(self):
        n_atom = 150
        n_query = 32
        n_key = 128
        num_blocks = math.ceil(n_atom / n_query)

        dt = self._make_transformer(n_query=n_query, n_key=n_key, c_a=32, c_z=8)

        mask = torch.ones((consts.batch_size, 1, n_atom))
        mask[..., -20:] = 0
        mask_bias = dt.get_mask_bias(mask)

        # Computed once for the sample dim of the batch features, not per sample
        self.assertEqual(
            mask_bias.shape, (consts.batch_size, 1, num_blocks, 1, n_query, n_key)
        )


class TestConditionedTransitionBlock(unittest.TestCase):
    def test_shape(self):
        batch_size = 2
        n_r = 5
        c_a = 14
        c_s = 7
        n = 11

        ct = ConditionedTransitionBlock(
            c_a=c_a,
            c_s=c_s,
            n=n,
        )

        a = torch.rand((batch_size, n_r, c_a))
        s = torch.rand((batch_size, n_r, c_s))

        shape_before = a.shape
        a = ct(a=a, s=s)
        shape_after = a.shape

        self.assertTrue(shape_before == shape_after)


if __name__ == "__main__":
    unittest.main()
