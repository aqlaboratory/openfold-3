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

import unittest

import torch

from openfold3.core.model.layers.attention_pair_bias import AttentionPairBias
from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer
from openfold3.core.model.layers.transition import ConditionedTransitionBlock
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts


class TestDiffusionTransformer(unittest.TestCase):
    def test_attention_pair_bias_static_projection_matches_reference(self):
        torch.manual_seed(0)
        batch_size = 2
        n_res = 5
        c_q = 32
        c_s = 16
        c_z = 12
        no_heads = 4

        module = AttentionPairBias(
            c_q=c_q,
            c_k=c_q,
            c_v=c_q,
            c_s=c_s,
            c_z=c_z,
            c_hidden=8,
            no_heads=no_heads,
            use_ada_layer_norm=True,
        ).eval()
        z = torch.randn(batch_size, n_res, n_res, c_z)

        with torch.inference_mode():
            actual = module.prep_static_pair_bias(z)
            expected = module.linear_z(module.layer_norm_z(z)).permute(0, 3, 1, 2)

        self.assertEqual(actual.shape, (batch_size, no_heads, n_res, n_res))
        torch.testing.assert_close(actual, expected)

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
