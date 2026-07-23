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

from openfold3.core.model.feature_embedders.template_embedders import (
    TemplatePairEmbedderAllAtom,
)
from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.utils.compare_utils import (
    assert_differences_along_last_dim,
    assert_max_abs_diff_small,
)


class TestTemplateEmbedderAllAtom(unittest.TestCase):
    def test_shape(self):
        batch_size = 2
        n_templ = 3
        n_token = 10

        of3_proj_entry = OF3ProjectEntry()
        of3_config = of3_proj_entry.get_model_config_with_presets()

        c_in = of3_config.architecture.template.template_pair_embedder.c_in

        embedder = TemplateEmbedderAllAtom(of3_config.architecture.template)

        batch = {
            "token_mask": torch.ones((batch_size, n_token)),
            "asym_id": torch.ones((batch_size, n_token)),
            "template_restype": torch.ones((batch_size, n_templ, n_token, 32)),
            "template_pseudo_beta_mask": torch.ones((batch_size, n_templ, n_token)),
            "template_backbone_frame_mask": torch.ones((batch_size, n_templ, n_token)),
            "template_distogram": torch.ones(
                (batch_size, n_templ, n_token, n_token, 39)
            ),
            "template_unit_vector": torch.ones(
                (batch_size, n_templ, n_token, n_token, 3)
            ),
        }

        z = torch.ones((batch_size, n_token, n_token, c_in))
        pair_mask = torch.randint(0, 2, size=(batch_size, n_token, n_token))

        t = embedder(batch=batch, z=z, pair_mask=pair_mask, chunk_size=None)

        self.assertTrue(t.shape == (batch_size, n_token, n_token, c_in))

    def test_multimer_masking(self):
        batch_size = 2
        n_token = 11
        n_templ = 4
        eps = 2e-2
        len_firsts = (6, 7)

        # load template embedder
        of3_proj_entry = OF3ProjectEntry()
        of3_config = of3_proj_entry.get_model_config_with_presets()

        c_in = of3_config.architecture.template.template_pair_embedder.c_in
        template_pair_embedder_conf = (
            of3_config.architecture.template.template_pair_embedder
        )

        asym_id = torch.ones((batch_size, n_token))
        for it, len_first in enumerate(len_firsts):
            for i in range(len_first):
                asym_id[it][i] += 1
        # construct dummy batch
        batch = {
            "token_mask": torch.ones((batch_size, n_token)),
            "asym_id": asym_id,
            "template_restype": torch.ones((batch_size, n_templ, n_token, 32)),
            "template_pseudo_beta_mask": torch.randn((batch_size, n_templ, n_token)),
            "template_backbone_frame_mask": torch.randn((batch_size, n_templ, n_token)),
            "template_distogram": torch.randn(
                (batch_size, n_templ, n_token, n_token, 39)
            ),
            "template_unit_vector": torch.randn(
                (batch_size, n_templ, n_token, n_token, 3)
            ),
        }
        z = torch.ones((batch_size, n_token, n_token, c_in))

        # construct default, monomer-only template_pair_embedder
        monomer_embedder = TemplatePairEmbedderAllAtom(**template_pair_embedder_conf)
        monomer_embedded = monomer_embedder(batch=batch, z=z)
        for it, len_first in enumerate(len_firsts):
            # there is information for all residues inside the monomers
            repeated_first = (
                monomer_embedded[it, 0, 0, 0, :]
                .squeeze(0)
                .tile([n_templ, len_first - 1, len_first - 1, 1])
            )
            repeated_last = (
                monomer_embedded[it, 0, -1, -1, :]
                .squeeze(0)
                .tile([n_templ, n_token - len_first - 1, n_token - len_first - 1, 1])
            )
            top_left = monomer_embedded[it, :, 1:len_first, 1:len_first, :].squeeze()
            bottom_right = monomer_embedded[
                it, :, len_first:-1, len_first:-1, :
            ].squeeze()
            assert_differences_along_last_dim(repeated_first, top_left)
            assert_differences_along_last_dim(repeated_last, bottom_right)

            # off the monomers is masked and always the same!
            repeated_off_top_right = (
                monomer_embedded[it, 0, len_first, len_first - 1, :]
                .squeeze(0)
                .tile([n_templ, len_first, n_token - len_first, 1])
            )
            repeated_off_bottom_left = (
                monomer_embedded[it, 0, len_first, len_first - 1, :]
                .squeeze(0)
                .tile([n_templ, n_token - len_first, len_first, 1])
            )
            top_right = monomer_embedded[it, :, :len_first, len_first:, :].squeeze()
            bottom_left = monomer_embedded[it, :, len_first:, :len_first, :].squeeze()
            assert_max_abs_diff_small(repeated_off_top_right, top_right, eps)
            assert_max_abs_diff_small(repeated_off_bottom_left, bottom_left, eps)

        # unmasking changes the off-diagonal!
        template_pair_embedder_conf["unmasked"] = True
        multimer_embedder = TemplatePairEmbedderAllAtom(**template_pair_embedder_conf)
        multimer_embedded = multimer_embedder(batch=batch, z=z)
        for it, len_first in enumerate(len_firsts):
            # there is information for all residues inside the monomers
            # **1**:len_first, lenf_first:**-1** --> do not check first and last again
            repeated_first = (
                multimer_embedded[it, 0, 0, 0, :]
                .squeeze(0)
                .tile([n_templ, len_first - 1, len_first - 1, 1])
            )
            repeated_last = (
                multimer_embedded[it, 0, -1, -1, :]
                .squeeze(0)
                .tile([n_templ, n_token - len_first - 1, n_token - len_first - 1, 1])
            )
            top_left = multimer_embedded[it, :, 1:len_first, 1:len_first, :].squeeze()
            bottom_right = multimer_embedded[
                it, :, len_first:-1, len_first:-1, :
            ].squeeze()
            assert_differences_along_last_dim(repeated_first, top_left)
            assert_differences_along_last_dim(repeated_last, bottom_right)

            # off the monomers is unmasked and always different again!
            repeated_off_top_right = (
                multimer_embedded[it, 0, len_first, len_first - 1, :]
                .squeeze(0)
                .tile([n_templ, len_first, n_token - len_first, 1])
            )
            repeated_off_bottom_left = (
                multimer_embedded[it, 0, len_first, len_first - 1, :]
                .squeeze(0)
                .tile([n_templ, n_token - len_first, len_first - 1, 1])
            )
            top_right = multimer_embedded[it, :, :len_first, len_first:, :].squeeze()
            bottom_left = multimer_embedded[
                it, :, len_first:, : len_first - 1, :
            ].squeeze()
            assert_differences_along_last_dim(repeated_off_top_right, top_right)
            assert_differences_along_last_dim(repeated_off_bottom_left, bottom_left)


if __name__ == "__main__":
    unittest.main()
