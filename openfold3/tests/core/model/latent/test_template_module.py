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

from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.utils.data_utils import random_template_features


def _random_template_batch(
    batch_size: int, n_templ: int, n_token: int
) -> dict[str, torch.Tensor]:
    """Random template features plus two chains, so that inter-chain masking of
    the pair features is exercised."""
    asym_id = torch.ones((batch_size, n_token))
    asym_id[:, n_token // 2 :] = 2.0
    return {
        "token_mask": torch.ones((batch_size, n_token)),
        "asym_id": asym_id,
        **random_template_features(batch_size, n_templ, n_token),
    }


class TestTemplateEmbedderAllAtom:
    @pytest.fixture
    def embedder(self, seeded_rng):
        of3_config = OF3ProjectEntry().get_model_config_with_presets()
        return TemplateEmbedderAllAtom(of3_config.architecture.template).eval()

    @staticmethod
    def _random_pair_inputs(embedder, batch_size, n_token):
        c_in = embedder.config.template_pair_embedder.c_in
        z = torch.randn((batch_size, n_token, n_token, c_in))
        pair_mask = torch.randint(0, 2, size=(batch_size, n_token, n_token)).float()
        return z, pair_mask

    def test_shape(self, embedder):
        batch_size = 2
        n_templ = 3
        n_token = 10

        batch = _random_template_batch(batch_size, n_templ, n_token)
        z, pair_mask = self._random_pair_inputs(embedder, batch_size, n_token)

        t = embedder(batch=batch, z=z, pair_mask=pair_mask, chunk_size=None)

        assert t.shape == z.shape
        assert torch.isfinite(t).all()

    def test_duplicated_templates_match_single_template(self, embedder):
        """Templates are averaged, so a template given twice embeds like once."""
        batch_size = 2
        n_token = 8

        batch_single = _random_template_batch(batch_size, 1, n_token)
        batch_double = {
            k: v.repeat_interleave(2, dim=1) if k.startswith("template_") else v
            for k, v in batch_single.items()
        }
        z, pair_mask = self._random_pair_inputs(embedder, batch_size, n_token)

        with torch.no_grad():
            t_single = embedder(
                batch=batch_single, z=z, pair_mask=pair_mask, chunk_size=None
            )
            t_double = embedder(
                batch=batch_double, z=z, pair_mask=pair_mask, chunk_size=None
            )

        torch.testing.assert_close(t_double, t_single, atol=1e-5, rtol=1e-4)

    def test_token_permutation_equivariance(self, embedder):
        """Permuting tokens in every input permutes both token axes of the output.

        The distogram and unit-vector features are built from coordinates inside
        the embedder, so this checks they are computed consistently per token pair.
        """
        batch_size = 2
        n_templ = 2
        n_token = 8

        batch = _random_template_batch(batch_size, n_templ, n_token)
        z, pair_mask = self._random_pair_inputs(embedder, batch_size, n_token)
        perm = torch.randperm(n_token)

        batch_perm = {
            k: v[:, :, perm] if k.startswith("template_") else v[:, perm]
            for k, v in batch.items()
        }
        z_perm = z[:, perm][:, :, perm]
        pair_mask_perm = pair_mask[:, perm][:, :, perm]

        with torch.no_grad():
            t = embedder(batch=batch, z=z, pair_mask=pair_mask, chunk_size=None)
            t_perm = embedder(
                batch=batch_perm, z=z_perm, pair_mask=pair_mask_perm, chunk_size=None
            )

        torch.testing.assert_close(t_perm, t[:, perm][:, :, perm], atol=1e-5, rtol=1e-4)
