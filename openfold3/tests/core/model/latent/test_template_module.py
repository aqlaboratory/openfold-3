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

import importlib
import os
import unittest
from unittest import mock

import torch

from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry


class TestTemplateEmbedderAllAtom(unittest.TestCase):
    @staticmethod
    def _run_embedder_with_env(
        embedder,
        batch,
        z,
        pair_mask,
        env: dict[str, str],
    ) -> torch.Tensor:
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            with mock.patch.dict(os.environ, env), torch.no_grad():
                return embedder(
                    batch=batch,
                    z=z,
                    pair_mask=pair_mask,
                    inplace_safe=True,
                )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

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

    @unittest.skipUnless(
        torch.cuda.is_available() and importlib.util.find_spec("triton") is not None,
        "Requires CUDA and Triton",
    )
    def test_template_fast_paths_match_eager(self):
        torch.manual_seed(23)
        batch_size = 1
        n_templ = 2
        n_token = 64

        of3_proj_entry = OF3ProjectEntry()
        of3_config = of3_proj_entry.get_model_config_with_presets()
        template_config = of3_config.architecture.template
        c_in = template_config.template_pair_embedder.c_in

        embedder = TemplateEmbedderAllAtom(template_config).cuda().eval()
        batch = {
            "token_mask": torch.ones((batch_size, n_token), device="cuda"),
            "asym_id": torch.ones((batch_size, n_token), device="cuda"),
            "template_restype": torch.randn(
                batch_size, n_templ, n_token, 32, device="cuda"
            ),
            "template_pseudo_beta_mask": torch.ones(
                batch_size, n_templ, n_token, device="cuda"
            ),
            "template_backbone_frame_mask": torch.ones(
                batch_size, n_templ, n_token, device="cuda"
            ),
            "template_distogram": torch.randn(
                batch_size, n_templ, n_token, n_token, 39, device="cuda"
            ),
            "template_unit_vector": torch.randn(
                batch_size, n_templ, n_token, n_token, 3, device="cuda"
            ),
        }
        z = torch.randn(batch_size, n_token, n_token, c_in, device="cuda")
        pair_mask = torch.ones(batch_size, n_token, n_token, device="cuda")

        eager = self._run_embedder_with_env(
            embedder,
            batch,
            z,
            pair_mask,
            {
                "OPENFOLD3_FUSED_LN_LINEAR": "0",
                "OPENFOLD3_FUSED_SWIGLU_TRANSITION": "0",
            },
        )
        fused = self._run_embedder_with_env(
            embedder,
            batch,
            z,
            pair_mask,
            {
                "OPENFOLD3_FUSED_LN_LINEAR": "1",
                "OPENFOLD3_FUSED_SWIGLU_TRANSITION": "1",
            },
        )

        torch.testing.assert_close(fused, eager, atol=2e-4, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
