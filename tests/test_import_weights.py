# Copyright 2021 AlQuraishi Laboratory
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

import os
import unittest
from pathlib import Path

import numpy as np
import torch

from openfold3.core.utils.import_weights import (
    import_jax_weights_,
    import_openfold_weights_,
)
from openfold3.projects import registry
from tests.config import monomer_consts
from tests import compare_utils


@compare_utils.skip_af2_test()
class TestImportWeights(unittest.TestCase):
    def test_import_jax_weights_(self):
        npz_path = (
            Path(__file__).parent.resolve()
            / f"../openfold3/resources/params/params_{monomer_consts.model_preset}.npz"
        )

        project_entry = registry.get_project_entry("af2_monomer")
        c = registry.make_config_with_presets(
            project_entry, [monomer_consts.model_preset]
        )
        c.globals.blocks_per_ckpt = None

        model = project_entry.model_runner(c, _compile=False).model
        model.eval()

        import_jax_weights_(model, npz_path, version=monomer_consts.model_preset)

        data = np.load(npz_path)
        prefix = "alphafold/alphafold_iteration/"

        test_pairs = [
            # Normal linear weight
            (
                torch.as_tensor(
                    data[prefix + "structure_module/initial_projection//weights"]
                ).transpose(-1, -2),
                model.structure_module.linear_in.weight,
            ),
            # Normal layer norm param
            (
                torch.as_tensor(
                    data[prefix + "evoformer/prev_pair_norm//offset"],
                ),
                model.recycling_embedder.layer_norm_z.bias,
            ),
            # From a stack
            (
                torch.as_tensor(
                    data[
                        prefix
                        + (
                            "evoformer/evoformer_iteration/outer_product_mean/"
                            "left_projection//weights"
                        )
                    ][1].transpose(-1, -2)
                ),
                model.evoformer.blocks[1].outer_product_mean.linear_1.weight,
            ),
        ]

        for w_alpha, w_repro in test_pairs:
            self.assertTrue(torch.all(w_alpha == w_repro))

    def test_import_openfold_weights_(self):
        model_name = "initial_training"
        pt_path = (
            Path(__file__).parent.resolve()
            / f"../openfold3/resources/openfold_params/{model_name}.pt"
        )

        if os.path.exists(pt_path):
            project_entry = registry.get_project_entry("af2_monomer")
            c = registry.make_config_with_presets(project_entry, [model_name])
            c.globals.blocks_per_ckpt = None
            model = project_entry.model_runner(c, _compile=False).model
            model.eval()

            d = torch.load(pt_path, weights_only=True)
            import_openfold_weights_(
                model=model,
                state_dict=d,
            )
