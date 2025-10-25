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

import unittest

import pytest
import torch

from openfold3.core.model.feature_embedders.input_embedders import (
    InputEmbedderAllAtom,
    MSAModuleEmbedder,
)
from openfold3.core.model.feature_embedders.template_embedders import (
    TemplatePairEmbedderAllAtom,
)
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.tests.config import consts
from openfold3.tests.data_utils import random_asym_ids, random_of3_features


class TestInputEmbedderAllAtom:
    def test_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res

        proj_entry = OF3ProjectEntry()
        of3_config = proj_entry.get_model_config_with_presets()

        c_s_input = of3_config.architecture.input_embedder.c_s_input
        c_s = of3_config.architecture.input_embedder.c_s
        c_z = of3_config.architecture.input_embedder.c_z

        batch = random_of3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )

        ie = InputEmbedderAllAtom(**of3_config.architecture.input_embedder)

        s_input, s, z = ie(batch=batch)

        assert s_input.shape == (batch_size, n_token, c_s_input)
        assert s.shape == (batch_size, n_token, c_s)
        assert z.shape == (batch_size, n_token, n_token, c_z)


class TestMSAModuleEmbedder:
    @pytest.mark.parametrize(
        "n_total_msa_seq,subsample_all_msa",
        [(200, False), (1, False), (15000, True), (100, True), (1, True)],
    )
    def test_msa_module_embedder_shape_and_sampling(
        self, n_total_msa_seq, subsample_all_msa
    ):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_token = 768
        c_s_input = c_token + 65
        one_hot_dim = 32

        proj_entry = OF3ProjectEntry()
        of3_config = proj_entry.get_model_config_with_presets()

        msa_emb_config = of3_config.architecture.msa.msa_module_embedder
        msa_emb_config.update({"c_s_input": c_s_input})

        if subsample_all_msa:
            msa_emb_config.update(
                {
                    "subsample_main_msa": False,
                    "subsample_all_msa": True,
                    "min_subsampled_all_msa": 1024,
                    "max_subsampled_all_msa": 1024,
                }
            )

        batch_asym_ids = [
            torch.as_tensor(random_asym_ids(n_token)) for _ in range(batch_size)
        ]
        batch_asym_ids = torch.stack(batch_asym_ids)

        if n_total_msa_seq > 0:
            num_paired = torch.randint(
                low=max(0, n_total_msa_seq // 4),
                high=max(1, n_total_msa_seq // 2),
                size=(batch_size,),
            )
        else:
            num_paired = torch.zeros((batch_size,), dtype=torch.long)

        batch = {
            "msa": torch.rand((batch_size, n_total_msa_seq, n_token, one_hot_dim)),
            "has_deletion": torch.ones((batch_size, n_total_msa_seq, n_token)),
            "deletion_value": torch.rand((batch_size, n_total_msa_seq, n_token)),
            "msa_mask": torch.ones((batch_size, n_total_msa_seq, n_token)),
            "num_paired_seqs": num_paired,
            "asym_id": batch_asym_ids,
        }

        s_input = torch.rand(batch_size, n_token, c_s_input)

        ie = MSAModuleEmbedder(**msa_emb_config)
        msa, msa_mask = ie(batch=batch, s_input=s_input)

        n_sampled_seqs = msa.shape[-3]

        assert msa.shape == (batch_size, n_sampled_seqs, n_token, msa_emb_config.c_m)
        assert msa_mask.shape == (batch_size, n_sampled_seqs, n_token)

        if subsample_all_msa:
            expected = min(n_total_msa_seq, 1024)
            assert n_sampled_seqs == expected
        else:
            if n_total_msa_seq == 0:
                assert n_sampled_seqs == 0
            else:
                max_paired_seqs = torch.max(batch["num_paired_seqs"])
                assert (n_sampled_seqs > max_paired_seqs) and (
                    n_sampled_seqs <= n_total_msa_seq
                )


class TestTemplatePairEmbedders:
    def test_all_atom(self):
        batch_size = 2
        n_templ = 3
        n_token = 10

        proj_entry = OF3ProjectEntry()
        of3_config = proj_entry.get_model_config_with_presets()

        c_in = of3_config.architecture.template.template_pair_embedder.c_in
        c_t = of3_config.architecture.template.template_pair_embedder.c_out

        tpe = TemplatePairEmbedderAllAtom(
            **of3_config.architecture.template.template_pair_embedder
        )

        batch = {
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

        emb = tpe(batch, z)

        assert emb.shape == (batch_size, n_templ, n_token, n_token, c_t)


if __name__ == "__main__":
    unittest.main()
