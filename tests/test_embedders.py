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

import unittest

import torch

from openfold3.core.model.feature_embedders.input_embedders import (
    InputEmbedder,
    InputEmbedderAllAtom,
    InputEmbedderMultimer,
    MSAModuleEmbedder,
    PreembeddingEmbedder,
    RecyclingEmbedder,
)
from openfold3.core.model.feature_embedders.template_embedders import (
    TemplatePairEmbedderAllAtom,
    TemplatePairEmbedderMonomer,
    TemplatePairEmbedderMultimer,
    TemplateSingleEmbedderMonomer,
    TemplateSingleEmbedderMultimer,
)
from openfold3.projects import registry
from tests.config import consts, monomer_consts, multimer_consts
from tests.data_utils import random_af3_features, random_asym_ids, random_template_feats


class TestInputEmbedder(unittest.TestCase):
    def test_shape(self):
        c_z = 5
        c_m = 7
        b = 13
        n_res = 17
        n_clust = 19

        monomer_project_entry = registry.get_project_entry(monomer_consts.model_name)
        config = registry.make_config_with_presets(
            monomer_project_entry, [monomer_consts.model_preset]
        )
        input_emb_config = config.model.input_embedder
        input_emb_config.update({"c_z": c_z, "c_m": c_m})

        tf = torch.rand((b, n_res, input_emb_config.tf_dim))
        ri = torch.rand((b, n_res))
        msa = torch.rand((b, n_clust, n_res, input_emb_config.msa_dim))
        asym_ids_flat = torch.Tensor(random_asym_ids(n_res))
        asym_id = torch.tile(asym_ids_flat.unsqueeze(0), (b, 1))
        entity_id = asym_id
        sym_id = torch.zeros_like(entity_id)

        ie = InputEmbedder(**input_emb_config)
        msa_emb, pair_emb = ie(tf=tf, ri=ri, msa=msa, inplace_safe=False)

        self.assertTrue(msa_emb.shape == (b, n_clust, n_res, c_m))
        self.assertTrue(pair_emb.shape == (b, n_res, n_res, c_z))

        multimer_project_entry = registry.get_project_entry(multimer_consts.model_name)
        config = registry.make_config_with_presets(
            multimer_project_entry, [multimer_consts.model_preset]
        )
        input_emb_config = config.model.input_embedder
        input_emb_config.update({"c_z": c_z, "c_m": c_m})

        ie = InputEmbedderMultimer(**input_emb_config)

        batch = {
            "target_feat": torch.rand((b, n_res, input_emb_config.tf_dim)),
            "residue_index": ri,
            "msa_feat": torch.rand((b, n_clust, n_res, input_emb_config.msa_dim)),
            "asym_id": asym_id,
            "entity_id": entity_id,
            "sym_id": sym_id,
        }
        msa_emb, pair_emb = ie(batch)

        self.assertTrue(msa_emb.shape == (b, n_clust, n_res, c_m))
        self.assertTrue(pair_emb.shape == (b, n_res, n_res, c_z))


class TestInputEmbedderAllAtom(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res

        af3_proj = registry.get_project_entry("af3_all_atom")
        af3_proj_config = af3_proj.get_config_with_preset()
        af3_config = af3_proj_config.model

        c_s_input = af3_config.architecture.input_embedder.c_s_input
        c_s = af3_config.architecture.input_embedder.c_s
        c_z = af3_config.architecture.input_embedder.c_z

        batch = random_af3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
        )

        ie = InputEmbedderAllAtom(**af3_config.architecture.input_embedder)

        s_input, s, z = ie(batch=batch)

        self.assertTrue(s_input.shape == (batch_size, n_token, c_s_input))
        self.assertTrue(s.shape == (batch_size, n_token, c_s))
        self.assertTrue(z.shape == (batch_size, n_token, n_token, c_z))


class TestMSAModuleEmbedder(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_total_msa_seq = 200
        c_token = 768
        c_s_input = c_token + 65
        one_hot_dim = 32

        proj_entry = registry.get_project_entry("af3_all_atom")
        af3_proj_config = proj_entry.get_config_with_preset()
        af3_config = af3_proj_config.model

        msa_emb_config = af3_config.architecture.msa.msa_module_embedder
        msa_emb_config.update({"c_s_input": c_s_input})

        batch = {
            "msa": torch.rand((batch_size, n_total_msa_seq, n_token, one_hot_dim)),
            "has_deletion": torch.ones((batch_size, n_total_msa_seq, n_token)),
            "deletion_value": torch.rand((batch_size, n_total_msa_seq, n_token)),
            "msa_mask": torch.ones((batch_size, n_total_msa_seq, n_token)),
            "num_paired_seqs": torch.randint(
                low=n_total_msa_seq // 4, high=n_total_msa_seq // 2, size=(batch_size,)
            ),
        }

        s_input = torch.rand(batch_size, n_token, c_s_input)

        ie = MSAModuleEmbedder(**msa_emb_config)

        msa, msa_mask = ie(batch=batch, s_input=s_input)
        n_sampled_seqs = msa.shape[-3]

        # Check that the number of sampled sequences is between the number of
        # uniprot seqs and the total number of sequences
        max_paired_seqs = torch.max(batch["num_paired_seqs"])
        self.assertTrue(
            (n_sampled_seqs > max_paired_seqs) & (n_sampled_seqs < n_total_msa_seq)
        )
        self.assertTrue(
            msa.shape == (batch_size, n_sampled_seqs, n_token, msa_emb_config.c_m)
        )
        self.assertTrue(msa_mask.shape == (batch_size, n_sampled_seqs, n_token))


class TestPreembeddingEmbedder(unittest.TestCase):
    def test_shape(self):
        tf_dim = 22
        preembedding_dim = 1280
        c_z = 4
        c_m = 6
        relpos_k = 10

        batch_size = 4
        num_res = 20

        tf = torch.rand((batch_size, num_res, tf_dim))
        ri = torch.rand((batch_size, num_res))
        preemb = torch.rand((batch_size, num_res, preembedding_dim))

        pe = PreembeddingEmbedder(
            tf_dim,
            preembedding_dim,
            c_z,
            c_m,
            relpos_k,
        )

        seq_emb, pair_emb = pe(tf, ri, preemb)
        self.assertTrue(seq_emb.shape == (batch_size, 1, num_res, c_m))
        self.assertTrue(pair_emb.shape == (batch_size, num_res, num_res, c_z))


class TestRecyclingEmbedder(unittest.TestCase):
    def test_shape(self):
        batch_size = 2
        n = 3
        c_z = 5
        c_m = 7
        min_bin = 0
        max_bin = 10
        no_bins = 9

        re = RecyclingEmbedder(
            c_m,
            c_z,
            min_bin,
            max_bin,
            no_bins,
        )

        m_1 = torch.rand((batch_size, n, c_m))
        z = torch.rand((batch_size, n, n, c_z))
        x = torch.rand((batch_size, n, 3))

        m_1, z = re(m_1, z, x)

        self.assertTrue(z.shape == (batch_size, n, n, c_z))
        self.assertTrue(m_1.shape == (batch_size, n, c_m))


class TestTemplateSingleEmbedders(unittest.TestCase):
    def test_shape(self):
        batch_size = 4
        n_templ = 4
        n_res = 256

        monomer_project_entry = registry.get_project_entry(monomer_consts.model_name)
        c = registry.make_config_with_presets(
            monomer_project_entry, [monomer_consts.model_preset]
        )
        c_m = c.model.template.template_single_embedder.c_out

        batch = random_template_feats(n_templ, n_res, batch_size=batch_size)
        batch = {k: torch.as_tensor(v) for k, v in batch.items()}

        tae = TemplateSingleEmbedderMonomer(
            c.model.template.template_single_embedder.c_in,
            c_m,
        )

        x = tae(batch)

        self.assertTrue(x.shape == (batch_size, n_templ, n_res, c_m))

        multimer_project_entry = registry.get_project_entry(multimer_consts.model_name)
        c = registry.make_config_with_presets(
            multimer_project_entry, [multimer_consts.model_preset]
        )
        c_m = c.model.template.template_single_embedder.c_out

        tae = TemplateSingleEmbedderMultimer(
            c.model.template.template_single_embedder.c_in,
            c_m,
        )

        x = tae(batch)
        x = x["template_single_embedding"]

        self.assertTrue(x.shape == (batch_size, n_templ, n_res, c_m))


class TestTemplatePairEmbedders(unittest.TestCase):
    def test_shape(self):
        batch_size = 2
        n_templ = 4
        n_res = 5

        monomer_project_entry = registry.get_project_entry(monomer_consts.model_name)
        c = registry.make_config_with_presets(
            monomer_project_entry, [monomer_consts.model_preset]
        )
        c_t = c.model.template.template_pair_embedder.c_out

        batch = random_template_feats(n_templ, n_res, batch_size=batch_size)
        batch = {k: torch.as_tensor(v) for k, v in batch.items()}

        tpe = TemplatePairEmbedderMonomer(**c.model.template.template_pair_embedder)

        x = tpe(
            batch=batch,
            distogram_config=c.model.template.distogram,
            use_unit_vector=False,
            inf=monomer_consts.inf,
            eps=monomer_consts.eps,
        )

        self.assertTrue(x.shape == (batch_size, n_templ, n_res, n_res, c_t))

        multimer_project_entry = registry.get_project_entry(multimer_consts.model_name)
        c = registry.make_config_with_presets(
            multimer_project_entry, [multimer_consts.model_preset]
        )
        c_z = c.model.template.template_pair_embedder.c_in
        c_t = c.model.template.template_pair_embedder.c_out

        z = torch.rand((batch_size, n_res, n_res, c_z))
        asym_ids = torch.as_tensor(random_asym_ids(n_res))
        asym_ids = torch.tile(asym_ids[None, :], (batch_size, 1))
        multichain_mask_2d = (asym_ids[..., None] == asym_ids[..., None, :]).to(
            dtype=z.dtype
        )

        tpe = TemplatePairEmbedderMultimer(**c.model.template.template_pair_embedder)

        x = tpe(
            batch=batch,
            distogram_config=c.model.template.distogram,
            query_embedding=z,
            multichain_mask_2d=multichain_mask_2d,
            inf=multimer_consts.inf,
        )

        self.assertTrue(x.shape == (batch_size, n_templ, n_res, n_res, c_t))

    def test_all_atom(self):
        batch_size = 2
        n_templ = 3
        n_token = 10

        proj_entry = registry.get_project_entry("af3_all_atom")
        af3_proj_config = proj_entry.get_config_with_preset()
        af3_config = af3_proj_config.model

        c_z = af3_config.architecture.template.template_pair_embedder.c_z
        c_t = af3_config.architecture.template.template_pair_embedder.c_out

        tpe = TemplatePairEmbedderAllAtom(
            **af3_config.architecture.template.template_pair_embedder
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

        z = torch.ones((batch_size, n_token, n_token, c_z))

        emb = tpe(batch, z)

        self.assertTrue(emb.shape == (batch_size, n_templ, n_token, n_token, c_t))


if __name__ == "__main__":
    unittest.main()
