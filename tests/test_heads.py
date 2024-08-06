import unittest

import numpy as np
import torch

from openfold3.core.model.heads.head_modules import AuxiliaryHeadsAllAtom
from openfold3.core.model.heads.prediction_heads import (
    ExperimentallyResolvedHeadAllAtom,
    PairformerEmbedding,
    PerResidueLDDAllAtom,
    PredictedAlignedErrorHead,
    PredictedDistanceErrorHead,
)
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
from openfold3.model_implementations import registry
from openfold3.model_implementations.af3_all_atom.config.base_config import (
    max_atoms_per_token,
)
from tests.config import consts
from tests.data_utils import random_af3_features


class TestPredictedAlignedErrorHead(unittest.TestCase):
    def test_predicted_aligned_error_head_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_z = consts.c_z
        c_out = 50

        pae_head = PredictedAlignedErrorHead(c_z, c_out)

        zij = torch.ones((batch_size, n_token, n_token, c_z))
        out = pae_head(zij)

        expected_shape = (batch_size, n_token, n_token, c_out)
        np.testing.assert_array_equal(out.shape, expected_shape)


class TestPredictedDistanceErrorHead(unittest.TestCase):
    def test_predicted_distance_error_head_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_z = consts.c_z
        c_out = 50

        pde_head = PredictedDistanceErrorHead(c_z, c_out)

        zij = torch.ones((batch_size, n_token, n_token, c_z))
        out = pde_head(zij)

        expected_shape = (batch_size, n_token, n_token, c_out)
        np.testing.assert_array_equal(out.shape, expected_shape)


class TestPLDDTHead(unittest.TestCase):
    def test_plddt_head_shape(self):
        batch_size = 1
        n_token = consts.n_res
        c_s = consts.c_s
        c_out = 50

        plddt_head = PerResidueLDDAllAtom(
            c_s, c_out, max_atoms_per_token=max_atoms_per_token.get()
        )

        si = torch.ones((batch_size, n_token, c_s))
        token_mask = torch.ones((batch_size, n_token))
        num_atoms_per_token = torch.randint(
            0, max_atoms_per_token.get(), (batch_size, n_token)
        )
        n_atom = torch.sum(num_atoms_per_token, dim=-1).int().item()

        max_atom_per_token_mask = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=token_mask,
            max_num_atoms_per_token=max_atoms_per_token.get(),
        )

        out = plddt_head(s=si, max_atom_per_token_mask=max_atom_per_token_mask)

        expected_shape = (batch_size, n_atom, c_out)
        np.testing.assert_array_equal(out.shape, expected_shape)


class TestExperimentallyResolvedHeadAllAtom(unittest.TestCase):
    def test_experimentally_resolved_head_all_atom_shape(self):
        batch_size = 1
        n_token = consts.n_res
        c_s = consts.c_s
        c_out = 50

        exp_res_head = ExperimentallyResolvedHeadAllAtom(
            c_s, c_out, max_atoms_per_token=max_atoms_per_token.get()
        )

        si = torch.ones((batch_size, n_token, c_s))
        token_mask = torch.ones((batch_size, n_token))
        num_atoms_per_token = torch.randint(
            0, max_atoms_per_token.get(), (batch_size, n_token)
        )
        n_atom = torch.sum(num_atoms_per_token, dim=-1).int().item()

        max_atom_per_token_mask = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=token_mask,
            max_num_atoms_per_token=max_atoms_per_token.get(),
        )

        out = exp_res_head(s=si, max_atom_per_token_mask=max_atom_per_token_mask)

        expected_shape = (batch_size, n_atom, c_out)
        np.testing.assert_array_equal(out.shape, expected_shape)


class TestPairformerEmbedding(unittest.TestCase):
    def test_pairformer_embedding_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res

        config = registry.make_config_with_preset("af3_all_atom")

        c_s_input = config.globals.c_s_input
        c_s = config.globals.c_s
        c_z = config.globals.c_z

        pair_emb = PairformerEmbedding(**config.model.heads.pairformer_embedding).eval()

        si_input = torch.ones(batch_size, n_token, c_s_input)
        si = torch.ones(batch_size, n_token, c_s)
        zij = torch.ones(batch_size, n_token, n_token, c_z)
        x_pred = torch.ones(batch_size, n_token, 3)
        single_mask = torch.randint(
            0,
            2,
            size=(
                batch_size,
                n_token,
            ),
        )
        pair_mask = torch.randint(0, 2, size=(batch_size, n_token, n_token))

        out_single, out_pair = pair_emb(
            si_input, si, zij, x_pred, single_mask, pair_mask, chunk_size=4
        )

        expected_shape_single = (batch_size, n_token, c_s)
        np.testing.assert_array_equal(out_single.shape, expected_shape_single)

        expected_shape_pair = (batch_size, n_token, n_token, c_z)
        np.testing.assert_array_equal(out_pair.shape, expected_shape_pair)


class TestAuxiliaryHeadsAllAtom(unittest.TestCase):
    def test_auxiliary_heads_all_atom_shape(self):
        batch_size = 1
        n_token = consts.n_res
        n_msa = 10
        n_templ = 3

        config = registry.make_config_with_preset("af3_all_atom")
        c_s_input = config.globals.c_s_input
        c_s = config.globals.c_s
        c_z = config.globals.c_z

        batch = random_af3_features(
            batch_size=batch_size, n_token=n_token, n_msa=n_msa, n_templ=n_templ
        )
        n_atom = torch.max(batch["num_atoms_per_token"].sum(dim=-1)).int().item()

        heads_config = config.model.heads
        heads_config.pae.enabled = True
        aux_head = AuxiliaryHeadsAllAtom(heads_config).eval()

        si_input = torch.ones(batch_size, n_token, c_s_input)
        si = torch.ones(batch_size, n_token, c_s)
        zij = torch.ones(batch_size, n_token, n_token, c_z)
        x_pred = torch.randn(batch_size, n_atom, 3)

        outputs = {
            "si_trunk": si,
            "zij_trunk": zij,
            "x_pred": x_pred,
        }

        aux_out = aux_head(
            batch,
            si_input,
            outputs,
            chunk_size=4,
        )

        expected_shape_distogram = (
            batch_size,
            n_token,
            n_token,
            heads_config.distogram.c_out,
        )
        np.testing.assert_array_equal(
            aux_out["distogram_logits"].shape, expected_shape_distogram
        )

        expected_shape_pae = (batch_size, n_token, n_token, heads_config.pae.c_out)
        np.testing.assert_array_equal(aux_out["pae_logits"].shape, expected_shape_pae)

        expected_shape_pde = (batch_size, n_token, n_token, heads_config.pde.c_out)
        np.testing.assert_array_equal(aux_out["pde_logits"].shape, expected_shape_pde)

        expected_shape_plddt = (batch_size, n_atom, heads_config.lddt.c_out)
        np.testing.assert_array_equal(
            aux_out["plddt_logits"].shape, expected_shape_plddt
        )

        expected_shape_exp_res = (
            batch_size,
            n_atom,
            heads_config.experimentally_resolved.c_out,
        )
        np.testing.assert_array_equal(
            aux_out["experimentally_resolved_logits"].shape, expected_shape_exp_res
        )


if __name__ == "__main__":
    unittest.main()
