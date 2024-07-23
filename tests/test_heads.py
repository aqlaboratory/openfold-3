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
from openfold3.model_implementations.af3_all_atom.config import config
from tests.config import consts

# a few additional consts for stronger tests with token_to_atom_idx including modules
# PerResidueLDDAllAtom, ExperimentallyResolvedHeadAllAtom, and AuxiliaryHeadsAllAtom
n_token = consts.n_res

# Varying number of atoms per given token.
# 1 smallest (small molecules heavy atoms), 22 largest (Adenosine with Phosphate).
dummy_atom_per_token = torch.randint(1, 22, (n_token,))


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
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = torch.sum(dummy_atom_per_token).item()

        c_s = consts.c_s
        c_out = 50

        plddt_head = PerResidueLDDAllAtom(c_s, c_out)

        token_identity = torch.eye(n_token)  # shape: [n_token, n_token]
        token_to_atom_expansion = token_identity.repeat_interleave(
            dummy_atom_per_token, dim=0
        )
        token_to_atom_idx = token_to_atom_expansion.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # adding batch size. shape: [batch_size, n_atom, n_token]

        si = torch.ones((batch_size, n_token, c_s))
        out = plddt_head(si, token_to_atom_idx)

        expected_shape = (batch_size, n_atom, c_out)
        np.testing.assert_array_equal(out.shape, expected_shape)


class TestExperimentallyResolvedHeadAllAtom(unittest.TestCase):
    def test_experimentally_resolved_head_all_atom_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = torch.sum(dummy_atom_per_token).item()
        c_s = consts.c_s
        c_out = 50

        plddt_head = ExperimentallyResolvedHeadAllAtom(c_s, c_out)

        token_identity = torch.eye(n_token)  # shape: [n_token, n_token]
        token_to_atom_expansion = token_identity.repeat_interleave(
            dummy_atom_per_token, dim=0
        )
        token_to_atom_idx = token_to_atom_expansion.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # adding batch_size [batch_size, n_atom, n_token]

        si = torch.ones((batch_size, n_token, c_s))
        out = plddt_head(si, token_to_atom_idx)

        expected_shape = (batch_size, n_atom, c_out)
        np.testing.assert_array_equal(out.shape, expected_shape)


class TestPairformerEmbedding(unittest.TestCase):
    def test_pairformer_embedding_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res

        c_s = config.globals.c_s
        c_z = config.globals.c_z

        pair_emb = PairformerEmbedding(**config.model.heads.pairformer_embedding).eval()

        si_input = torch.ones(batch_size, n_token, c_s)
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
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = torch.sum(dummy_atom_per_token).item()

        c_s = config.globals.c_s
        c_z = config.globals.c_z

        heads_config = config.model.heads
        heads_config.pae.enabled = True
        aux_head = AuxiliaryHeadsAllAtom(heads_config).eval()

        si_input = torch.ones(batch_size, n_token, c_s)
        si = torch.ones(batch_size, n_token, c_s)
        zij = torch.ones(batch_size, n_token, n_token, c_z)
        outputs = {
            "single": si,
            "pair": zij,
        }

        x_pred = torch.ones(batch_size, n_atom, 3)

        token_representative_atom_idx = torch.zeros((n_token, n_atom))
        token_idx = torch.arange(n_token)
        atom_idx = torch.cumsum(dummy_atom_per_token, dim=0) - 1
        token_representative_atom_idx[token_idx, atom_idx] = (
            1  # each token gets its representative atom
        )
        token_representative_atom_idx = token_representative_atom_idx.unsqueeze(
            0
        ).repeat(batch_size, 1, 1)

        token_identity = torch.eye(n_token)  # shape: [n_token, n_token]
        token_to_atom_expansion = token_identity.repeat_interleave(
            dummy_atom_per_token, dim=0
        )
        token_to_atom_idx = token_to_atom_expansion.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # adding batch_size shape: [batch_size, n_atom, n_token]

        single_mask = torch.randint(
            0,
            2,
            size=(
                batch_size,
                n_token,
            ),
        )
        pair_mask = torch.randint(0, 2, size=(batch_size, n_token, n_token))

        aux_out = aux_head(
            si_input,
            outputs,
            x_pred,
            token_representative_atom_idx,
            token_to_atom_idx,
            single_mask,
            pair_mask,
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
