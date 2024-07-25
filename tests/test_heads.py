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
from openfold3.core.np.token_atom_constants import (
    DNA_NUCLEOTIDE_TYPES,
    PROTEIN_RESTYPES,
    RNA_NUCLEOTIDE_TYPES,
    TOKEN_NAME_TO_ATOM_NAMES,
    TOKEN_TYPES,
)
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.model_implementations.af3_all_atom.config import (
    config,
    finetune3_config_update,
    max_atoms_per_token,
)
from tests.config import consts
from tests.data_utils import random_asym_ids


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
    # NOTE: Will periodically fail until Yeqing's fixes for unknown tokens are merged
    def test_auxiliary_heads_all_atom_shape(self):
        batch_size = 1
        n_token = consts.n_res

        # TODO: Move to data utils
        restypes_flat = torch.randint(0, len(TOKEN_TYPES), (n_token,))
        restypes_names = [TOKEN_TYPES[token_idx] for token_idx in restypes_flat]
        restypes_one_hot = torch.nn.functional.one_hot(
            restypes_flat,
            len(TOKEN_TYPES),
        )

        num_atoms_per_token = torch.Tensor(
            [len(TOKEN_NAME_TO_ATOM_NAMES[name]) for name in restypes_names]
        )

        is_protein = torch.Tensor(
            [1 if name in PROTEIN_RESTYPES else 0 for name in restypes_names]
        )
        is_rna = torch.Tensor(
            [1 if name in RNA_NUCLEOTIDE_TYPES else 0 for name in restypes_names]
        )
        is_dna = torch.Tensor(
            [1 if name in DNA_NUCLEOTIDE_TYPES else 0 for name in restypes_names]
        )

        n_atom = torch.max(torch.sum(num_atoms_per_token, dim=-1)).int().item()

        c_s = config.globals.c_s
        c_z = config.globals.c_z

        config.update(finetune3_config_update)
        heads_config = config.model.heads
        heads_config.distogram.enabled = True
        aux_head = AuxiliaryHeadsAllAtom(heads_config).eval()

        si_input = torch.ones(batch_size, n_token, c_s)
        si = torch.ones(batch_size, n_token, c_s)
        zij = torch.ones(batch_size, n_token, n_token, c_z)
        x_pred = torch.randn(batch_size, n_atom, 3)

        outputs = {
            "si_trunk": si,
            "zij_trunk": zij,
            "x_pred": x_pred,
        }

        start_atom_index = (
            torch.cumsum(num_atoms_per_token, dim=-1) - num_atoms_per_token[..., 0]
        )

        token_mask = torch.ones(n_token)

        batch = {
            "asym_id": torch.Tensor(random_asym_ids(n_token)),
            "token_mask": token_mask,
            "start_atom_index": start_atom_index,
            "num_atoms_per_token": num_atoms_per_token,
            "restype": restypes_one_hot,
            "is_protein": is_protein,
            "is_dna": is_dna,
            "is_rna": is_rna,
            "is_ligand": torch.zeros(n_token),
            "is_atomized": torch.zeros(n_token),
        }

        def to_dtype_batch(t):
            return t.to(dtype=torch.float32).unsqueeze(0)

        batch = tensor_tree_map(to_dtype_batch, batch)

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
