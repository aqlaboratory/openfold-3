import torch
import unittest 
import numpy as np

from openfold3.core.model.heads.token_heads import (
    AuxiliaryHeads, 
    Pairformer_Embedding, 
    PAEHead, 
    PDEHead, 
    PerResidueLDDTHead, 
    ExperimentallyResolvedHead,
)

from tests.config import consts 

class TestPAEHead(unittest.TestCase):
    def test_paehead_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_z = consts.c_z
        c_out = 50

        pae_head = PAEHead(c_z, c_out)

        zij = torch.ones((batch_size, n_token, n_token, c_z))
        out = pae_head(zij)

        expected_shape = (batch_size, n_token, n_token, c_out)
        np.testing.assert_array_equal(out.shape, expected_shape)

class TestPDEHead(unittest.TestCase):
    def test_pdehead_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_z = consts.c_z
        c_out = 50

        pde_head = PDEHead(c_z, c_out)

        zij = torch.ones((batch_size, n_token, n_token, c_z))
        out = pde_head(zij)

        expected_shape = (batch_size, n_token, n_token, c_out)
        np.testing.assert_array_equal(out.shape, expected_shape)

class TestPLDDTHead(unittest.TestCase):
    def test_plddthead_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
        c_s = consts.c_s
        c_out = 50

        plddt_head = PerResidueLDDTHead(c_s, c_out)

        token_identity = torch.eye(n_token) #shape: [n_token, n_token]
        token_to_atom_expansion = token_identity.repeat_interleave(4, dim = 0) #for a simple test, assume that each token has 4 atoms for all tokens. shape: [n_atom, n_token] with n_atom == n_token * 4 
        token_to_atom_idx = token_to_atom_expansion.unsqueeze(0).repeat(batch_size, 1, 1) #adding batch size. shape: [batch_size, n_atom, n_token]

        si = torch.ones((batch_size, n_token, c_s))
        out = plddt_head(si, token_to_atom_idx)

        expected_shape = (batch_size, n_atom, c_out)
        np.testing.assert_array_equal(out.shape, expected_shape)

class TestResolvedHead(unittest.TestCase):
    def test_resolved_head_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
        c_s = consts.c_s
        c_out = 50

        plddt_head = ExperimentallyResolvedHead(c_s, c_out)

        token_identity = torch.eye(n_token) #shape: [n_token, n_token]
        token_to_atom_expansion = token_identity.repeat_interleave(4, dim = 0) #for a simple test, assume that each token has 4 atoms. shape: [n_atom, n_token] 
        token_to_atom_idx = token_to_atom_expansion.unsqueeze(0).repeat(batch_size, 1, 1) #adding batch_size [batch_size, n_atom, n_token]

        si = torch.ones((batch_size, n_token, c_s))
        out = plddt_head(si, token_to_atom_idx)

        expected_shape = (batch_size, n_atom, c_out)
        np.testing.assert_array_equal(out.shape, expected_shape)

class TestPairformer_Embedding(unittest.TestCase):
    def test_pairformer_embedding_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s = consts.c_s
        c_z = consts.c_z

        min_bin = 3.25
        max_bin = 20.75
        no_bin = 15
        inf = 1e8

        c_hidden_pair_bias = 12
        no_heads_pair_bias = 3
        c_hidden_mul = 19
        c_hidden_pair_att = 14
        no_heads_pair = 7
        no_blocks = 2
        transition_n = 2
        pair_dropout = 0.25
        inf = 1e9
        eps = 1e-10

        pairformer_stack_config = {'c_s': c_s, 
                 'c_z': c_z, 
                 'c_hidden_pair_bias': c_hidden_pair_bias, 
                 'no_heads_pair_bias': no_heads_pair_bias, 
                 'c_hidden_mul': c_hidden_mul, 
                 'c_hidden_pair_att': c_hidden_pair_att, 
                 'no_heads_pair': no_heads_pair, 
                 'no_blocks': no_blocks, 
                 'transition_n': transition_n, 
                 'pair_dropout': pair_dropout, 
                 'fuse_projection_weights': False, 
                 'blocks_per_ckpt': None, 
                 'inf': inf, 
                 'eps': eps
                 }

        pair_emb = Pairformer_Embedding(min_bin, max_bin, no_bin, inf, c_s, c_z, pairformer_stack_config)

        si_input = torch.ones(batch_size, n_token, c_s)
        si = torch.ones(batch_size, n_token, c_s) 
        zij = torch.ones(batch_size, n_token, n_token, c_z) 
        x_pred = torch.ones(batch_size, n_token, 3)
        single_mask = torch.randint(0, 2, size=(batch_size, n_token,))
        pair_mask = torch.randint(0, 2, size=(batch_size, n_token, n_token))

        out_single, out_pair = pair_emb(si_input, 
                                        si, 
                                        zij, 
                                        x_pred, 
                                        single_mask, 
                                        pair_mask, 
                                        chuck_size = 4
                                        )
        
        expected_shape = (batch_size, n_token, c_s)
        np.testing.assert_array_equal(out_single.shape, expected_shape)
        
        expected_shape = (batch_size, n_token, n_token, c_z)
        np.testing.assert_array_equal(out_pair.shape, expected_shape)        

class TestAuxiliaryHeads(unittest.TestCase):
    def test_aux_head_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = consts.n_res * 4
        c_s = consts.c_s
        c_z = consts.c_z
        c_out = 50
        min_bin = 3.25
        max_bin = 20.75
        no_bin = 15
        inf = 1e8

        c_hidden_pair_bias = 12
        no_heads_pair_bias = 3
        c_hidden_mul = 19
        c_hidden_pair_att = 14
        no_heads_pair = 7
        no_blocks = 4
        transition_n = 2
        pair_dropout = 0.25
        inf = 1e9
        eps = 1e-10

        config = {
                'pae': 
                    {'c_z' : c_z, 
                     'c_out' : c_out
                     },
                'pde': 
                    {'c_z' : c_z, 
                     'c_out' : c_out
                     },
                'lddt': 
                    {'c_s' : c_s, 
                     'c_out' : c_out
                     },
                'distogram': 
                    {'c_z' : c_z, 
                     'c_out' : c_out
                     },
                'experimentally_resolved': 
                    {'c_s': c_s, 
                     'c_out': c_out
                     },
                'pairformer_embedding': 
                    {'min_bin': min_bin, 
                     'max_bin': max_bin, 
                     'no_bin': no_bin, 
                     'inf': inf, 
                     'c_s': c_s, 
                     'c_z': c_z, 
                     'config': {
                        'c_s': c_s, 
                        'c_z': c_z, 
                        'c_hidden_pair_bias': c_hidden_pair_bias, 
                        'no_heads_pair_bias': no_heads_pair_bias, 
                        'c_hidden_mul': c_hidden_mul, 
                        'c_hidden_pair_att': c_hidden_pair_att, 
                        'no_heads_pair': no_heads_pair, 
                        'no_blocks': no_blocks, 
                        'transition_n': transition_n, 
                        'pair_dropout': pair_dropout, 
                        'fuse_projection_weights': False, 
                        'blocks_per_ckpt': None, 
                        'inf': inf, 
                        'eps': eps
                        },
                     },
                'tm': 
                    {'enabled': False,
                     },
                  }
        
        aux_head = AuxiliaryHeads(config)

        si_input = torch.ones(batch_size, n_token, c_s)
        si = torch.ones(batch_size, n_token, c_s)
        zij = torch.ones(batch_size, n_token, n_token, c_z)
        x_pred = torch.ones(batch_size, n_token, 3)
        outputs = {'single': si, 'pair': zij, 'coordinates': x_pred}

        token_identity = torch.eye(n_token) #shape: [n_token, n_token]
        token_to_atom_expansion = token_identity.repeat_interleave(4, dim = 0) #for a simple test, assume that each token has 4 atoms. shape: [n_atom, n_token] 
        token_to_atom_idx = token_to_atom_expansion.unsqueeze(0).repeat(batch_size, 1, 1) #adding batch_size shape: [batch_size, n_atom, n_token]

        single_mask = torch.randint(0, 2, size=(batch_size, n_token,))
        pair_mask = torch.randint(0, 2, size=(batch_size, n_token, n_token))

        d = aux_head(si_input, 
                     outputs, 
                     token_to_atom_idx, 
                     single_mask, 
                     pair_mask, 
                     chuck_size = 4
                     )
        
        expected_shape_distogram = (batch_size, n_token, n_token, c_out)
        np.testing.assert_array_equal(d['distogram_logits'].shape, expected_shape_distogram)        

        expected_shape_pae = (batch_size, n_token, n_token, c_out)
        np.testing.assert_array_equal(d['pae_logits'].shape, expected_shape_pae)        

        expected_shape_pde = (batch_size, n_token, n_token, c_out)
        np.testing.assert_array_equal(d['pde_logits'].shape, expected_shape_pde)        

        expected_shape_plddt = (batch_size, n_atom, c_out)
        np.testing.assert_array_equal(d['plddt_logits'].shape, expected_shape_plddt)        

if __name__ == "__main__":
    unittest.main()
