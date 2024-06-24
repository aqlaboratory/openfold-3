import torch
import unittest 

from openfold3.core.model.heads.token_heads import (
    AuxiliaryHeads, 
    Pairformer_Embedding, 
    PAEHead, 
    PDEHead, 
    PerResidueLDDTHead, 
    ExperimentallyResolvedHead,
    DistogramHead
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

        self.assertTrue(out.shape == (batch_size, n_token, n_token, c_out))

class TestPDEHead(unittest.TestCase):
    def test_pdehead_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_z = consts.c_z
        c_out = 50

        pde_head = PDEHead(c_z, c_out)

        zij = torch.ones((batch_size, n_token, n_token, c_z))
        out = pde_head(zij)

        self.assertTrue(out.shape == (batch_size, n_token, n_token, c_out))

class TestPLDDTHead(unittest.TestCase):
    def test_plddthead_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
        c_s = consts.c_s
        c_out = 50

        plddt_head = PerResidueLDDTHead(c_s, c_out)

        token_to_atom_idx = torch.eye(n_token).repeat_interleave(4, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
        si = torch.ones((batch_size, n_token, c_s))
        out = plddt_head(si, token_to_atom_idx)

        self.assertTrue(out.shape == (batch_size, n_atom, c_out))

class TestResolvedHead(unittest.TestCase):
    def test_resolved_head_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
        c_s = consts.c_s
        c_out = 50

        plddt_head = ExperimentallyResolvedHead(c_s, c_out)

        token_to_atom_idx = torch.eye(n_token).repeat_interleave(4, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
        si = torch.ones((batch_size, n_token, c_s))
        out = plddt_head(si, token_to_atom_idx)

        self.assertTrue(out.shape == (batch_size, n_atom, c_out))



#Pairformer_Embedding
#    def __init__(self, min_bin, no_bin, max_bin, inf, c_s, c_z, config):
#    def forward(self, si_input, si, zij, x_pred, n_block,): 

class TestPairEmbed(unittest.TestCase):
    def test_embed_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        c_s = consts.c_s
        c_z = consts.c_z

        min_bin = 3.25
        max_bin = 20.25
        no_bin = 15
        inf = 1e8

        pair_emb = Pairformer_Embedding(min_bin, max_bin, no_bin, inf, c_s, c_z,)

        si_input = torch.ones(batch_size, n_token, c_s)
        si = torch.ones(batch_size, n_token, c_s) 
        zij = torch.ones(batch_size, n_token, n_token, c_z) 
        x_pred = torch.ones(batch_size, n_token, 3)

        out_single, out_pair = pair_emb(si_input, si, zij, x_pred)

        self.assertTrue(out_single.shape == (batch_size, n_token, c_s))
        self.assertTrue(out_pair.shape == (batch_size, n_token, c_z))


class TestAuxiliaryHeads(unittest.TestCase):
    def test_aux_head_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = consts.n_res * 4
        c_s = consts.c_s
        c_z = consts.c_z
        c_out = 50
        min_bin = consts.min_bin
        max_bin = consts.max_binz
        no_bin = consts.no_bin
        inf = consts.inf
        c_out = 50 

        config = {'pae': {'c_z' : c_z, 'c_out' : c_out},
                'pde': {'c_z' : c_z, 'c_out' : c_out},
                'lddt': {'c_s' : c_s, 'c_out' : c_out},
                'distogram': {'c_z' : c_z, 'c_out' : c_out},
                'experimentally_resolved': {'c_s': c_s, 'c_out': c_out},
                'pairformer': {'min_bin': min_bin, 'max_bin': max_bin, 'no_bin': no_bin, 'inf': inf, 'c_s': c_s, 'c_z': c_z,},
                }
        aux_head = AuxiliaryHeads(config)

        si_input = torch.ones(batch_size, n_token, c_s)
        si = torch.ones(batch_size, n_token, c_s)
        zij = torch.ones(batch_size, n_token, n_token, c_z)
        x_pred = torch.ones(batch_size, n_token, 3)
        outputs = {'si': si, 'zij': zij, 'coordinate': x_pred}
        token_to_atom_idx = torch.eye(n_token).repeat_interleave(4, dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

        d = aux_head(si_input, outputs, token_to_atom_idx)

        self.assertTrue(d.distogram_logits.shape == (batch_size, n_token, n_token, c_out))
        self.assertTrue(d.pae_logits.shape == (batch_size, n_token, n_token, c_out))
        self.assertTrue(d.pde_logits.shape == (batch_size, n_token, n_token, c_out))
        self.assertTrue(d.plddt_logits.shape == (batch_size, n_token, c_out))
        
