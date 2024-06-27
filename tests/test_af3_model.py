import torch
import unittest

from openfold3.model_implementations.af3_all_atom.config import config
from openfold3.model_implementations.af3_all_atom.model import AlphaFold3
from tests.config import consts


class TestAF3Model(unittest.TestCase):

    def test_shape(self):

        af3 = AlphaFold3(config)
        # batch_size = consts.batch_size
        # n_atom = 4 * consts.n_res
        # c_atom_ref = 390
        # c_atom = 64
        # c_atom_pair = 16

        # embedder = RefAtomFeatureEmbedder(c_atom_ref=c_atom_ref,
        #                                   c_atom=c_atom,
        #                                   c_atom_pair=c_atom_pair)
        
        # batch = {
        #     'ref_pos': torch.randn((batch_size, n_atom, 3)),
        #     'ref_mask': torch.ones((batch_size, n_atom)),
        #     'ref_element': torch.ones((batch_size, n_atom, 128)),
        #     'ref_charge': torch.ones((batch_size, n_atom)),
        #     'ref_atom_name_chars': torch.ones((batch_size, n_atom, 4, 64)),
        #     'ref_space_uid': torch.zeros((batch_size, n_atom)),
        # }

        # cl, plm = embedder(batch)

        # self.assertTrue(cl.shape == (batch_size, n_atom, c_atom))
        # self.assertTrue(plm.shape == (batch_size, n_atom, n_atom, c_atom_pair))


if __name__ == "__main__":
    unittest.main()
