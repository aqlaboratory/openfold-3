import torch
import unittest

from openfold3.model_implementations.af3_all_atom.config import config
from openfold3.model_implementations.af3_all_atom.model import AlphaFold3
from tests.config import consts
from tests.data_utils import random_af3_features


class TestAF3Model(unittest.TestCase):

    def test_shape(self):
        batch_size = consts.batch_size
        n_token = consts.n_res
        n_atom = 4 * consts.n_res
        n_msa = 10
        n_templ = 3

        af3 = AlphaFold3(config)
        
        batch = random_af3_features(batch_size=batch_size,
                                    n_token=n_token,
                                    n_atom=n_atom,
                                    n_msa=n_msa,
                                    n_templ=n_templ)

        xl_gt = torch.ones((batch_size, n_atom, 3))

        af3(batch=batch, xl_gt=xl_gt)


if __name__ == "__main__":
    unittest.main()
