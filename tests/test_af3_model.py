import unittest

import torch

from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.model_implementations.af3_all_atom.config import config
from openfold3.model_implementations.af3_all_atom.model import AlphaFold3
from tests.config import consts
from tests.data_utils import random_af3_features


class TestAF3Model(unittest.TestCase):
    def test_shape(self):
        batch_size = consts.batch_size
        n_token = 16
        n_atom = 4 * n_token
        n_msa = 10
        n_templ = 3
        device = "cuda" if torch.cuda.is_available() else "cpu"

        config.model.pairformer.no_blocks = 4  # To avoid memory issues in CI
        af3 = AlphaFold3(config).to(device)

        batch = random_af3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_atom=n_atom,
            n_msa=n_msa,
            n_templ=n_templ,
        )

        def to_device(t):
            return t.to(torch.device(device))

        batch = tensor_tree_map(to_device, batch)

        outputs = af3(batch=batch)
        x_pred = outputs["x_pred"].cpu()
        x_sample = outputs["x_sample"].cpu()

        self.assertTrue(x_pred.shape == (batch_size, n_atom, 3))
        self.assertTrue(
            x_sample.shape == (batch_size, config.globals.no_samples, n_atom, 3)
        )


if __name__ == "__main__":
    unittest.main()
