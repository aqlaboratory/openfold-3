import unittest

import torch

from openfold3.core.loss.loss_module import AlphaFold3Loss
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.model_implementations import registry
from tests.config import consts
from tests.data_utils import random_af3_features


class TestAF3Model(unittest.TestCase):
    def setUp(self):
        self.config = registry.MODEL_REGISTRY["af3_all_atom"].base_config

    def run_model(
        self,
        batch_size,
        n_token,
        n_msa,
        n_templ,
        dtype,
        train=True,
        reduce_model_size=True,
        use_deepspeed_evo_attention=False,
        use_block_sparse=False,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # To avoid memory issues in CI
        self.config.model.pairformer.no_blocks = 4
        self.config.model.diffusion_module.diffusion_transformer.no_blocks = 4

        af3 = registry.get_lightning_module(self.config).to(device)

        batch = random_af3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=n_msa,
            n_templ=n_templ,
        )

        n_atom = torch.max(batch["num_atoms_per_token"].sum(dim=-1)).int().item()

        def to_device_dtype(t):
            return t.to(device=torch.device(device), dtype=dtype)

        batch = tensor_tree_map(to_device_dtype, batch)

        if train:
            af3_loss = AlphaFold3Loss(config=self.config.loss)

            outputs = af3(batch=batch)

            loss, loss_breakdown = af3_loss(
                batch=batch, output=outputs, _return_breakdown=True
            )

            # TODO: Checkpointing will cause this to fail, skipping for now
            if self.config.globals.blocks_per_ckpt is None:
                loss.backward()

            x_pred = outputs["x_pred"]
            x_sample = outputs["x_sample"]

            self.assertTrue(
                x_sample.shape
                == (batch_size, self.config.globals.no_samples, n_atom, 3)
            )
            self.assertTrue(loss.shape == ())

        else:
            af3.eval()

            with torch.no_grad():
                outputs = af3(batch=batch)

            x_pred = outputs["x_pred"]

        self.assertTrue(
            x_sample.shape == (batch_size, self.config.globals.no_samples, n_atom, 3)
        )
        self.assertTrue(loss.shape == ())

    def test_shape_small_fp32(self):
        batch_size = consts.batch_size
        n_token = 16
        n_msa = 10
        n_templ = 3
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.config.model.heads.pae.enabled = True
        self.config.loss.confidence.pae.weight = 1.0
        self.config.loss.diffusion.bond_weight = 1.0

        # To avoid memory issues in CI
        self.config.model.pairformer.no_blocks = 4
        self.config.model.diffusion_module.diffusion_transformer.no_blocks = 4

        af3 = registry.get_lightning_module(self.config, model_name="af3_all_atom").to(
            device
        )

        batch = random_af3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=n_msa,
            n_templ=n_templ,
        )

        def to_device(t):
            return t.to(torch.device(device))

        batch = tensor_tree_map(to_device, batch)

        outputs = af3(batch=batch)

        af3_loss = AlphaFold3Loss(config=self.config.loss)
        loss, loss_breakdown = af3_loss(
            batch=batch, output=outputs, _return_breakdown=True
        )

        self.assertTrue(loss.shape == ())


if __name__ == "__main__":
    unittest.main()
