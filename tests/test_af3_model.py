import unittest

import torch

from openfold3.core.loss.loss_module import AlphaFold3Loss
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.model_implementations.af3_all_atom.config import (
    config,
    train_config_update,
)
from openfold3.model_implementations.af3_all_atom.model import AlphaFold3
from tests import compare_utils
from tests.config import consts
from tests.data_utils import random_af3_features


class TestAF3Model(unittest.TestCase):
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

        if train:
            config.update(train_config_update)
            config.globals.chunk_size = None

            # Needed to run large model
            if not reduce_model_size:
                config.globals.blocks_per_ckpt = 1

        if reduce_model_size:
            # To avoid memory issues in CI
            config.model.pairformer.no_blocks = 4
            config.model.diffusion_module.diffusion_transformer.no_blocks = 4

        config.globals.use_deepspeed_evo_attention = use_deepspeed_evo_attention
        config.model.input_embedder.atom_attn_enc.use_block_sparse_attn = (
            use_block_sparse
        )
        config.model.diffusion_module.atom_attn_enc.use_block_sparse_attn = (
            use_block_sparse
        )
        config.model.diffusion_module.atom_attn_dec.use_block_sparse_attn = (
            use_block_sparse
        )

        config.model.heads.distogram.enabled = True
        config.loss.diffusion.alpha_bond = 1.0

        af3 = AlphaFold3(config).to(device=torch.device(device), dtype=dtype)

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
            af3_loss = AlphaFold3Loss(config=config.loss)

            outputs = af3(batch=batch)

            loss, loss_breakdown = af3_loss(
                batch=batch, output=outputs, _return_breakdown=True
            )

            # TODO: Checkpointing will cause this to fail, skipping for now
            if config.globals.blocks_per_ckpt is None:
                loss.backward()

            x_pred = outputs["x_pred"]
            x_sample = outputs["x_sample"]

            self.assertTrue(
                x_sample.shape == (batch_size, config.globals.no_samples, n_atom, 3)
            )
            self.assertTrue(loss.shape == ())

        else:
            af3.eval()

            with torch.no_grad():
                outputs = af3(batch=batch)

            x_pred = outputs["x_pred"]

        self.assertTrue(x_pred.shape == (batch_size, n_atom, 3))

    def test_shape_small_fp32(self):
        batch_size = consts.batch_size
        n_token = 16
        n_msa = 10
        n_templ = 3

        # Train
        self.run_model(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=n_msa,
            n_templ=n_templ,
            dtype=torch.float32,
            train=True,
            reduce_model_size=True,
            use_deepspeed_evo_attention=False,
            use_block_sparse=False,
        )

        # Eval
        self.run_model(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=n_msa,
            n_templ=n_templ,
            dtype=torch.float32,
            train=False,
            reduce_model_size=True,
            use_deepspeed_evo_attention=False,
            use_block_sparse=False,
        )

    @compare_utils.skip_unless_triton_installed()
    @compare_utils.skip_unless_cuda_available()
    def test_shape_small_kernels(self):
        batch_size = consts.batch_size
        n_token = 16
        n_msa = 10
        n_templ = 3

        for dtype in [torch.float32, torch.bfloat16]:
            # Train
            self.run_model(
                batch_size=batch_size,
                n_token=n_token,
                n_msa=n_msa,
                n_templ=n_templ,
                dtype=dtype,
                train=True,
                reduce_model_size=True,
                use_deepspeed_evo_attention=True,
                use_block_sparse=True,
            )

            # Eval
            self.run_model(
                batch_size=batch_size,
                n_token=n_token,
                n_msa=n_msa,
                n_templ=n_templ,
                dtype=dtype,
                train=False,
                reduce_model_size=True,
                use_deepspeed_evo_attention=True,
                use_block_sparse=True,
            )

    @compare_utils.skip_unless_triton_installed()
    @compare_utils.skip_unless_cuda_available()
    def test_shape_large_eval(self):
        batch_size = 1
        n_token = 384
        n_msa = 16384
        n_templ = 4

        for dtype in [torch.float32, torch.bfloat16]:
            self.run_model(
                batch_size=batch_size,
                n_token=n_token,
                n_msa=n_msa,
                n_templ=n_templ,
                dtype=dtype,
                train=False,
                reduce_model_size=False,
                use_deepspeed_evo_attention=True,
                use_block_sparse=True,
            )

    @compare_utils.skip_unless_triton_installed()
    @compare_utils.skip_unless_cuda_available()
    def test_shape_large_bf16_train(self):
        batch_size = 1
        n_token = 384
        n_msa = 16384
        n_templ = 4

        self.run_model(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=n_msa,
            n_templ=n_templ,
            dtype=torch.bfloat16,
            train=True,
            reduce_model_size=False,
            use_deepspeed_evo_attention=True,
            use_block_sparse=True,
        )


if __name__ == "__main__":
    unittest.main()
