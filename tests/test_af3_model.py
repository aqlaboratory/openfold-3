import pytest
import torch

from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.core.utils.precision_utils import OF3DeepSpeedPrecision
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry
from openfold3.projects.of3_all_atom.runner import OpenFold3AllAtom
from tests import compare_utils
from tests.config import consts
from tests.data_utils import random_af3_features


class TestAF3Model:
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
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        project_entry = OF3ProjectEntry()
        config = project_entry.get_model_config_with_presets()

        if train:
            config.settings.blocks_per_ckpt = 1
            config.settings.ckpt_intermediate_steps = True

        if reduce_model_size:
            # To avoid memory issues in CI
            config.architecture.pairformer.no_blocks = 4
            config.architecture.diffusion_module.diffusion_transformer.no_blocks = 4

        config.settings.memory.train.use_deepspeed_evo_attention = (
            use_deepspeed_evo_attention
        )
        config.settings.memory.eval.use_deepspeed_evo_attention = (
            use_deepspeed_evo_attention
        )
        config.architecture.loss_module.diffusion.chunk_size = 16

        af3 = OpenFold3AllAtom(config, _compile=False).to(device=device, dtype=dtype)
        af3_loss = OpenFold3Loss(config=config.architecture.loss_module)

        batch = random_af3_features(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=n_msa,
            n_templ=n_templ,
            is_eval=(not train),
        )

        precision = "32-true" if dtype == torch.float32 else "bf16-mixed"
        batch = OF3DeepSpeedPrecision(precision=precision).convert_input(batch)

        n_atom = torch.max(batch["num_atoms_per_token"].sum(dim=-1)).int().item()
        num_rollout_samples = (
            config.architecture.shared.diffusion.no_mini_rollout_samples
            if train
            else config.architecture.shared.diffusion.no_full_rollout_samples
        )

        def to_device(t):
            return t.to(device=torch.device(device))

        batch = tensor_tree_map(to_device, batch)

        if train:
            batch, outputs = af3(batch=batch)

            loss, loss_breakdown = af3_loss(
                batch=batch, output=outputs, _return_breakdown=True
            )

            loss.backward()

            atom_positions_predicted = outputs["atom_positions_predicted"]
            atom_positions_diffusion = outputs["atom_positions_diffusion"]

            num_diffusion_samples = config.architecture.shared.diffusion.no_samples
            expected_diffusion_shape = (batch_size, num_diffusion_samples, n_atom, 3)
            assert atom_positions_diffusion.shape == expected_diffusion_shape

            assert loss.shape == ()

        else:
            af3.eval()

            # filters used by validation metrics
            assert "intra_filter_atomized" in batch["ground_truth"]
            assert "inter_filter_atomized" in batch["ground_truth"]

            with torch.no_grad():
                batch, outputs = af3(batch=batch)

                loss, loss_breakdown = af3_loss(
                    batch=batch, output=outputs, _return_breakdown=True
                )

                assert loss.shape == ()

            atom_positions_predicted = outputs["atom_positions_predicted"]

        assert atom_positions_predicted.shape == (
            batch_size,
            num_rollout_samples,
            n_atom,
            3,
        )

    @pytest.mark.parametrize(
        "model_phase", ["train", "eval"], ids=lambda p: f"model={p}"
    )
    def test_shape_small_fp32(self, model_phase):
        batch_size = consts.batch_size
        n_token = 18
        n_msa = 10
        n_templ = 3

        is_train = model_phase == "train"
        self.run_model(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=n_msa,
            n_templ=n_templ,
            dtype=torch.float32,
            train=is_train,
            reduce_model_size=True,
            use_deepspeed_evo_attention=False,
        )

    @compare_utils.skip_unless_triton_installed()
    @compare_utils.skip_unless_cuda_available()
    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.bfloat16], ids=lambda d: f"dtype={d}"
    )
    @pytest.mark.parametrize(
        "model_phase", ["train", "eval"], ids=lambda p: f"model={p}"
    )
    def test_shape_small_kernels(self, dtype, model_phase):
        batch_size = consts.batch_size
        n_token = 18
        n_msa = 10
        n_templ = 3

        is_train = model_phase == "train"
        self.run_model(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=n_msa,
            n_templ=n_templ,
            dtype=dtype,
            train=is_train,
            reduce_model_size=True,
            use_deepspeed_evo_attention=True,
        )

    @pytest.mark.skip(
        reason="Manually enable this for now, will add flag to run slow tests later."
    )
    @compare_utils.skip_unless_triton_installed()
    @compare_utils.skip_unless_cuda_available()
    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.bfloat16], ids=lambda d: f"dtype={d}"
    )
    def test_shape_large_eval(self, dtype):
        batch_size = 1
        n_token = 384
        n_msa = 16384
        n_templ = 4

        self.run_model(
            batch_size=batch_size,
            n_token=n_token,
            n_msa=n_msa,
            n_templ=n_templ,
            dtype=dtype,
            train=False,
            reduce_model_size=False,
            use_deepspeed_evo_attention=True,
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
        )
