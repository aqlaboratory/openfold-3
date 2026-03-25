# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest  # noqa: F401  - used for pytest tmp fixture
import torch

from openfold3.core.config import config_utils
from openfold3.core.utils.checkpoint_loading_utils import load_checkpoint
from openfold3.entry_points.experiment_runner import (
    InferenceExperimentRunner,
    TrainingExperimentRunner,
)
from openfold3.entry_points.parameters import (
    DEFAULT_CHECKPOINT_NAME,
    OPENFOLD_MODEL_CHECKPOINT_REGISTRY,
    get_default_checkpoint_dir,
)
from openfold3.entry_points.validator import (
    InferenceExperimentConfig,
    TrainingExperimentConfig,
)


@pytest.fixture
def default_ckpt_path():
    param_dir = get_default_checkpoint_dir()
    default_ckpt_path = Path(
        param_dir
        / OPENFOLD_MODEL_CHECKPOINT_REGISTRY[DEFAULT_CHECKPOINT_NAME].file_name
    )
    if not default_ckpt_path.exists():
        pytest.skip("Default checkpoint not found; skipping test.")
    return default_ckpt_path


class TestOF3ModelCheckpointing:
    def test_make_model_ckpt(
        self,
        tmp_path,
    ):
        test_dummy_file = tmp_path / "test.json"
        test_dummy_file.write_text("test")
        test_yaml_str = textwrap.dedent(f"""\
            model_update:
                presets:
                    - train
                custom:
                    settings:
                        memory:
                            train:
                                use_deepspeed_evo_attention: false
                    architecture:
                        pairformer:
                            no_blocks: 4
                        diffusion_module:
                            diffusion_transformer:
                                no_blocks: 4
                        loss_module:
                            diffusion:
                                chunk_size: 16
            
            dataset_configs:
                train:
                    weighted-pdb:
                        dataset_class: WeightedPDBDataset 
                        weight: 1 
                                        
            dataset_paths:
                weighted-pdb:
                    alignments_directory: null
                    alignment_db_directory: null
                    alignment_array_directory: {tmp_path} 
                    target_structures_directory: {tmp_path} 
                    target_structure_file_format: npz
                    dataset_cache_file: {test_dummy_file} 
                    reference_molecule_directory: {tmp_path} 
                    template_cache_directory: {tmp_path} 
                    template_structure_array_directory: {tmp_path} 
                    template_structures_directory: null
                    template_file_format: pkl
                    ccd_file: null
        """)
        test_yaml_file = tmp_path / "runner.yml"
        test_yaml_file.write_text(test_yaml_str)

        expt_config = TrainingExperimentConfig.model_validate(
            config_utils.load_yaml(test_yaml_file)
        )
        expt_runner = TrainingExperimentRunner(expt_config)
        expt_runner.setup()

        assert (
            "version_tensor" in expt_runner.lightning_module.ema.state_dict()["params"]
        )

        # abbreviated pytorch lightning checkpoint
        fake_pytorch_lightning_checkpoint = {
            "epoch": 0,
            "global_step": 0,
            "state_dict": expt_runner.lightning_module.state_dict(),
            "ema": expt_runner.lightning_module.ema.state_dict(),
        }
        param_path = tmp_path / "model_weights.ckpt"
        torch.save(fake_pytorch_lightning_checkpoint, param_path)

        reloaded_model = load_checkpoint(param_path)

        ema_params = reloaded_model["ema"]["params"]
        expected_version_number = "1.0.0"
        actual_version = ema_params["version_tensor"].long().tolist()
        actual_version_number = (
            f"{actual_version[0]}.{actual_version[1]}.{actual_version[2]}"
        )

        assert actual_version_number == expected_version_number

    def test_load_model_ckpt_with_no_version_warns(self, tmp_path, default_ckpt_path):
        """Test that warning is raised if version_tensor is the only key that is missing."""

        # Load checkpoint and remove version_tensor from EMA params
        ckpt = load_checkpoint(default_ckpt_path)
        ckpt.pop("version_tensor", None)
        ckpt_with_no_version_path = tmp_path / "model_weights_no_version.ckpt"
        torch.save(ckpt, ckpt_with_no_version_path)

        # Load via InferenceExperimentRunner and assert the warning is issued
        inference_config = InferenceExperimentConfig.model_validate(
            {"inference_ckpt_path": ckpt_with_no_version_path}
        )
        inference_runner = InferenceExperimentRunner(inference_config)
        with patch("openfold3.entry_points.experiment_runner.logger") as mock_logger:
            inference_runner.setup()
        warning_messages = [call.args[0] for call in mock_logger.warning.call_args_list]
        assert any("version_tensor" in msg for msg in warning_messages)

    def test_load_model_ckpt_with_missing_fields_fails(
        self, tmp_path, default_ckpt_path
    ):
        # Create a model checkpoint missing input_embedder param field
        # Check that model loading fails
        ckpt = load_checkpoint(default_ckpt_path)
        ckpt.pop("version_tensor", None)
        ckpt.pop(
            "input_embedder.atom_attn_enc.ref_atom_feature_embedder.linear_ref_pos.weight",
            None,
        )
        bad_ckpt_with_missing_fields = tmp_path / "bad.ckpt"
        torch.save(ckpt, bad_ckpt_with_missing_fields)

        # Load via InferenceExperimentRunner and assert the warning is issued
        inference_config = InferenceExperimentConfig.model_validate(
            {"inference_ckpt_path": bad_ckpt_with_missing_fields}
        )
        inference_runner = InferenceExperimentRunner(inference_config)
        with pytest.raises(RuntimeError):
            inference_runner.setup()
