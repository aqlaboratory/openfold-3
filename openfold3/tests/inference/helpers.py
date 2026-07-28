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

"""Shared helpers for the end-to-end inference tests in this package.

Everything here drives a real ``InferenceExperimentRunner``, so it requires an
accelerator and downloaded model weights; the test modules gate on that with
``skip_unless_accelerator_available``.
"""

import logging
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from openfold3.core.config import config_utils
from openfold3.entry_points.experiment_runner import InferenceExperimentRunner
from openfold3.entry_points.validator import (
    InferenceExperimentConfig,
)

logger = logging.getLogger(__name__)

inference_test_yaml_str = textwrap.dedent("""\
    model_update:
      presets:
        - predict
        - low_mem
    """)


def run_inference(
    query_set,
    output_dir: Path,
    *,
    use_msa_server: bool,
    use_templates: bool,
    num_diffusion_samples: int = 1,
    template_output_dir: Path | None = None,
) -> Path:
    """Run one inference job into ``output_dir`` and return it.

    Skips (``pytest.skip``) if no model checkpoint is available (escalated to a hard
    failure when ``OPENFOLD_SETUP_SCRIPT=1``). ``template_output_dir`` isolates the
    template cache per run (otherwise it lands in a persistent ``/tmp`` dir shared across
    runs and same-sequence queries).
    """
    runner_yaml = output_dir / "runner_config.yaml"
    yaml_str = inference_test_yaml_str
    if template_output_dir is not None:
        yaml_str += textwrap.dedent(f"""\
            template_preprocessor_settings:
              output_directory: {template_output_dir}
            """)
    runner_yaml.write_text(yaml_str)

    with patch("builtins.input", return_value="no"):
        experiment_config = InferenceExperimentConfig(
            **config_utils.load_yaml(runner_yaml)
        )
    runner = InferenceExperimentRunner(
        experiment_config,
        num_diffusion_samples=num_diffusion_samples,
        output_dir=output_dir,
        use_msa_server=use_msa_server,
        use_templates=use_templates,
    )
    try:
        runner.setup()
    except ValueError as e:
        if "is not a valid file or directory" in str(e):
            if os.environ.get("OPENFOLD_SETUP_SCRIPT") == "1":
                raise AssertionError(
                    "No checkpoint files found after running setup script. "
                    "Please check that the download completed successfully."
                ) from None
            logger.warning(
                "No checkpoint files found, skipping. Use the setup script to "
                "download the weights."
            )
            pytest.skip("No checkpoint files available")
        raise

    runner.run(query_set)
    runner.cleanup()

    err_log_dir = output_dir / "logs"
    if err_log_dir.exists():
        raise RuntimeError(
            f"Found error logs in directory {err_log_dir}, "
            "check for errors in inference."
        )
    return output_dir
