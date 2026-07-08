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

#!/usr/bin/env python3
"""
Setup script for OpenFold3 parameters.
Downloads model parameters and runs verification tests.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Literal

import biotite.setup_ccd
import click
import pydantic
from pydantic import Field

from openfold3.core.utils.s3 import download_s3_file, s3_file_matches_local
from openfold3.entry_points.parameters import (
    CHECKPOINT_ROOT_FILENAME,
    DEFAULT_CHECKPOINT_NAME,
    LEGACY_CHECKPOINTS,
    OPENFOLD_MODEL_CHECKPOINT_REGISTRY,
    download_model_parameters,
)

S3_BUCKET = "openfold3-data"
S3_KEY = "components.bcif"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class OpenFoldSetupConfig(pydantic.BaseModel):
    """Configuration for the OpenFold3 setup script.

    Can be serialised to/from JSON and passed to ``setup_openfold --config``.
    """

    openfold_cache: Path = Field(
        default_factory=lambda: Path.home() / ".openfold3",
        description="Root cache directory for OpenFold3 artifacts.",
    )
    param_directory: Path = Field(
        default_factory=lambda: Path.home() / ".openfold3",
        description=(
            "Directory where model parameters are downloaded. "
            "Defaults to openfold_cache."
        ),
    )
    # Accepts "default", "all", or any valid checkpoint name from
    # OPENFOLD_MODEL_CHECKPOINT_REGISTRY.
    selected_parameters: Literal["default", "all"] | str = Field(
        default="default",
        description=(
            'Which model parameters to download. "default" downloads only the '
            'default checkpoint, "all" downloads every non-legacy checkpoint, '
            "or supply a specific checkpoint name."
        ),
    )
    force_download_parameters: bool = Field(
        default=False,
        description=(
            "If True, re-download model parameters even when they already exist "
            "at the target path. Defaults to False (skip download if present)."
        ),
    )
    run_integration_tests: bool = Field(
        default=False,
        description="Whether to run integration tests after downloading parameters.",
    )

    @pydantic.field_validator("selected_parameters")
    @classmethod
    def _validate_selected_parameters(cls, v: str) -> str:
        valid = {"default", "all"} | set(OPENFOLD_MODEL_CHECKPOINT_REGISTRY)
        if v not in valid:
            valid_names = ", ".join(OPENFOLD_MODEL_CHECKPOINT_REGISTRY)
            raise ValueError(
                "selected_parameters must be 'default', 'all', or a valid checkpoint "
                f"name. Valid checkpoint names: {valid_names}"
            )
        return v


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prompt_for_config() -> OpenFoldSetupConfig:
    """Interactively prompt the user and return a populated config."""
    logger.info("Setting up OpenFold3...")

    default_cache = Path.home() / ".openfold3"
    user_input = input(
        f"Please specify the OpenFold cache directory (default: {default_cache}): "
    ).strip()
    openfold_cache = Path(user_input).expanduser() if user_input else default_cache

    user_input = input(
        "Please specify the directory for parameter download "
        f"(default: {openfold_cache}): "
    ).strip()
    param_directory = Path(user_input).expanduser() if user_input else openfold_cache

    all_checkpoints = [
        name
        for name in OPENFOLD_MODEL_CHECKPOINT_REGISTRY
        if name not in LEGACY_CHECKPOINTS
    ]
    logger.info("Select parameters to download:")
    logger.info(f"1) Download only the default checkpoint ({DEFAULT_CHECKPOINT_NAME})")
    logger.info(f"2) Download all parameters ({', '.join(all_checkpoints)})")
    logger.info("3) Download a specific parameter by name")
    choice = input("Enter your choice (1/2/3, default: 1): ").strip() or "1"

    if choice == "1":
        selected_parameters = "default"
    elif choice == "2":
        selected_parameters = "all"
    elif choice == "3":
        print("\nAvailable parameters:")
        for name in all_checkpoints:
            print(f"  - {name}")
        selected_parameters = input("Enter parameter name: ").strip()
    else:
        logger.error("Invalid choice. Exiting.")
        sys.exit(1)

    force_input = input(
        "Force re-download parameters even if they already exist? "
        "(yes/no, default: no) "
    ).strip()
    force_download_parameters = force_input.lower() in ["yes", "y"]

    confirm = input("Run integration tests? (yes/no) ").strip()
    run_integration_tests = confirm.lower() in ["yes", "y"]

    return OpenFoldSetupConfig(
        openfold_cache=openfold_cache,
        param_directory=param_directory,
        selected_parameters=selected_parameters,
        force_download_parameters=force_download_parameters,
        run_integration_tests=run_integration_tests,
    )


def _download_parameters(
    param_dir: Path, selected_parameters: str, *, force_download: bool
) -> None:
    all_checkpoints = [
        name
        for name in OPENFOLD_MODEL_CHECKPOINT_REGISTRY
        if name not in LEGACY_CHECKPOINTS
    ]
    logger.info("Starting parameter download...")
    if selected_parameters == "default":
        download_model_parameters(
            param_dir,
            DEFAULT_CHECKPOINT_NAME,
            force_download=force_download,
            skip_confirmation=True,
        )
    elif selected_parameters == "all":
        for name in all_checkpoints:
            download_model_parameters(
                param_dir, name, force_download=force_download, skip_confirmation=True
            )
    else:
        download_model_parameters(
            param_dir,
            selected_parameters,
            force_download=force_download,
            skip_confirmation=True,
        )
    logger.info("Download completed successfully.")


def setup_biotite_ccd(*, ccd_path: Path, force_download: bool) -> bool:
    def _ccd_is_stale(*, ccd_path: Path) -> bool:
        if not ccd_path.exists():
            return True
        return not s3_file_matches_local(ccd_path, S3_BUCKET, S3_KEY)

    logger.info("Starting Biotite CCD setup...")
    if force_download or _ccd_is_stale(ccd_path=ccd_path):
        download_s3_file(S3_BUCKET, S3_KEY, ccd_path)
        return True
    logger.info(
        f"Biotite CCD file at {ccd_path} is up-to-date with "
        f"s3://{S3_BUCKET}/{S3_KEY}, skipping."
    )
    return False


def _run_integration_tests() -> None:
    logger.info("Running integration tests...")
    os.environ["OPENFOLD_SETUP_SCRIPT"] = "1"
    import unittest

    root_logger = logging.getLogger()
    original_level = root_logger.level
    try:
        root_logger.setLevel(logging.WARNING)
        program = unittest.main(
            module="openfold3.tests.test_inference_full",
            exit=False,
            verbosity=2,
        )
    finally:
        root_logger.setLevel(original_level)

    if not program.result.wasSuccessful():
        logger.error("Integration tests failed. Please check the output above.")
        sys.exit(1)
    logger.info("Integration tests passed!")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_setup(config: OpenFoldSetupConfig) -> None:
    """Execute the setup described by *config* without any interactive prompts."""
    config.openfold_cache.mkdir(parents=True, exist_ok=True)
    os.environ["OPENFOLD_CACHE"] = str(config.openfold_cache)

    config.param_directory.mkdir(parents=True, exist_ok=True)
    ckpt_root_file = config.openfold_cache / CHECKPOINT_ROOT_FILENAME
    ckpt_root_file.write_text(str(config.param_directory))
    logger.info(f"Parameters directory set to: {config.param_directory}")

    _download_parameters(
        config.param_directory,
        config.selected_parameters,
        force_download=config.force_download_parameters,
    )
    setup_biotite_ccd(ccd_path=biotite.setup_ccd.OUTPUT_CCD, force_download=False)

    if config.run_integration_tests:
        _run_integration_tests()
    else:
        logger.info("Skipping integration tests.")


@click.command()
@click.option(
    "--non-interactive",
    is_flag=True,
    default=False,
    help="Non-interactively run setup using all default config values.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a JSON file containing an OpenFoldSetupConfig.",
)
def main(non_interactive: bool = False, config_path: Path | None = None):
    """Set up OpenFold3 model parameters."""
    if config_path is not None:
        config = OpenFoldSetupConfig.model_validate_json(config_path.read_text())
    elif non_interactive:
        config = OpenFoldSetupConfig()
    else:
        config = _prompt_for_config()

    run_setup(config)

    setup_config_path = config.openfold_cache / "setup_config.json"
    setup_config_path.write_text(config.model_dump_json(indent=4))
    logger.info(f"Setup configuration saved to {setup_config_path}")


if __name__ == "__main__":
    main()
