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

import importlib.util
import logging
import os
import sys
from pathlib import Path

import biotite.setup_ccd

from openfold3.core.utils.s3 import download_s3_file, s3_file_matches_local
from openfold3.entry_points.parameters import (
    DEFAULT_CHECKPOINT_NAME,
    OPENFOLD_MODEL_CHECKPOINT_REGISTRY,
    download_model_parameters,
)

S3_BUCKET = "openfold3-data"
S3_KEY = "components.bcif"


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def setup_openfold_cache() -> tuple[Path, Path]:
    """Set up the OpenFold cache directory."""
    logger.info("Setting up OpenFold cache directory...")

    default_cache = Path.home() / ".openfold3"
    user_input = input(
        f"Please specify the OpenFold cache directory (default: {default_cache}): "
    ).strip()

    # Use user input if provided, otherwise use default
    if user_input:
        openfold_cache = Path(user_input).expanduser()
    else:
        openfold_cache = default_cache

    openfold_cache.mkdir(parents=True, exist_ok=True)
    ckpt_root_file = openfold_cache / "ckpt_root"

    os.environ["OPENFOLD_CACHE"] = str(openfold_cache)

    return openfold_cache, ckpt_root_file


def setup_param_directory(
    openfold_cache: Path, ckpt_root_file: Path
) -> tuple[Path, bool]:
    """Check and set up the parameter directory."""

    # Check if parameters have already been downloaded
    ckpt_path = ckpt_root_file.read_text().strip() if ckpt_root_file.exists() else None
    checkpoint_file_name = OPENFOLD_MODEL_CHECKPOINT_REGISTRY[
        DEFAULT_CHECKPOINT_NAME
    ].file_name
    if ckpt_path and (Path(ckpt_path) / checkpoint_file_name).exists():
        existing_path = Path(ckpt_root_file.read_text().strip())
        logger.info(
            f"OpenFold3 parameters may already be installed at: {existing_path}"
        )
        logger.info("Do you want to:")
        logger.info("1) Use existing parameters (skip download)")
        logger.info("2) Download to a new location")
        logger.info("3) Re-download to existing location")

        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            logger.info(f"Using existing parameters at: {existing_path}")
            return existing_path, False  # Don't download
        elif choice == "2":
            user_input = input(
                "Please specify a new directory to save the parameters: "
            ).strip()
            if not user_input:
                logger.error("No directory specified. Exiting.")
                sys.exit(1)
            param_dir = Path(user_input).expanduser()
        elif choice == "3":
            logger.info(f"Re-downloading to: {existing_path}")
            param_dir = existing_path
        else:
            logger.error("Invalid choice. Exiting.")
            sys.exit(1)
    else:
        # First time setup
        logger.info("Downloading OpenFold3 parameters...")
        user_input = input(
            "Please specify the directory for parameter download "
            f"(default: {openfold_cache}): "
        ).strip()

        # Use user input if provided, otherwise use default (the cache directory)
        if user_input:
            param_dir = Path(user_input).expanduser()
        else:
            param_dir = openfold_cache

    # Create the directory if it doesn't exist
    param_dir.mkdir(parents=True, exist_ok=True)

    # Save the path to ckpt_root file
    ckpt_root_file.write_text(str(param_dir))

    logger.info(f"Parameters directory set to: {param_dir}")
    logger.info(f"Path saved to: {ckpt_root_file}")

    return param_dir, True  # Proceed with download


def download_parameters(param_dir) -> None:
    """Perform the parameter download."""
    all_checkpoints = list(OPENFOLD_MODEL_CHECKPOINT_REGISTRY.keys())

    logger.info("Select parameters to download:")
    logger.info(f"1) Download only the default checkpoint ({DEFAULT_CHECKPOINT_NAME})")
    logger.info(f"2) Download all parameters ({', '.join(all_checkpoints)}) (default)")
    logger.info("3) Download a specific parameter by name")

    choice = input("Enter your choice (1/2/3, default: 1): ").strip() or "1"

    logger.info("Starting parameter download...")

    if choice == "1":
        download_model_parameters(
            Path(param_dir),
            DEFAULT_CHECKPOINT_NAME,
            force_download=True,
            skip_confirmation=True,
        )
    elif choice == "2":
        for name in all_checkpoints:
            download_model_parameters(
                Path(param_dir), name, force_download=True, skip_confirmation=True
            )
    elif choice == "3":
        print("\nAvailable parameters:")
        for name in all_checkpoints:
            print(f"\n  - {name}")
        param_name = input("Enter parameter name: ").strip()
        if param_name not in OPENFOLD_MODEL_CHECKPOINT_REGISTRY:
            logger.error(
                f"Unknown parameter name '{param_name}'. "
                f"Available parameter names are: {', '.join(all_checkpoints)}"
            )
            sys.exit(1)
        download_model_parameters(
            Path(param_dir), param_name, force_download=True, skip_confirmation=True
        )
    else:
        logger.error("Invalid choice. Exiting.")
        sys.exit(1)

    logger.info("Download completed successfully.")


def setup_biotite_ccd(*, ccd_path: Path, force_download: bool) -> bool:
    def ccd_is_stale(*, ccd_path: Path) -> bool:
        if not ccd_path.exists():
            return True
        return not s3_file_matches_local(ccd_path, S3_BUCKET, S3_KEY)

    logger.info("Starting Biotite CCD setup...")
    if force_download or ccd_is_stale(ccd_path=ccd_path):
        download_s3_file(S3_BUCKET, S3_KEY, ccd_path)
        return True
    else:
        logger.info(
            f"Biotite CCD file at {ccd_path} is up-to-date with "
            f"s3://{S3_BUCKET}/{S3_KEY}, skipping."
        )
        return False


def run_integration_tests() -> None:
    """Run integration tests."""
    confirm = input("Run integration tests? (yes/no)")
    if confirm.lower() not in ["yes", "y"]:
        logger.info("Skipping integration tests, exiting setup.")
        return

    logger.info("Running integration tests...")
    pytest_is_installed = importlib.util.find_spec("pytest")
    if not pytest_is_installed:
        logger.error("Pytest is required to run integration tests.")
        logger.error(
            "Please install pytest e.g. `pip install pytest` and rerun the script."
        )
        return

    # Set environment variables for tests
    os.environ["OPENFOLD_SETUP_SCRIPT"] = "1"
    import pytest

    exit_code = pytest.main(
        [
            "-v",
            "--rootdir",
            str(Path(__file__).parent),
            "--log-cli-level=WARNING",
            Path(__file__).parent / "tests/test_inference_full.py",
            "-m",
            "inference_verification",
            "--skip-ccd-update",
        ]
    )

    if exit_code != 0:
        logger.error("Integration tests failed. Please check the output above.")
        sys.exit(1)

    logger.info("Integration tests passed!")


def main():
    """Main execution."""
    # Step 1: Set up OpenFold cache directory
    openfold_cache, ckpt_root_file = setup_openfold_cache()

    # Step 2: Set up checkpoint directory
    param_dir, should_download = setup_param_directory(openfold_cache, ckpt_root_file)

    # Step 3: Perform download if needed
    if should_download:
        download_parameters(param_dir)

    # Step 4: Setup CCD with biotite
    setup_biotite_ccd(ccd_path=biotite.setup_ccd.OUTPUT_CCD, force_download=False)

    # Step 5: Run tests (always run regardless of download status)
    run_integration_tests()


if __name__ == "__main__":
    main()
