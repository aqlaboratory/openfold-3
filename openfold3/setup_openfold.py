# Copyright 2025 AlQuraishi Laboratory
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
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def setup_conda_commands():
    """Check if running in a conda environment."""
    logger.info("Setting up conda shell environment...")

    if not os.environ.get("CONDA_PREFIX"):
        logger.error("Error: This script must be run from within a conda environment.")
        logger.error("Please activate your conda environment first:")
        logger.error("  conda activate your_env_name")
        sys.exit(1)

    logger.info(
        f"Running in conda environment: {os.path.basename(os.environ['CONDA_PREFIX'])}"
    )


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

    # Set conda environment variable
    try:
        subprocess.run(
            [
                "conda",
                "env",
                "config",
                "vars",
                "set",
                f"OPENFOLD_CACHE={openfold_cache}",
            ],
            check=True,
            capture_output=True,
        )
        logger.info(f"OPENFOLD_CACHE set to: {openfold_cache}")
        logger.info(
            "Variable will persist when you reactivate: "
            f"conda activate {os.path.basename(os.environ['CONDA_PREFIX'])}"
        )
    except subprocess.CalledProcessError:
        logger.warning(
            "Warning: Could not set OPENFOLD_CACHE in conda environment config"
        )
        logger.warning("Variable is set for current session only")

    return openfold_cache, ckpt_root_file


def setup_param_directory(
    openfold_cache: Path, ckpt_root_file: Path
) -> tuple[Path, bool]:
    """Check and set up the parameter directory."""

    # Check if parameters have already been downloaded
    if ckpt_root_file.exists():
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
    """Perform the download using the download script."""
    logger.info("Starting parameter download...")

    # Get the directory where this script is located
    script_dir = Path(__file__).parent / "scripts"
    download_script = script_dir / "download_openfold3_params.sh"

    result = subprocess.run(
        ["bash", str(download_script), f"--download_dir={param_dir}"], check=False
    )

    if result.returncode != 0:
        logger.error("Download failed. Exiting.")
        sys.exit(1)

    logger.info("Download completed successfully.")


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
            "--rootdir", str(Path(__file__).parent),
            "--log-cli-level=WARNING",
            Path(__file__).parent / "tests/test_inference_full.py",
            "-m",
            "inference_verification",
        ]
    )

    if exit_code != 0:
        logger.error("Integration tests failed. Please check the output above.")
        sys.exit(1)

    logger.info("Integration tests passed!")


def main():
    """Main execution."""
    # Step 1: Set up conda environment
    setup_conda_commands()

    # Step 2: Set up OpenFold cache directory
    openfold_cache, ckpt_root_file = setup_openfold_cache()

    # Step 3: Set up checkpoint directory
    param_dir, should_download = setup_param_directory(openfold_cache, ckpt_root_file)

    # Step 4: Perform download if needed
    if should_download:
        download_parameters(param_dir)

    # Step 5: Run tests (always run regardless of download status)
    run_integration_tests()


if __name__ == "__main__":
    main()
