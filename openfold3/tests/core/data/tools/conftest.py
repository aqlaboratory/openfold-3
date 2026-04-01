"""Pytest configuration for tools tests -- VCR cassette directory."""

from pathlib import Path

import pytest

import openfold3

_CASSETTE_DIR = (
    Path(openfold3.__file__).parent / "tests" / "test_data" / "cassettes" / "test_rscb"
)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "cassette_library_dir": str(_CASSETTE_DIR),
    }
