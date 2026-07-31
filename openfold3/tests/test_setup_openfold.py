from unittest.mock import patch

import pytest

from openfold3.setup_openfold import (
    OpenFoldSetupConfig,
    _require_pytest,
    run_setup,
    setup_biotite_ccd,
)


def test_setup_biotite_ccd(tmp_path):
    ccd_path = tmp_path / "test_ccd.cif"
    has_downloaded = setup_biotite_ccd(ccd_path=ccd_path, force_download=False)
    assert ccd_path.exists()
    assert has_downloaded

    has_downloaded = setup_biotite_ccd(ccd_path=ccd_path, force_download=False)
    assert not has_downloaded


def test_require_pytest_passes_when_installed():
    """This suite runs under pytest, so the probe must find it."""
    _require_pytest()


def test_require_pytest_exits_with_install_instructions(caplog):
    with (
        patch("importlib.util.find_spec", return_value=None),
        pytest.raises(SystemExit) as exit_info,
    ):
        _require_pytest()
    assert exit_info.value.code == 1
    assert "openfold3[dev]" in caplog.text


@pytest.mark.parametrize(
    ("run_integration_tests", "pytest_installed", "expect_exit"),
    [
        pytest.param(True, False, True, id="tests-requested-pytest-missing"),
        pytest.param(True, True, False, id="tests-requested-pytest-present"),
        pytest.param(False, False, False, id="tests-skipped-pytest-irrelevant"),
    ],
)
def test_pytest_is_checked_before_anything_is_downloaded(
    tmp_path, run_integration_tests, pytest_installed, expect_exit
):
    """A missing test dependency must not surface after a multi-gigabyte download.

    Downloads are stubbed out, so reaching them at all is the failure being guarded
    against: when pytest is missing and tests were requested, setup has to exit before
    ``_download_parameters`` is ever called.
    """
    config = OpenFoldSetupConfig(
        openfold_cache=tmp_path / "cache",
        param_directory=tmp_path / "params",
        run_integration_tests=run_integration_tests,
    )
    find_spec_result = object() if pytest_installed else None

    with (
        patch("importlib.util.find_spec", return_value=find_spec_result),
        patch("openfold3.setup_openfold._download_parameters") as download,
        patch("openfold3.setup_openfold.setup_biotite_ccd") as ccd,
        patch("openfold3.setup_openfold._run_integration_tests") as integration_tests,
    ):
        if expect_exit:
            with pytest.raises(SystemExit) as exit_info:
                run_setup(config)
            assert exit_info.value.code == 1
        else:
            run_setup(config)

    assert download.called is not expect_exit
    assert ccd.called is not expect_exit
    assert integration_tests.called == (run_integration_tests and not expect_exit)
