from unittest.mock import patch

import pytest

from openfold3.setup_openfold import (
    OpenFoldSetupConfig,
    _prompt_for_config,
    _pytest_is_installed,
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


def test_pytest_probe_finds_pytest_when_installed():
    """This suite runs under pytest, so the probe must find it."""
    assert _pytest_is_installed()


def test_pytest_probe_reports_missing():
    with patch("importlib.util.find_spec", return_value=None):
        assert not _pytest_is_installed()


def _prompts_seen(
    *, pytest_installed: bool, answer: str = ""
) -> tuple[list[str], bool]:
    """Drive ``_prompt_for_config`` on defaults, returning the prompts and the flag.

    ``answer`` is given only to the integration-test question; every other prompt takes
    its default via an empty string.
    """
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return answer if "integration test" in prompt.lower() else ""

    with (
        patch(
            "openfold3.setup_openfold._pytest_is_installed",
            return_value=pytest_installed,
        ),
        patch("builtins.input", side_effect=fake_input),
    ):
        config = _prompt_for_config()
    return prompts, config.run_integration_tests


def test_prompt_offers_integration_tests_when_pytest_installed():
    prompts, run_integration_tests = _prompts_seen(pytest_installed=True, answer="yes")
    assert any("integration test" in p.lower() for p in prompts)
    assert run_integration_tests


def test_prompt_skips_integration_question_when_pytest_missing(caplog):
    """Asking is pointless when the answer could not be honoured either way."""
    prompts, run_integration_tests = _prompts_seen(pytest_installed=False, answer="yes")
    assert not any("integration test" in p.lower() for p in prompts)
    assert not run_integration_tests
    assert "openfold3[dev]" in caplog.text


@pytest.mark.parametrize(
    ("run_integration_tests", "pytest_installed", "expect_tests_run"),
    [
        pytest.param(True, False, False, id="tests-requested-pytest-missing"),
        pytest.param(True, True, True, id="tests-requested-pytest-present"),
        pytest.param(False, False, False, id="tests-skipped-pytest-irrelevant"),
    ],
)
def test_missing_pytest_warns_without_losing_the_downloads(
    tmp_path, caplog, run_integration_tests, pytest_installed, expect_tests_run
):
    """A missing test dependency downgrades the request; it never aborts setup.

    pytest is not needed to *set up* openfold3, and the downloads are the expensive
    part — exiting over it would force the user to redo the whole run. So the parameters
    and CCD are fetched regardless, and only the tests themselves are dropped.
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
        run_setup(config)

    assert download.called
    assert ccd.called
    assert integration_tests.called == expect_tests_run

    warned = run_integration_tests and not pytest_installed
    assert ("openfold3[dev]" in caplog.text) == warned
