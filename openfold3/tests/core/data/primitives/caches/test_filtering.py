from datetime import date

import pytest

from openfold3.core.data.primitives.caches.filtering import filter_by_resolution
from openfold3.core.data.primitives.caches.format import PreprocessingStructureData


def _make_structure(resolution, experimental_method="X-RAY DIFFRACTION"):
    return PreprocessingStructureData(
        status="success",
        release_date=date(2025, 1, 1),
        experimental_method=experimental_method,
        resolution=resolution,
        chains={},
        interfaces=[],
        token_count=100,
    )


# Dummy cache shared across tests
_CACHE = {
    "HIGH": _make_structure(2.0),
    "MID": _make_structure(5.0),
    "LOW": _make_structure(9.0),
    "VLOW": _make_structure(9.1),
    "NORES": _make_structure(None),
    "NMR1": _make_structure(None, experimental_method="SOLUTION NMR"),
    "NMR2": _make_structure(None, experimental_method="SOLID-STATE NMR"),
}


@pytest.mark.parametrize(
    "max_resolution, ignore_nmr, expected_pdb_ids",
    [
        pytest.param(
            9.0,
            True,
            {"HIGH", "MID", "LOW", "NMR1", "NMR2"},
            id="default-keeps-nmr-filters-none-resolution",
        ),
        pytest.param(
            9.0,
            False,
            {"HIGH", "MID", "LOW"},
            id="no-ignore-nmr-filters-all-without-resolution",
        ),
        pytest.param(
            2.0,
            True,
            {"HIGH", "NMR1", "NMR2"},
            id="strict-resolution-keeps-nmr",
        ),
        pytest.param(
            2.0,
            False,
            {"HIGH"},
            id="strict-resolution-no-nmr",
        ),
        pytest.param(
            100.0,
            True,
            {"HIGH", "MID", "LOW", "VLOW", "NMR1", "NMR2"},
            id="very-high-max-keeps-all-except-none-non-nmr",
        ),
        pytest.param(
            0.0,
            True,
            {"NMR1", "NMR2"},
            id="zero-max-only-nmr-survive",
        ),
    ],
)
def test_filter_by_resolution(max_resolution, ignore_nmr, expected_pdb_ids):
    result = filter_by_resolution(_CACHE, max_resolution, ignore_nmr=ignore_nmr)
    assert set(result.keys()) == expected_pdb_ids
