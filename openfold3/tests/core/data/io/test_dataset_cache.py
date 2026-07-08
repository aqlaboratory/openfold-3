import json
import math
from dataclasses import dataclass
from datetime import date
from io import StringIO

import pytest

from openfold3.core.data.io.dataset_cache import (
    _encode_datacache_types,
    write_datacache_to_json,
)
from openfold3.core.data.resources.residues import MoleculeType


@pytest.mark.parametrize(
    "input_obj, expected",
    [
        # NaN floats are converted to None
        pytest.param(float("nan"), None, id="nan-float"),
        pytest.param(math.nan, None, id="math-nan"),
        # Normal floats pass through
        pytest.param(3.14, 3.14, id="normal-float"),
        pytest.param(0.0, 0.0, id="zero-float"),
        pytest.param(-1.0, -1.0, id="negative-float"),
        pytest.param(float("inf"), float("inf"), id="positive-inf"),
        pytest.param(float("-inf"), float("-inf"), id="negative-inf"),
        # Dates are converted to ISO format strings
        pytest.param(date(2025, 1, 15), "2025-01-15", id="date-2025"),
        pytest.param(date(2000, 12, 31), "2000-12-31", id="date-2000"),
        # MoleculeType enums are converted to their name
        pytest.param(MoleculeType.PROTEIN, "PROTEIN", id="molecule-protein"),
        pytest.param(MoleculeType.RNA, "RNA", id="molecule-rna"),
        pytest.param(MoleculeType.DNA, "DNA", id="molecule-dna"),
        pytest.param(MoleculeType.LIGAND, "LIGAND", id="molecule-ligand"),
        # Everything else passes through unchanged
        pytest.param(None, None, id="none-passthrough"),
        pytest.param(42, 42, id="int-passthrough"),
        pytest.param("hello", "hello", id="str-passthrough"),
        pytest.param(True, True, id="bool-passthrough"),
        pytest.param([1, 2, 3], [1, 2, 3], id="list-passthrough"),
        pytest.param({"key": "value"}, {"key": "value"}, id="dict-passthrough"),
    ],
)
def test_encode_datacache_types(input_obj, expected):
    result = _encode_datacache_types(input_obj)
    if expected is None:
        assert result is None
    elif isinstance(expected, float) and math.isinf(expected):
        assert math.isinf(result) and (result > 0) == (expected > 0)
    else:
        assert result == expected


@dataclass
class _DummyCache:
    name: str
    resolution: float | None
    release_date: date
    molecule_type: MoleculeType


@pytest.mark.parametrize(
    "cache, expected_resolution",
    [
        pytest.param(
            _DummyCache("test", None, date(2025, 3, 1), MoleculeType.PROTEIN),
            None,
            id="none-resolution",
        ),
        pytest.param(
            _DummyCache("test", float("nan"), date(2025, 3, 1), MoleculeType.RNA),
            None,
            id="nan-resolution-becomes-null",
        ),
        pytest.param(
            _DummyCache("test", 2.5, date(2025, 3, 1), MoleculeType.DNA),
            2.5,
            id="normal-resolution",
        ),
        pytest.param(
            _DummyCache("test", 0.0, date(2025, 3, 1), MoleculeType.LIGAND),
            0.0,
            id="zero-resolution",
        ),
    ],
)
def test_write_datacache_to_json(cache, expected_resolution):
    buf = StringIO()
    write_datacache_to_json(cache, buf)
    raw = buf.getvalue()

    assert "NaN" not in raw
    result = json.loads(raw)  # valid JSON
    assert result["resolution"] == expected_resolution
