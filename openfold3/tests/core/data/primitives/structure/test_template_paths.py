"""Tests for template structure path resolution."""

from pathlib import Path

import pytest

from openfold3.core.data.primitives.structure.template_paths import (
    iter_template_structure_paths,
    resolve_template_structure_path,
)


def test_literal_directory_resolution(tmp_path: Path):
    structure = tmp_path / "1abc.cif"
    structure.touch()

    assert resolve_template_structure_path(tmp_path, "1abc", "1abc.cif") == structure


def test_sharded_directory_resolution(tmp_path: Path):
    structure = tmp_path / "md" / "4md7.cif"
    structure.parent.mkdir()
    structure.touch()

    assert resolve_template_structure_path(tmp_path, "4md7", "4md7.cif") == structure


def test_flat_directory_takes_precedence(tmp_path: Path):
    flat = tmp_path / "4md7.cif"
    sharded = tmp_path / "md" / "4md7.cif"
    sharded.parent.mkdir()
    flat.touch()
    sharded.touch()

    assert resolve_template_structure_path(tmp_path, "4md7", "4md7.cif") == flat


def test_discovery_finds_flat_and_sharded_files_once(tmp_path: Path):
    flat = tmp_path / "1abc.cif"
    sharded = tmp_path / "de" / "2def.cif"
    duplicate = tmp_path / "ab" / "1abc.cif"
    sharded.parent.mkdir()
    duplicate.parent.mkdir()
    flat.touch()
    sharded.touch()
    duplicate.touch()

    assert list(iter_template_structure_paths(tmp_path, "cif")) == [flat, sharded]


def test_short_id_uses_flat_directory(tmp_path: Path):
    structure = tmp_path / "ab.cif"
    structure.touch()

    assert resolve_template_structure_path(tmp_path, "ab", "ab.cif") == structure


def test_missing_template_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No template structure found"):
        resolve_template_structure_path(tmp_path, "missing", "missing.cif")
