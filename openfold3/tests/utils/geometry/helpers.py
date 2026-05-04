"""Shared helpers for geometry tests."""

from __future__ import annotations

import math

import torch

from openfold3.core.utils.geometry.rigid_matrix_vector import Rigid3Array
from openfold3.core.utils.geometry.rotation_matrix import Rot3Array
from openfold3.core.utils.geometry.vector import Vec3Array

_Translation = tuple[float, float, float]


def v(x: float, y: float, z: float) -> Vec3Array:
    """Build a scalar Vec3Array from three floats."""
    return Vec3Array(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(z, dtype=torch.float32),
    )


def vb(coords: list[list[float]]) -> Vec3Array:
    """Build a batched Vec3Array from a list of [x, y, z] triples."""
    t = torch.tensor(coords, dtype=torch.float32)
    return Vec3Array(t[:, 0], t[:, 1], t[:, 2])


def rot_x(theta: float) -> Rot3Array:
    """Rotation about the X axis by *theta* radians."""
    c, s = math.cos(theta), math.sin(theta)
    return Rot3Array.from_array(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ]
        )
    )


def rot_y(theta: float) -> Rot3Array:
    """Rotation about the Y axis by *theta* radians."""
    c, s = math.cos(theta), math.sin(theta)
    return Rot3Array.from_array(
        torch.tensor(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ]
        )
    )


def rot_z(theta: float) -> Rot3Array:
    """Rotation about the Z axis by *theta* radians."""
    c, s = math.cos(theta), math.sin(theta)
    return Rot3Array.from_array(
        torch.tensor(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )


def rigid(rot: Rot3Array, translation: _Translation) -> Rigid3Array:
    """Build a Rigid3Array from a rotation and a translation triple."""
    return Rigid3Array(rot, v(*translation))
