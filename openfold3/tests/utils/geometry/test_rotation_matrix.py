"""Tests for Rot3Array rotation matrix class."""

from __future__ import annotations

import math

import pytest
import torch

from openfold3.core.utils.geometry.rotation_matrix import Rot3Array
from openfold3.core.utils.geometry.vector import Vec3Array
from openfold3.tests.utils.geometry.helpers import rot_x as _rot_x
from openfold3.tests.utils.geometry.helpers import rot_y as _rot_y
from openfold3.tests.utils.geometry.helpers import rot_z as _rot_z
from openfold3.tests.utils.geometry.helpers import v as _v

# ===================================================================
# Construction & conversion
# ===================================================================


class TestConstruction:
    def test_identity_is_eye(self):
        eye = Rot3Array.identity((1,), device="cpu")
        expected = torch.eye(3).unsqueeze(0)
        assert torch.allclose(eye.to_tensor(), expected)

    def test_from_array_round_trip(self):
        mat = torch.tensor(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rot = Rot3Array.from_array(mat)
        assert torch.allclose(rot.to_tensor(), mat)

    def test_from_array_batched(self):
        batch = torch.stack([torch.eye(3), torch.eye(3)])
        rot = Rot3Array.from_array(batch)
        assert rot.xx.shape == (2,)

    def test_from_quaternion_identity(self):
        # Quaternion (1, 0, 0, 0) -> identity rotation
        rot = Rot3Array.from_quaternion(
            w=torch.tensor(1.0),
            x=torch.tensor(0.0),
            y=torch.tensor(0.0),
            z=torch.tensor(0.0),
        )
        expected = torch.eye(3)
        assert torch.allclose(rot.to_tensor(), expected, atol=1e-6)

    @pytest.mark.parametrize(
        "w,x,y,z",
        [
            pytest.param(0.0, 1.0, 0.0, 0.0, id="180-about-x"),
            pytest.param(0.0, 0.0, 1.0, 0.0, id="180-about-y"),
            pytest.param(0.0, 0.0, 0.0, 1.0, id="180-about-z"),
        ],
    )
    def test_from_quaternion_180(self, w, x, y, z):
        rot = Rot3Array.from_quaternion(
            w=torch.tensor(w),
            x=torch.tensor(x),
            y=torch.tensor(y),
            z=torch.tensor(z),
        )
        # R @ R should give identity for 180-degree rotations
        composed = rot @ rot
        assert torch.allclose(composed.to_tensor(), torch.eye(3), atol=1e-6)

    def test_from_quaternion_90_about_z(self):
        # 90 deg about Z: w=cos(45)=sqrt(2)/2, z=sin(45)=sqrt(2)/2
        s = math.sqrt(2) / 2
        rot = Rot3Array.from_quaternion(
            w=torch.tensor(s),
            x=torch.tensor(0.0),
            y=torch.tensor(0.0),
            z=torch.tensor(s),
        )
        expected = _rot_z(math.pi / 2)
        assert torch.allclose(rot.to_tensor(), expected.to_tensor(), atol=1e-6)


# ===================================================================
# from_two_vectors
# ===================================================================

_TWO_VEC_CASES = [
    pytest.param(
        _v(1, 0, 0),
        _v(0, 1, 0),
        id="standard-xy",
    ),
    pytest.param(
        _v(0, 0, 1),
        _v(0, 1, 0),
        id="z-and-y",
    ),
    pytest.param(
        _v(3, 0, 0),
        _v(1, 2, 0),
        id="scaled-e0-tilted-e1",
    ),
]


class TestFromTwoVectors:
    @pytest.mark.parametrize("e0,e1", _TWO_VEC_CASES)
    def test_result_is_orthogonal(self, e0, e1):
        """R^T R should equal identity."""
        rot = Rot3Array.from_two_vectors(e0, e1)
        rtr = (rot.inverse() @ rot).to_tensor()
        assert torch.allclose(rtr, torch.eye(3), atol=1e-5)

    @pytest.mark.parametrize("e0,e1", _TWO_VEC_CASES)
    def test_det_is_one(self, e0, e1):
        """Proper rotation has determinant +1."""
        rot = Rot3Array.from_two_vectors(e0, e1)
        det = torch.det(rot.to_tensor())
        assert torch.allclose(det, torch.tensor(1.0), atol=1e-5)

    def test_e0_maps_to_x_axis(self):
        e0 = _v(0, 0, 5)
        e1 = _v(0, 3, 0)
        rot = Rot3Array.from_two_vectors(e0, e1)
        # The constructed frame's first column is the normalized e0 direction,
        # so applying the *inverse* frame to e0 should give the +x axis.
        result = rot.apply_inverse_to_point(e0.normalized(epsilon=0))
        assert torch.allclose(result.to_tensor(), _v(1, 0, 0).to_tensor(), atol=1e-5)


# ===================================================================
# Inverse
# ===================================================================


class TestInverse:
    @pytest.mark.parametrize(
        "rot",
        [
            pytest.param(Rot3Array.identity((), device="cpu"), id="identity"),
            pytest.param(_rot_z(math.pi / 4), id="45-deg-z"),
            pytest.param(_rot_x(math.pi / 3), id="60-deg-x"),
        ],
    )
    def test_inverse_is_transpose(self, rot):
        inv = rot.inverse()
        assert torch.allclose(inv.to_tensor(), rot.to_tensor().T, atol=1e-6)

    @pytest.mark.parametrize(
        "rot",
        [
            pytest.param(Rot3Array.identity((), device="cpu"), id="identity"),
            pytest.param(_rot_z(math.pi / 2), id="90-deg-z"),
            pytest.param(_rot_y(1.0), id="1-rad-y"),
        ],
    )
    def test_R_Rinv_eq_identity(self, rot):
        composed = rot @ rot.inverse()
        assert torch.allclose(composed.to_tensor(), torch.eye(3), atol=1e-6)


# ===================================================================
# apply_to_point
# ===================================================================

_APPLY_CASES = [
    pytest.param(
        _rot_z(math.pi / 2),
        _v(1, 0, 0),
        _v(0, 1, 0),
        id="90z-rotates-x-to-y",
    ),
    pytest.param(
        _rot_x(math.pi / 2),
        _v(0, 1, 0),
        _v(0, 0, 1),
        id="90x-rotates-y-to-z",
    ),
    pytest.param(
        _rot_y(math.pi / 2),
        _v(0, 0, 1),
        _v(1, 0, 0),
        id="90y-rotates-z-to-x",
    ),
    pytest.param(
        _rot_z(math.pi),
        _v(1, 0, 0),
        _v(-1, 0, 0),
        id="180z-flips-x",
    ),
]


class TestApplyToPoint:
    @pytest.mark.parametrize("rot,point,expected", _APPLY_CASES)
    def test_known_rotations(self, rot, point, expected):
        result = rot.apply_to_point(point)
        assert torch.allclose(result.to_tensor(), expected.to_tensor(), atol=1e-5)

    def test_identity_is_noop(self):
        p = _v(1, 2, 3)
        eye = Rot3Array.identity((), device="cpu")
        result = eye.apply_to_point(p)
        assert torch.allclose(result.to_tensor(), p.to_tensor())

    def test_apply_inverse_undoes_apply(self):
        rot = _rot_z(math.pi / 6)  # 30 degrees about Z
        p = _v(1, 2, 3)
        transformed = rot.apply_to_point(p)
        recovered = rot.apply_inverse_to_point(transformed)
        assert torch.allclose(recovered.to_tensor(), p.to_tensor(), atol=1e-5)


# ===================================================================
# Composition (matmul)
# ===================================================================


class TestComposition:
    def test_two_90z_eq_180z(self):
        r90 = _rot_z(math.pi / 2)
        r180 = r90 @ r90
        expected = _rot_z(math.pi)
        assert torch.allclose(r180.to_tensor(), expected.to_tensor(), atol=1e-5)

    def test_xyz_composition(self):
        """Composing rotations about different axes."""
        rx = _rot_x(math.pi / 2)
        ry = _rot_y(math.pi / 2)
        composed = rx @ ry
        # Verify by applying to a test point
        p = _v(1, 0, 0)
        result_composed = composed.apply_to_point(p)
        result_sequential = rx.apply_to_point(ry.apply_to_point(p))
        assert torch.allclose(
            result_composed.to_tensor(), result_sequential.to_tensor(), atol=1e-5
        )

    def test_identity_is_neutral(self):
        r = _rot_z(0.7)
        eye = Rot3Array.identity((), device="cpu")
        assert torch.allclose((eye @ r).to_tensor(), r.to_tensor(), atol=1e-6)
        assert torch.allclose((r @ eye).to_tensor(), r.to_tensor(), atol=1e-6)


# ===================================================================
# Scalar multiply
# ===================================================================


class TestScalarMultiply:
    def test_mul_by_one_is_noop(self):
        r = _rot_z(math.pi / 4)
        result = r * torch.tensor(1.0)
        assert torch.allclose(result.to_tensor(), r.to_tensor())

    def test_mul_by_zero(self):
        r = _rot_z(math.pi / 4)
        result = r * torch.tensor(0.0)
        assert torch.allclose(result.to_tensor(), torch.zeros(3, 3))

    def test_mul_by_pi(self):
        r = _rot_z(math.pi / 4)
        result = r * torch.tensor(math.pi)
        s = math.pi / math.sqrt(2)
        expected_tensor = torch.tensor(
            [
                [s, -s, 0.0],
                [s, s, 0.0],
                [0.0, 0.0, math.pi],
            ]
        )
        assert torch.allclose(result.to_tensor(), expected_tensor, atol=1e-6)


# ===================================================================
# Batched operations
# ===================================================================


class TestBatched:
    def test_identity_batch(self):
        rot = Rot3Array.identity((3,), device="cpu")
        assert rot.xx.shape == (3,)
        assert torch.allclose(rot.to_tensor(), torch.eye(3).expand(3, 3, 3))

    def test_indexing(self):
        r0 = _rot_z(0.0)
        r1 = _rot_z(math.pi / 2)
        batch = Rot3Array.cat(
            [
                r0.map_tensor_fn(lambda t: t.unsqueeze(0)),
                r1.map_tensor_fn(lambda t: t.unsqueeze(0)),
            ],
            dim=0,
        )
        assert batch.xx.shape == (2,)
        first = batch[0]
        assert torch.allclose(first.to_tensor(), r0.to_tensor(), atol=1e-6)

    def test_apply_to_point_batched(self):
        """Batch of two rotations applied to a batch of two points."""
        batch = Rot3Array.from_array(
            torch.stack(
                [
                    _rot_z(0.0).to_tensor(),
                    _rot_z(math.pi / 2).to_tensor(),
                ]
            )
        )
        points = Vec3Array(
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
        )
        result = batch.apply_to_point(points)
        # First rotation is identity -> (1,0,0) unchanged
        assert torch.allclose(result.x[0], torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(result.y[0], torch.tensor(0.0), atol=1e-5)
        # Second rotation is 90-deg Z -> (1,0,0) becomes (0,1,0)
        assert torch.allclose(result.x[1], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(result.y[1], torch.tensor(1.0), atol=1e-5)

    def test_reshape(self):
        rot = Rot3Array.identity((2, 3), device="cpu")
        reshaped = rot.reshape((6,))
        assert reshaped.xx.shape == (6,)

    def test_cat(self):
        a = Rot3Array.identity((2,), device="cpu")
        b = Rot3Array.identity((3,), device="cpu")
        c = Rot3Array.cat([a, b], dim=0)
        assert c.xx.shape == (5,)


# ===================================================================
# Gradient control
# ===================================================================


class TestStopGradient:
    def test_stop_gradient_detaches(self):
        t = torch.tensor(1.0, requires_grad=True)
        rot = Rot3Array(t, t, t, t, t, t, t, t, t)
        stopped = rot.stop_gradient()
        assert not stopped.xx.requires_grad
