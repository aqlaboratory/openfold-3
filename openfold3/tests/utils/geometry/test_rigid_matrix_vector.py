"""Tests for Rigid3Array SE(3) transformation class."""

from __future__ import annotations

import math

import pytest
import torch

from openfold3.core.utils.geometry.rigid_matrix_vector import Rigid3Array
from openfold3.core.utils.geometry.rotation_matrix import Rot3Array
from openfold3.core.utils.geometry.vector import Vec3Array
from openfold3.tests.utils.geometry.helpers import rigid as _rigid
from openfold3.tests.utils.geometry.helpers import rot_x as _rot_x
from openfold3.tests.utils.geometry.helpers import rot_z as _rot_z
from openfold3.tests.utils.geometry.helpers import v as _v

# ===================================================================
# Construction & conversion
# ===================================================================


class TestConstruction:
    def test_identity(self):
        rig = Rigid3Array.identity((), device="cpu")
        assert torch.allclose(rig.rotation.to_tensor(), torch.eye(3))
        assert torch.allclose(rig.translation.to_tensor(), torch.zeros(3))

    def test_identity_batched(self):
        rig = Rigid3Array.identity((4,), device="cpu")
        assert rig.shape == (4,)

    def test_from_array_round_trip(self):
        mat = torch.eye(4)
        mat[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
        rig = Rigid3Array.from_array(mat)
        recovered = rig.to_tensor()
        assert torch.allclose(recovered, mat)

    def test_from_array4x4(self):
        mat = torch.eye(4)
        mat[:3, 3] = torch.tensor([10.0, 20.0, 30.0])
        rig = Rigid3Array.from_array4x4(mat)
        assert torch.allclose(
            rig.translation.to_tensor(), torch.tensor([10.0, 20.0, 30.0])
        )

    def test_to_tensor_has_homogeneous_row(self):
        rig = _rigid(_rot_z(0.5), (1.0, 2.0, 3.0))
        mat = rig.to_tensor()
        assert mat.shape == (4, 4)
        assert torch.allclose(mat[3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]))


# ===================================================================
# Properties
# ===================================================================


class TestProperties:
    def test_shape(self):
        rig = Rigid3Array.identity((2, 3), device="cpu")
        assert rig.shape == (2, 3)

    def test_dtype(self):
        rig = Rigid3Array.identity((), device="cpu")
        assert rig.dtype == torch.float32

    def test_device(self):
        rig = Rigid3Array.identity((), device="cpu")
        assert rig.device == torch.device("cpu")


# ===================================================================
# apply_to_point
# ===================================================================

_APPLY_CASES = [
    pytest.param(
        _rigid(Rot3Array.identity((), "cpu"), (1, 0, 0)),
        _v(0, 0, 0),
        _v(1, 0, 0),
        id="pure-translation-x",
    ),
    pytest.param(
        _rigid(Rot3Array.identity((), "cpu"), (0, 5, 0)),
        _v(1, 2, 3),
        _v(1, 7, 3),
        id="translate-y-by-5",
    ),
    pytest.param(
        _rigid(_rot_z(math.pi / 2), (0, 0, 0)),
        _v(1, 0, 0),
        _v(0, 1, 0),
        id="pure-90z-rotation",
    ),
    pytest.param(
        _rigid(_rot_z(math.pi / 2), (10, 0, 0)),
        _v(1, 0, 0),
        _v(10, 1, 0),
        id="rotate-90z-then-translate",
    ),
]


class TestApplyToPoint:
    @pytest.mark.parametrize("rig,point,expected", _APPLY_CASES)
    def test_known_transforms(self, rig, point, expected):
        result = rig.apply_to_point(point)
        assert torch.allclose(result.to_tensor(), expected.to_tensor(), atol=1e-5)

    def test_apply_tensor_interface(self):
        """apply() accepts a raw [..., 3] tensor and returns one."""
        rig = _rigid(Rot3Array.identity((), "cpu"), (1, 2, 3))
        point = torch.tensor([0.0, 0.0, 0.0])
        result = rig.apply(point)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_identity_is_noop(self):
        rig = Rigid3Array.identity((), device="cpu")
        p = _v(5, 10, 15)
        result = rig.apply_to_point(p)
        assert torch.allclose(result.to_tensor(), p.to_tensor())


# ===================================================================
# apply_inverse_to_point
# ===================================================================


class TestApplyInverse:
    @pytest.mark.parametrize(
        "rig,point",
        [
            pytest.param(
                _rigid(Rot3Array.identity((), "cpu"), (1, 2, 3)),
                _v(5, 6, 7),
                id="pure-translation",
            ),
            pytest.param(
                _rigid(_rot_z(math.pi / 4), (0, 0, 0)),
                _v(1, 0, 0),
                id="pure-rotation-45z",
            ),
            pytest.param(
                _rigid(_rot_x(math.pi / 3), (10, 20, 30)),
                _v(1, 2, 3),
                id="rotation-and-translation",
            ),
        ],
    )
    def test_apply_then_inverse_recovers_point(self, rig, point):
        transformed = rig.apply_to_point(point)
        recovered = rig.apply_inverse_to_point(transformed)
        assert torch.allclose(recovered.to_tensor(), point.to_tensor(), atol=1e-5)

    def test_invert_apply_tensor_interface(self):
        """invert_apply() accepts a raw [..., 3] tensor and returns one."""
        rig = _rigid(Rot3Array.identity((), "cpu"), (1, 2, 3))
        point = torch.tensor([1.0, 2.0, 3.0])
        result = rig.invert_apply(point)
        assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0]), atol=1e-5)


# ===================================================================
# Inverse
# ===================================================================


class TestInverse:
    @pytest.mark.parametrize(
        "rig",
        [
            pytest.param(
                Rigid3Array.identity((), device="cpu"),
                id="identity",
            ),
            pytest.param(
                _rigid(Rot3Array.identity((), "cpu"), (1, 2, 3)),
                id="pure-translation",
            ),
            pytest.param(
                _rigid(_rot_z(math.pi / 2), (0, 0, 0)),
                id="pure-rotation",
            ),
            pytest.param(
                _rigid(_rot_z(0.7), (5, -3, 8)),
                id="general-transform",
            ),
        ],
    )
    def test_T_Tinv_eq_identity(self, rig):
        composed = rig @ rig.inverse()
        # Rotation should be identity
        assert torch.allclose(composed.rotation.to_tensor(), torch.eye(3), atol=1e-5)
        # Translation should be zero
        assert torch.allclose(
            composed.translation.to_tensor(), torch.zeros(3), atol=1e-5
        )

    def test_inverse_of_pure_translation(self):
        rig = _rigid(Rot3Array.identity((), "cpu"), (1, 2, 3))
        inv = rig.inverse()
        assert torch.allclose(
            inv.translation.to_tensor(), torch.tensor([-1.0, -2.0, -3.0])
        )


# ===================================================================
# Composition (matmul)
# ===================================================================

_COMPOSE_CASES = [
    pytest.param(
        _rigid(Rot3Array.identity((), "cpu"), (1, 0, 0)),
        _rigid(Rot3Array.identity((), "cpu"), (0, 2, 0)),
        _v(0, 0, 0),
        _v(1, 2, 0),
        id="two-translations-add",
    ),
    pytest.param(
        _rigid(_rot_z(math.pi / 2), (0, 0, 0)),
        _rigid(Rot3Array.identity((), "cpu"), (1, 0, 0)),
        _v(0, 0, 0),
        # First translate (1,0,0), then rotate 90-Z -> (0,1,0)
        _v(0, 1, 0),
        id="rotate-after-translate",
    ),
    pytest.param(
        _rigid(Rot3Array.identity((), "cpu"), (1, 0, 0)),
        _rigid(_rot_z(math.pi / 2), (0, 0, 0)),
        _v(1, 0, 0),
        # First rotate (1,0,0) -> (0,1,0), then translate by (1,0,0) -> (1,1,0)
        _v(1, 1, 0),
        id="translate-after-rotate",
    ),
]


class TestComposition:
    @pytest.mark.parametrize("t1,t2,point,expected", _COMPOSE_CASES)
    def test_compose_apply(self, t1, t2, point, expected):
        composed = t1 @ t2
        result = composed.apply_to_point(point)
        assert torch.allclose(result.to_tensor(), expected.to_tensor(), atol=1e-5)

    def test_compose_method_matches_matmul(self):
        a = _rigid(_rot_z(0.5), (1, 2, 3))
        b = _rigid(_rot_x(0.3), (4, 5, 6))
        via_matmul = a @ b
        via_method = a.compose(b)
        assert torch.allclose(via_matmul.to_tensor(), via_method.to_tensor(), atol=1e-6)

    def test_identity_is_neutral(self):
        rig = _rigid(_rot_z(1.0), (5, 10, 15))
        eye = Rigid3Array.identity((), device="cpu")
        assert torch.allclose((eye @ rig).to_tensor(), rig.to_tensor(), atol=1e-6)
        assert torch.allclose((rig @ eye).to_tensor(), rig.to_tensor(), atol=1e-6)

    def test_compose_rotation(self):
        rig = _rigid(Rot3Array.identity((), "cpu"), (1, 2, 3))
        rot = _rot_z(math.pi / 2)
        result = rig.compose_rotation(rot)
        # Translation is unchanged (cloned)
        assert torch.allclose(
            result.translation.to_tensor(), torch.tensor([1.0, 2.0, 3.0])
        )
        # Rotation is now the composed rotation
        assert torch.allclose(result.rotation.to_tensor(), rot.to_tensor(), atol=1e-6)


# ===================================================================
# Scalar multiply & scale_translation
# ===================================================================


class TestScaling:
    def test_scalar_mul(self):
        rig = _rigid(Rot3Array.identity((), "cpu"), (1, 2, 3))
        scaled = rig * torch.tensor(2.0)
        # Both rotation entries and translation scale
        assert torch.allclose(
            scaled.translation.to_tensor(), torch.tensor([2.0, 4.0, 6.0])
        )

    def test_scale_translation(self):
        rig = _rigid(_rot_z(math.pi / 4), (2, 4, 6))
        scaled = rig.scale_translation(0.5)
        # Rotation should be unchanged
        assert torch.allclose(scaled.rotation.to_tensor(), rig.rotation.to_tensor())
        # Translation should be halved
        assert torch.allclose(
            scaled.translation.to_tensor(), torch.tensor([1.0, 2.0, 3.0])
        )


# ===================================================================
# Indexing, reshape, unsqueeze, cat
# ===================================================================


class TestIndexing:
    def test_getitem(self):
        a = _rigid(Rot3Array.identity((), "cpu"), (1, 0, 0))
        b = _rigid(Rot3Array.identity((), "cpu"), (0, 2, 0))
        batch = Rigid3Array.cat(
            [
                a.map_tensor_fn(lambda t: t.unsqueeze(0)),
                b.map_tensor_fn(lambda t: t.unsqueeze(0)),
            ],
            dim=0,
        )
        assert batch.shape == (2,)
        first = batch[0]
        assert torch.allclose(
            first.translation.to_tensor(), torch.tensor([1.0, 0.0, 0.0])
        )

    def test_unsqueeze(self):
        rig = Rigid3Array.identity((3,), device="cpu")
        u = rig.unsqueeze(0)
        assert u.shape == (1, 3)

    def test_reshape(self):
        rig = Rigid3Array.identity((2, 3), device="cpu")
        reshaped = rig.reshape((6,))
        assert reshaped.shape == (6,)

    def test_cat(self):
        a = Rigid3Array.identity((2,), device="cpu")
        b = Rigid3Array.identity((3,), device="cpu")
        c = Rigid3Array.cat([a, b], dim=0)
        assert c.shape == (5,)


# ===================================================================
# Gradient control
# ===================================================================


class TestStopRotGradient:
    def test_rotation_detached_translation_kept(self):
        rot_param = torch.tensor(1.0, requires_grad=True)
        rot = Rot3Array(
            rot_param,
            rot_param,
            rot_param,
            rot_param,
            rot_param,
            rot_param,
            rot_param,
            rot_param,
            rot_param,
        )
        trans_param = torch.tensor(2.0, requires_grad=True)
        trans = Vec3Array(trans_param, trans_param, trans_param)

        rig = Rigid3Array(rot, trans)
        stopped = rig.stop_rot_gradient()
        assert not stopped.rotation.xx.requires_grad
        assert stopped.translation.x.requires_grad


# ===================================================================
# Batched apply
# ===================================================================


class TestBatchedApply:
    def test_batch_of_transforms_on_batch_of_points(self):
        """Two different transforms applied element-wise to two points."""
        # Transform 0: pure translation by (10, 0, 0)
        # Transform 1: 90-deg Z rotation, no translation
        r0 = Rot3Array.identity((), "cpu")
        r1 = _rot_z(math.pi / 2)
        rot = Rot3Array.from_array(torch.stack([r0.to_tensor(), r1.to_tensor()]))
        trans = Vec3Array(
            torch.tensor([10.0, 0.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
        )
        rig = Rigid3Array(rot, trans)

        points = Vec3Array(
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.0]),
        )

        result = rig.apply_to_point(points)
        # Point 0: identity rot + translate (10,0,0) -> (11, 0, 0)
        assert torch.allclose(result.x[0], torch.tensor(11.0), atol=1e-5)
        assert torch.allclose(result.y[0], torch.tensor(0.0), atol=1e-5)
        # Point 1: 90-Z rotation on (1,0,0) -> (0,1,0), no translation
        assert torch.allclose(result.x[1], torch.tensor(0.0), atol=1e-5)
        assert torch.allclose(result.y[1], torch.tensor(1.0), atol=1e-5)
