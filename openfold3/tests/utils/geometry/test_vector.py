"""Tests for Vec3Array and geometry free functions."""

from __future__ import annotations

import math

import pytest
import torch

from openfold3.core.utils.geometry.vector import (
    Vec3Array,
    cross,
    dihedral_angle,
    dot,
    euclidean_distance,
    norm,
    normalized,
    square_euclidean_distance,
)
from openfold3.tests.utils.geometry.helpers import v as _v
from openfold3.tests.utils.geometry.helpers import vb as _vb

# ===================================================================
# Construction & round-trip
# ===================================================================


class TestConstruction:
    def test_from_array_round_trip(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        v = Vec3Array.from_array(tensor)
        assert torch.equal(v.x, torch.tensor([1.0, 4.0]))
        assert torch.equal(v.y, torch.tensor([2.0, 5.0]))
        assert torch.equal(v.z, torch.tensor([3.0, 6.0]))
        assert torch.equal(v.to_tensor(), tensor)

    def test_zeros(self):
        v = Vec3Array.zeros((2, 3))
        assert v.shape == (2, 3)
        assert torch.equal(v.x, torch.zeros(2, 3))

    def test_shape_property(self):
        v = _vb([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert v.shape == (3,)

    def test_cat(self):
        a = _vb([[1, 0, 0], [0, 1, 0]])
        b = _vb([[0, 0, 1]])
        c = Vec3Array.cat([a, b], dim=0)
        assert c.shape == (3,)
        assert torch.allclose(c.z, torch.tensor([0.0, 0.0, 1.0]))


# ===================================================================
# Arithmetic operators
# ===================================================================

_ARITH_CASES = [
    pytest.param(
        _v(1, 2, 3),
        _v(4, 5, 6),
        _v(5, 7, 9),
        _v(-3, -3, -3),
        id="simple-integers",
    ),
    pytest.param(
        _v(0, 0, 0),
        _v(1, 2, 3),
        _v(1, 2, 3),
        _v(-1, -2, -3),
        id="zero-vector",
    ),
]


class TestArithmetic:
    @pytest.mark.parametrize("a,b,expected_sum,expected_diff", _ARITH_CASES)
    def test_add(self, a, b, expected_sum, expected_diff):
        result = a + b
        assert torch.allclose(result.to_tensor(), expected_sum.to_tensor())

    @pytest.mark.parametrize("a,b,expected_sum,expected_diff", _ARITH_CASES)
    def test_sub(self, a, b, expected_sum, expected_diff):
        result = a - b
        assert torch.allclose(result.to_tensor(), expected_diff.to_tensor())

    def test_scalar_mul(self):
        v = _v(1, 2, 3)
        result = v * 2.0
        assert torch.allclose(result.to_tensor(), _v(2, 4, 6).to_tensor())

    def test_rmul(self):
        v = _v(1, 2, 3)
        result = 3.0 * v
        assert torch.allclose(result.to_tensor(), _v(3, 6, 9).to_tensor())

    def test_truediv(self):
        v = _v(2, 4, 6)
        result = v / 2.0
        assert torch.allclose(result.to_tensor(), _v(1, 2, 3).to_tensor())

    def test_neg(self):
        v = _v(1, -2, 3)
        result = -v
        assert torch.allclose(result.to_tensor(), _v(-1, 2, -3).to_tensor())

    def test_pos(self):
        v = _v(1, -2, 3)
        result = +v
        assert torch.allclose(result.to_tensor(), v.to_tensor())


# ===================================================================
# Indexing, iteration, reshape
# ===================================================================


class TestIndexing:
    def test_getitem_single(self):
        v = _vb([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        second = v[1]
        assert torch.allclose(second.to_tensor(), torch.tensor([0.0, 1.0, 0.0]))

    def test_getitem_slice(self):
        v = _vb([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        sliced = v[:2]
        assert sliced.shape == (2,)

    def test_iter_yields_xyz_tensors(self):
        v = _v(1, 2, 3)
        x, y, z = v
        assert torch.equal(x, torch.tensor(1.0))
        assert torch.equal(y, torch.tensor(2.0))
        assert torch.equal(z, torch.tensor(3.0))

    def test_reshape(self):
        v = _vb([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        reshaped = v.reshape((2, 2))
        assert reshaped.shape == (2, 2)

    def test_unsqueeze(self):
        v = _vb([[1, 0, 0], [0, 1, 0]])
        u = v.unsqueeze(0)
        assert u.shape == (1, 2)

    def test_sum(self):
        v = _vb([[1, 2, 3], [4, 5, 6]])
        s = v.sum(dim=0)
        assert torch.allclose(s.to_tensor(), torch.tensor([5.0, 7.0, 9.0]))

    def test_clone(self):
        v = _v(1, 2, 3)
        c = v.clone()
        assert torch.equal(v.to_tensor(), c.to_tensor())
        # clone should produce independent storage
        assert v.x.data_ptr() != c.x.data_ptr()


# ===================================================================
# Dot product
# ===================================================================

_DOT_CASES = [
    pytest.param(_v(1, 0, 0), _v(1, 0, 0), 1.0, id="parallel-unit"),
    pytest.param(_v(1, 0, 0), _v(0, 1, 0), 0.0, id="perpendicular"),
    pytest.param(_v(1, 0, 0), _v(-1, 0, 0), -1.0, id="antiparallel"),
    pytest.param(_v(1, 2, 3), _v(4, 5, 6), 32.0, id="general"),
]


class TestDot:
    @pytest.mark.parametrize("a,b,expected", _DOT_CASES)
    def test_method(self, a, b, expected):
        assert torch.allclose(a.dot(b), torch.tensor(expected))

    @pytest.mark.parametrize("a,b,expected", _DOT_CASES)
    def test_free_function(self, a, b, expected):
        assert torch.allclose(dot(a, b), torch.tensor(expected))


# ===================================================================
# Cross product
# ===================================================================

_CROSS_CASES = [
    pytest.param(_v(1, 0, 0), _v(0, 1, 0), _v(0, 0, 1), id="x-cross-y-eq-z"),
    pytest.param(_v(0, 1, 0), _v(0, 0, 1), _v(1, 0, 0), id="y-cross-z-eq-x"),
    pytest.param(_v(0, 0, 1), _v(1, 0, 0), _v(0, 1, 0), id="z-cross-x-eq-y"),
    pytest.param(_v(1, 0, 0), _v(1, 0, 0), _v(0, 0, 0), id="parallel-gives-zero"),
    pytest.param(_v(2, 0, 0), _v(0, 3, 0), _v(0, 0, 6), id="scaled-axes"),
]


class TestCross:
    @pytest.mark.parametrize("a,b,expected", _CROSS_CASES)
    def test_method(self, a, b, expected):
        result = a.cross(b)
        assert torch.allclose(result.to_tensor(), expected.to_tensor())

    @pytest.mark.parametrize("a,b,expected", _CROSS_CASES)
    def test_free_function(self, a, b, expected):
        result = cross(a, b)
        assert torch.allclose(result.to_tensor(), expected.to_tensor())

    def test_anticommutativity(self):
        a, b = _v(1, 2, 3), _v(4, 5, 6)
        assert torch.allclose(a.cross(b).to_tensor(), (-b.cross(a)).to_tensor())


# ===================================================================
# Norm / normalized
# ===================================================================

_NORM_CASES = [
    pytest.param(_v(1, 0, 0), 1.0, id="unit-x"),
    pytest.param(_v(0, 3, 0), 3.0, id="along-y"),
    pytest.param(_v(3, 4, 0), 5.0, id="3-4-5-triangle"),
    pytest.param(_v(1, 1, 1), math.sqrt(3), id="diagonal"),
]


class TestNorm:
    @pytest.mark.parametrize("v,expected", _NORM_CASES)
    def test_norm_method(self, v, expected):
        assert torch.allclose(v.norm(epsilon=0), torch.tensor(expected))

    @pytest.mark.parametrize("v,expected", _NORM_CASES)
    def test_norm_free(self, v, expected):
        assert torch.allclose(norm(v, epsilon=0), torch.tensor(expected))

    def test_norm2(self):
        v = _v(3, 4, 0)
        assert torch.allclose(v.norm2(), torch.tensor(25.0))

    def test_norm_epsilon_clamp(self):
        """Near-zero vector is clamped so norm >= epsilon."""
        v = _v(0, 0, 0)
        eps = 1e-6
        assert v.norm(epsilon=eps) >= eps


class TestNormalized:
    @pytest.mark.parametrize(
        "v",
        [
            pytest.param(_v(5, 0, 0), id="along-x"),
            pytest.param(_v(0, 0, -7), id="along-neg-z"),
            pytest.param(_v(1, 1, 1), id="diagonal"),
        ],
    )
    def test_unit_length(self, v):
        u = v.normalized(epsilon=0)
        assert torch.allclose(u.norm(epsilon=0), torch.tensor(1.0), atol=1e-6)

    def test_direction_preserved(self):
        v = _v(3, 0, 0)
        u = v.normalized(epsilon=0)
        assert torch.allclose(u.to_tensor(), _v(1, 0, 0).to_tensor(), atol=1e-6)

    def test_free_function(self):
        v = _v(0, 4, 0)
        u = normalized(v, epsilon=0)
        assert torch.allclose(u.to_tensor(), _v(0, 1, 0).to_tensor(), atol=1e-6)


# ===================================================================
# Distance functions
# ===================================================================

_DIST_CASES = [
    pytest.param(_v(0, 0, 0), _v(1, 0, 0), 1.0, id="unit-apart-x"),
    pytest.param(_v(0, 0, 0), _v(3, 4, 0), 5.0, id="3-4-5"),
    pytest.param(_v(1, 1, 1), _v(1, 1, 1), 0.0, id="same-point"),
]


class TestDistance:
    @pytest.mark.parametrize("a,b,expected", _DIST_CASES)
    def test_euclidean(self, a, b, expected):
        # epsilon=0 for exact match on these clean cases
        result = euclidean_distance(a, b, epsilon=0)
        assert torch.allclose(result, torch.tensor(expected), atol=1e-6)

    @pytest.mark.parametrize("a,b,expected", _DIST_CASES)
    def test_square_euclidean(self, a, b, expected):
        result = square_euclidean_distance(a, b, epsilon=0)
        assert torch.allclose(result, torch.tensor(expected**2), atol=1e-6)

    def test_symmetry(self):
        a, b = _v(1, 2, 3), _v(4, 5, 6)
        assert torch.allclose(
            euclidean_distance(a, b, epsilon=0),
            euclidean_distance(b, a, epsilon=0),
        )


# ===================================================================
# Dihedral angle
# ===================================================================

_DIHEDRAL_CASES = [
    pytest.param(
        _v(1, 1, 0),
        _v(0, 1, 0),
        _v(0, 0, 0),
        _v(1, 0, 0),
        0.0,
        id="coplanar-cis",
    ),
    pytest.param(
        _v(1, 1, 0),
        _v(0, 1, 0),
        _v(0, 0, 0),
        _v(-1, 0, 0),
        math.pi,
        id="coplanar-trans",
    ),
    pytest.param(
        _v(1, 1, 0),
        _v(0, 1, 0),
        _v(0, 0, 0),
        _v(0, 0, 1),
        math.pi / 2,
        id="perpendicular-pos",
    ),
    pytest.param(
        _v(1, 1, 0),
        _v(0, 1, 0),
        _v(0, 0, 0),
        _v(0, 0, -1),
        -math.pi / 2,
        id="perpendicular-neg",
    ),
]


class TestDihedralAngle:
    @pytest.mark.parametrize("a,b,c,d,expected_rad", _DIHEDRAL_CASES)
    def test_known_angles(self, a, b, c, d, expected_rad):
        result = dihedral_angle(a, b, c, d)
        assert torch.allclose(result, torch.tensor(expected_rad), atol=1e-5)


# ===================================================================
# Batched operations
# ===================================================================


class TestBatched:
    def test_dot_batched(self):
        a = _vb([[1, 0, 0], [0, 1, 0]])
        b = _vb([[1, 0, 0], [0, 0, 1]])
        result = a.dot(b)
        assert torch.allclose(result, torch.tensor([1.0, 0.0]))

    def test_cross_batched(self):
        a = _vb([[1, 0, 0], [0, 1, 0]])
        b = _vb([[0, 1, 0], [0, 0, 1]])
        result = a.cross(b)
        expected = _vb([[0, 0, 1], [1, 0, 0]])
        assert torch.allclose(result.to_tensor(), expected.to_tensor())

    def test_norm_batched(self):
        v = _vb([[3, 4, 0], [0, 0, 5]])
        result = v.norm(epsilon=0)
        assert torch.allclose(result, torch.tensor([5.0, 5.0]))

    def test_map_tensor_fn(self):
        v = _vb([[1, 2, 3], [4, 5, 6]])
        doubled = v.map_tensor_fn(lambda t: t * 2)
        expected = _vb([[2, 4, 6], [8, 10, 12]])
        assert torch.allclose(doubled.to_tensor(), expected.to_tensor())
