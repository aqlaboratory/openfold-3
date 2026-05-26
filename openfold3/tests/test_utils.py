# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import unittest

import torch

from openfold3.core.model.primitives import Linear
from openfold3.core.utils.chunk_utils import _chunk_slice, _plan_chunks, chunk_layer
from openfold3.core.utils.rigid_utils import (
    Rigid,
    Rotation,
    quat_to_rot,
    rot_to_quat,
)

X_90_ROT = torch.tensor(
    [
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]
)

X_NEG_90_ROT = torch.tensor(
    [
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0],
    ]
)


class TestUtils(unittest.TestCase):
    def test_rigid_from_3_points_shape(self):
        batch_size = 2
        n_res = 5

        x1 = torch.rand((batch_size, n_res, 3))
        x2 = torch.rand((batch_size, n_res, 3))
        x3 = torch.rand((batch_size, n_res, 3))

        r = Rigid.from_3_points(x1, x2, x3)

        rot, tra = r.get_rots().get_rot_mats(), r.get_trans()

        self.assertTrue(rot.shape == (batch_size, n_res, 3, 3))
        self.assertTrue(torch.all(tra == x2))

    def test_rigid_from_4x4(self):
        batch_size = 2
        transf = [
            [1, 0, 0, 1],
            [0, 0, -1, 2],
            [0, 1, 0, 3],
            [0, 0, 0, 1],
        ]
        transf = torch.tensor(transf)

        true_rot = transf[:3, :3]
        true_trans = transf[:3, 3]

        transf = torch.stack([transf for _ in range(batch_size)], dim=0)

        r = Rigid.from_tensor_4x4(transf)

        rot, tra = r.get_rots().get_rot_mats(), r.get_trans()

        self.assertTrue(torch.all(rot == true_rot.unsqueeze(0)))
        self.assertTrue(torch.all(tra == true_trans.unsqueeze(0)))

    def test_rigid_shape(self):
        batch_size = 2
        n = 5
        transf = Rigid(
            Rotation(rot_mats=torch.rand((batch_size, n, 3, 3))),
            torch.rand((batch_size, n, 3)),
        )

        self.assertTrue(transf.shape == (batch_size, n))

    def test_rigid_cat(self):
        batch_size = 2
        n = 5
        transf = Rigid(
            Rotation(rot_mats=torch.rand((batch_size, n, 3, 3))),
            torch.rand((batch_size, n, 3)),
        )

        transf_cat = Rigid.cat([transf, transf], dim=0)

        transf_rots = transf.get_rots().get_rot_mats()
        transf_cat_rots = transf_cat.get_rots().get_rot_mats()

        self.assertTrue(transf_cat_rots.shape == (batch_size * 2, n, 3, 3))

        transf_cat = Rigid.cat([transf, transf], dim=1)
        transf_cat_rots = transf_cat.get_rots().get_rot_mats()

        self.assertTrue(transf_cat_rots.shape == (batch_size, n * 2, 3, 3))

        self.assertTrue(torch.all(transf_cat_rots[:, :n] == transf_rots))
        self.assertTrue(torch.all(transf_cat.get_trans()[:, :n] == transf.get_trans()))

    def test_rigid_compose(self):
        trans_1 = [0, 1, 0]
        trans_2 = [0, 0, 1]

        t1 = Rigid(Rotation(rot_mats=X_90_ROT), torch.tensor(trans_1))
        t2 = Rigid(Rotation(rot_mats=X_NEG_90_ROT), torch.tensor(trans_2))

        t3 = t1.compose(t2)

        self.assertTrue(torch.all(t3.get_rots().get_rot_mats() == torch.eye(3)))
        self.assertTrue(torch.all(t3.get_trans() == 0))

    def test_rigid_apply(self):
        rots = torch.stack([X_90_ROT, X_NEG_90_ROT], dim=0)
        trans = torch.tensor([1, 1, 1])
        trans = torch.stack([trans, trans], dim=0)

        t = Rigid(Rotation(rot_mats=rots), trans)

        x = torch.arange(30)
        x = torch.stack([x, x], dim=0)
        x = x.view(2, -1, 3)  # [2, 10, 3]

        pts = t[..., None].apply(x)

        # All simple consequences of the two x-axis rotations
        self.assertTrue(torch.all(pts[..., 0] == x[..., 0] + 1))
        self.assertTrue(torch.all(pts[0, :, 1] == x[0, :, 2] * -1 + 1))
        self.assertTrue(torch.all(pts[1, :, 1] == x[1, :, 2] + 1))
        self.assertTrue(torch.all(pts[0, :, 2] == x[0, :, 1] + 1))
        self.assertTrue(torch.all(pts[1, :, 2] == x[1, :, 1] * -1 + 1))

    def test_quat_to_rot(self):
        forty_five = math.pi / 4
        quat = torch.tensor([math.cos(forty_five), math.sin(forty_five), 0, 0])
        rot = quat_to_rot(quat)
        eps = 1e-07
        self.assertTrue(torch.all(torch.abs(rot - X_90_ROT) < eps))

    def test_rot_to_quat(self):
        quat = rot_to_quat(X_90_ROT)
        eps = 1e-07
        ans = torch.tensor([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])
        self.assertTrue(torch.all(torch.abs(quat - ans) < eps))

    def test_chunk_layer_tensor(self):
        x = torch.rand(2, 4, 5, 15)
        l = Linear(15, 30)
        chunked = chunk_layer(l, {"input": x}, chunk_size=4, no_batch_dims=3)
        unchunked = l(x)

        self.assertTrue(torch.all(chunked == unchunked))

    def test_chunk_layer_dict(self):
        class LinearDictLayer(Linear):
            def forward(self, input):
                out = super().forward(input)
                return {"out": out, "inner": {"out": out + 1}}

        x = torch.rand(2, 4, 5, 15)
        l = LinearDictLayer(15, 30)

        chunked = chunk_layer(l, {"input": x}, chunk_size=4, no_batch_dims=3)
        unchunked = l(x)

        self.assertTrue(torch.all(chunked["out"] == unchunked["out"]))
        self.assertTrue(torch.all(chunked["inner"]["out"] == unchunked["inner"]["out"]))

    def test_chunk_slice_dict(self):
        x = torch.rand(3, 4, 3, 5)
        x_flat = x.view(-1, 5)

        prod = 1
        for d in x.shape[:-1]:
            prod = prod * d

        for i in range(prod):
            for j in range(i + 1, prod + 1):
                chunked = _chunk_slice(x, i, j, len(x.shape[:-1]))
                chunked_flattened = x_flat[i:j]

                self.assertTrue(torch.all(chunked == chunked_flattened))

    def _collect_plan(self, batch_dims, chunk_size):
        return list(_plan_chunks(batch_dims, chunk_size))

    def _slices_size(self, slices):
        prod = 1
        for s in slices:
            prod *= s.stop - s.start
        return prod

    def test_plan_chunks_inner_fits_one_outer_per_chunk(self):
        # (5, 76) cs=128: inner 76 fits, only 1 outer per chunk (152 > 128)
        plan = self._collect_plan((5, 76), 128)
        self.assertEqual(len(plan), 5)
        for i, slices in enumerate(plan):
            self.assertEqual(slices, (slice(i, i + 1), slice(0, 76)))

    def test_plan_chunks_inner_split(self):
        # (5, 76) cs=64: inner 76 > 64, split into [64, 12]; one outer at a time
        plan = self._collect_plan((5, 76), 64)
        self.assertEqual(len(plan), 10)
        # First sample: [0:64], [64:76]
        self.assertEqual(plan[0], (slice(0, 1), slice(0, 64)))
        self.assertEqual(plan[1], (slice(0, 1), slice(64, 76)))
        # Last sample
        self.assertEqual(plan[-2], (slice(4, 5), slice(0, 64)))
        self.assertEqual(plan[-1], (slice(4, 5), slice(64, 76)))

    def test_plan_chunks_pack_outer(self):
        # (5, 76) cs=256: inner 76 fits, 256//76 = 3 outer per chunk
        plan = self._collect_plan((5, 76), 256)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0], (slice(0, 3), slice(0, 76)))
        self.assertEqual(plan[1], (slice(3, 5), slice(0, 76)))

    def test_plan_chunks_covers_index_space_exactly(self):
        # For varied (batch_dims, chunk_size), ensure the plan tiles the full index space
        # and each chunk's volume is <= chunk_size.
        for batch_dims in [(5, 76), (2, 100), (3,), (4, 8, 5), (1, 50)]:
            for cs in [16, 32, 64, 128, 256]:
                with self.subTest(batch_dims=batch_dims, cs=cs):
                    plan = self._collect_plan(batch_dims, cs)
                    # Volume per chunk
                    for slices in plan:
                        self.assertLessEqual(self._slices_size(slices), cs)
                    # Total covered = product(batch_dims)
                    total = sum(self._slices_size(s) for s in plan)
                    expected = 1
                    for d in batch_dims:
                        expected *= d
                    self.assertEqual(total, expected)

    def test_plan_chunks_chunk_size_larger_than_total(self):
        # cs >= product(batch_dims): single chunk covering everything
        plan = self._collect_plan((3, 4), 100)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0], (slice(0, 3), slice(0, 4)))

    def test_plan_chunks_single_dim(self):
        plan = self._collect_plan((10,), 4)
        self.assertEqual(plan, [
            (slice(0, 4),), (slice(4, 8),), (slice(8, 10),)
        ])

    def test_chunk_layer_rejects_incompatible_batch_dims(self):
        # When two inputs have different sizes in the same batch dim and
        # neither is 1, broadcasting is undefined; chunk_layer should error
        # rather than silently producing wrong results.
        a = torch.rand(5, 76, 4)
        b = torch.rand(3, 76, 4)
        with self.assertRaises(ValueError):
            chunk_layer(
                lambda a, b: a + b,
                {"a": a, "b": b},
                chunk_size=32,
                no_batch_dims=2,
            )

    def test_chunk_layer_broadcast(self):
        # B1*B2 isn't evenly divisble by any power of 2, so we can test the
        # uneven chunk tail behavior.
        B1, B2, N, C = 5, 76, 76, 4
        a = torch.rand(B1, B2, N, C)
        b = torch.rand(B1, 1, N, C)

        def add_layer(a, b):
            return a + b

        unchunked = add_layer(a, b)

        for cs in [32, 64, 128, 256]:
            with self.subTest(chunk_size=cs):
                chunked = chunk_layer(
                    add_layer,
                    {"a": a, "b": b},
                    chunk_size=cs,
                    no_batch_dims=2,
                )
                self.assertEqual(chunked.shape, unchunked.shape)
                self.assertTrue(torch.allclose(chunked, unchunked))

    def test_chunk_layer_broadcast_does_not_materialize(self):
        B1, B2, N, C = 5, 76, 76, 4
        a = torch.rand(B1, B2, N, C)
        b = torch.rand(B1, 1, N, C)

        seen_b_shapes = []

        def spy(a, b):
            seen_b_shapes.append(tuple(b.shape))
            return a + b

        chunk_layer(
            spy,
            {"a": a, "b": b},
            chunk_size=128,
            no_batch_dims=2,
        )
        # Every call's b must keep the broadcast `1` in its second dim.
        self.assertGreater(len(seen_b_shapes), 0)
        for shape in seen_b_shapes:
            self.assertEqual(
                shape[1], 1,
                f"b was materialized along the broadcast dim: {shape}",
            )
