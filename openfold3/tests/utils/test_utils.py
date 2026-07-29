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
from openfold3.core.utils.chunk_utils import ChunkSizeTuner, _chunk_slice, chunk_layer
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

    def test_chunk_size_tuner_caches(self):
        tuner = ChunkSizeTuner()

        def fn(t, chunk_size):
            if chunk_size > 2 ** t.dim() * t.dtype.itemsize:
                raise RuntimeError("Chunk size too large")
            return t

        spy_fn = unittest.mock.Mock(side_effect=fn)

        first = tuner.tune_chunk_size(
            representative_fn=spy_fn,
            args=(torch.randn(2, 3, 4, 5),),
            max_chunk_size=256,
        )

        first_call_count = spy_fn.call_count
        second = tuner.tune_chunk_size(
            representative_fn=spy_fn,
            args=(torch.randn(2, 3, 4, 5),),
            max_chunk_size=256,
        )

        self.assertEqual(
            first,
            second,
            "Chunk size should have been cached for identical arg shapes and dtypes",
        )
        self.assertEqual(
            first_call_count,
            spy_fn.call_count,
            "Representative function should not have been called again for identical arg shapes and dtypes",
        )

    def test_chunk_size_tuner_does_not_retest_candidates(self):
        # Based on previous bug: the binary search forgot which candidates it
        # had already proven non-viable and re-tested them.
        for max_viable in (128, 64, 256, 512):
            with self.subTest(max_viable=max_viable):
                tested = []

                def fn(arg, chunk_size, _max=max_viable, tested=tested):
                    tested.append(chunk_size)
                    if chunk_size > _max:
                        raise RuntimeError("simulated OOM")

                ChunkSizeTuner._determine_favorable_chunk_size(
                    fn, args=(None,), max_chunk_size=1024
                )

                self.assertEqual(
                    len(tested),
                    len(set(tested)),
                    f"Some candidate was tested more than once: {tested}",
                )

    def test_chunk_size_tuner_picks_largest_viable(self):
        # When the cutoff sits between two power-of-2 candidates, the tuner
        # should pick the largest viable power of 2 at or below the cutoff.
        cases = [
            # (max_viable, expected_chunk_size)
            (1024, 1024),
            (512, 512),
            (511, 256),
            (256, 256),
            (255, 128),
            (128, 128),
            (4, 4),
            (3, 2),
            (1, 1),
        ]
        for max_viable, expected in cases:
            with self.subTest(max_viable=max_viable):

                def fn(arg, chunk_size, _max=max_viable):
                    if chunk_size > _max:
                        raise RuntimeError("simulated OOM")

                result = ChunkSizeTuner._determine_favorable_chunk_size(
                    fn, args=(None,), max_chunk_size=1024
                )
                self.assertEqual(result, expected)

    def test_chunk_size_tuner_non_power_of_two_max_fits(self):
        # When max_chunk_size isn't a power of 2, it should still be tried as
        # a candidate (and returned when viable).
        def fits_all(arg, chunk_size):
            return None

        self.assertEqual(
            ChunkSizeTuner._determine_favorable_chunk_size(
                fits_all, args=(None,), max_chunk_size=500
            ),
            500,
        )

    def test_chunk_size_tuner_non_power_of_two_max_does_not_fit(self):
        # And when only powers of 2 below the max are viable, fall back to the
        # largest such power of 2.
        def fits_up_to_256(arg, chunk_size):
            if chunk_size > 256:
                raise RuntimeError("simulated OOM")

        self.assertEqual(
            ChunkSizeTuner._determine_favorable_chunk_size(
                fits_up_to_256, args=(None,), max_chunk_size=500
            ),
            256,
        )

    def test_chunk_size_tuner_caps_at_max_chunk_size(self):
        # max_chunk_size is the config-level ceiling: even when much larger
        # values would fit, the tuner must not exceed it.
        for max_chunk_size in (4, 16, 128, 512, 1024):
            with self.subTest(max_chunk_size=max_chunk_size):

                def fn(arg, chunk_size):
                    return None  # never raises -- any chunk_size "fits"

                result = ChunkSizeTuner._determine_favorable_chunk_size(
                    fn, args=(None,), max_chunk_size=max_chunk_size
                )
                self.assertEqual(result, max_chunk_size)

    def test_chunk_size_tuner_retunes_different_shape(self):
        # Different arg shapes should invalidate the cache and trigger
        # re-tuning.
        tuner = ChunkSizeTuner()

        def fn(t, chunk_size):
            if chunk_size > t.shape[-1]:
                raise RuntimeError("simulated OOM")

        first = tuner.tune_chunk_size(
            representative_fn=fn,
            args=(torch.zeros(2, 3, 16),),
            max_chunk_size=256,
        )
        second = tuner.tune_chunk_size(
            representative_fn=fn,
            args=(torch.zeros(2, 3, 128),),
            max_chunk_size=256,
        )

        self.assertNotEqual(
            first,
            second,
            "Chunk size should have been re-tuned for new arg shape",
        )

    def test_chunk_size_tuner_retunes_different_rank(self):
        tuner = ChunkSizeTuner()

        def fn(t, chunk_size):
            if chunk_size > 2 ** t.dim() * t.dtype.itemsize:
                raise RuntimeError("Chunk size too large")
            return t

        first = tuner.tune_chunk_size(
            representative_fn=fn,
            args=(torch.zeros(2, 3, 4, 5),),
            max_chunk_size=256,
        )
        second = tuner.tune_chunk_size(
            representative_fn=fn,
            args=(torch.zeros(2, 3, 4, 5, 6),),
            max_chunk_size=256,
        )

        self.assertNotEqual(
            first, second, "Chunk size should have been re-tuned for new arg rank"
        )

    def test_chunk_size_tuner_retunes_different_dtype_bytes(self):
        tuner = ChunkSizeTuner()

        def fn(t, chunk_size):
            if chunk_size > 2 ** t.dim() * t.dtype.itemsize:
                raise RuntimeError("Chunk size too large")
            return t

        first = tuner.tune_chunk_size(
            representative_fn=fn,
            args=(torch.zeros(2, 3, 4, 5, dtype=torch.float32),),
            max_chunk_size=256,
        )
        second = tuner.tune_chunk_size(
            representative_fn=fn,
            args=(torch.zeros(2, 3, 4, 5, dtype=torch.bfloat16),),
            max_chunk_size=256,
        )

        self.assertNotEqual(
            first, second, "Chunk size should have been re-tuned for new dtype bytes"
        )

    def test_chunk_size_tuner_retunes_different_arg_count(self):
        tuner = ChunkSizeTuner()

        def fn(*args, chunk_size):
            if chunk_size > 2 ** len(args):
                raise RuntimeError("Chunk size too large")
            return args

        first = tuner.tune_chunk_size(
            representative_fn=fn,
            args=(1, 2, 3, 4, 5),
            max_chunk_size=256,
        )
        second = tuner.tune_chunk_size(
            representative_fn=fn,
            args=(1, 2, 3, 4, 5, 6),
            max_chunk_size=256,
        )

        self.assertNotEqual(
            first, second, "Chunk size should have been re-tuned for new arg count"
        )
