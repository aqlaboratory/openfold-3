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

"""Tests for MsaArray class."""

import numpy as np
import pandas as pd
import pytest

from openfold3.core.data.primitives.sequence.msa import MsaArray


def _make_msa_array(
    msa: list[str],
    deletion_matrix: list[list[int]] | None = None,
    metadata: pd.DataFrame | list | np.ndarray | None = None,
) -> MsaArray:
    """Helper to build an MsaArray from readable string lists."""
    msa_np = np.array([list(seq) for seq in msa])
    if deletion_matrix is None:
        deletion_matrix = np.zeros(msa_np.shape, dtype=int)
    else:
        deletion_matrix = np.array(deletion_matrix)
    if metadata is None:
        metadata = pd.DataFrame()
    return MsaArray(msa=msa_np, deletion_matrix=deletion_matrix, metadata=metadata)


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


class TestLen:
    @pytest.mark.parametrize(
        "msa_rows,expected",
        [
            pytest.param(["ACG"], 1, id="single_sequence"),
            pytest.param(["ACG", "A-G", "ACC"], 3, id="three_sequences"),
        ],
    )
    def test_len(self, msa_rows, expected):
        arr = _make_msa_array(msa_rows)
        assert len(arr) == expected


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    @pytest.mark.parametrize(
        "row_slice,inplace,expected_nrows",
        [
            pytest.param(2, False, 2, id="int_not_inplace"),
            pytest.param(2, True, 2, id="int_inplace"),
            pytest.param(slice(1, 3), False, 3, id="slice_stop_clamped"),
            pytest.param(5, False, 3, id="int_exceeds_length"),
        ],
    )
    def test_truncate_shapes(self, row_slice, inplace, expected_nrows):
        arr = _make_msa_array(["ACG", "A-G", "ACC"])
        result = arr.truncate(row_slice, inplace=inplace)
        if inplace:
            assert result is None
            assert arr.msa.shape[0] == expected_nrows
        else:
            assert result.msa.shape[0] == expected_nrows
            # original unchanged
            assert arr.msa.shape[0] == 3

    def test_truncate_preserves_data(self):
        arr = _make_msa_array(
            ["ACG", "A-G", "ACC"],
            deletion_matrix=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        )
        result = arr.truncate(2, inplace=False)
        np.testing.assert_array_equal(result.msa, np.array([list("ACG"), list("A-G")]))
        np.testing.assert_array_equal(
            result.deletion_matrix, np.array([[0, 0, 0], [1, 0, 0]])
        )

    def test_truncate_invalid_type(self):
        arr = _make_msa_array(["ACG"])
        with pytest.raises(ValueError, match="integer or a slice"):
            arr.truncate("bad")

    def test_truncate_with_dataframe_metadata(self):
        meta = pd.DataFrame({"species": ["A", "B", "C"]})
        arr = _make_msa_array(["ACG", "A-G", "ACC"], metadata=meta)
        result = arr.truncate(2, inplace=False)
        assert len(result.metadata) == 2

    def test_truncate_with_list_metadata(self):
        arr = _make_msa_array(["ACG", "A-G", "ACC"], metadata=["s1", "s2", "s3"])
        result = arr.truncate(2, inplace=False)
        assert len(result.metadata) == 2


# ---------------------------------------------------------------------------
# concatenate
# ---------------------------------------------------------------------------


class TestConcatenate:
    @pytest.mark.parametrize(
        "axis,inplace",
        [
            pytest.param(0, False, id="axis0_not_inplace"),
            pytest.param(0, True, id="axis0_inplace"),
            pytest.param(1, False, id="axis1_not_inplace"),
            pytest.param(1, True, id="axis1_inplace"),
        ],
    )
    def test_concatenate_shape(self, axis, inplace):
        a = _make_msa_array(["ACG", "A-G"])
        b = (
            _make_msa_array(["ACG", "ACC"])
            if axis == 0
            else _make_msa_array(["TT", "GG"])
        )
        result = a.concatenate(b, axis=axis, inplace=inplace)
        if inplace:
            assert result is None
            target = a
        else:
            target = result
        if axis == 0:
            assert target.msa.shape == (4, 3)
        else:
            assert target.msa.shape == (2, 5)

    def test_concatenate_axis0_preserves_list_metadata(self):
        a = _make_msa_array(["ACG"], metadata=np.array(["s1"]))
        b = _make_msa_array(["A-G"], metadata=np.array(["s2"]))
        result = a.concatenate(b, axis=0, inplace=False)
        np.testing.assert_array_equal(result.metadata, np.array(["s1", "s2"]))

    def test_concatenate_axis1_drops_metadata(self):
        a = _make_msa_array(["AC"], metadata=np.array(["s1"]))
        b = _make_msa_array(["GT"], metadata=np.array(["s2"]))
        result = a.concatenate(b, axis=1, inplace=False)
        assert isinstance(result.metadata, pd.DataFrame)
        assert result.metadata.empty

    def test_concatenate_axis0_mismatched_cols(self):
        a = _make_msa_array(["ACG"])
        b = _make_msa_array(["AC"])
        with pytest.raises(ValueError, match="number of columns must match"):
            a.concatenate(b, axis=0)

    def test_concatenate_axis1_mismatched_rows(self):
        a = _make_msa_array(["AC", "GT"])
        b = _make_msa_array(["AC"])
        with pytest.raises(ValueError, match="number of rows must match"):
            a.concatenate(b, axis=1)

    def test_concatenate_invalid_axis(self):
        a = _make_msa_array(["ACG"])
        b = _make_msa_array(["ACG"])
        with pytest.raises(ValueError, match="Axis must be 0"):
            a.concatenate(b, axis=2)


# ---------------------------------------------------------------------------
# multi_concatenate
# ---------------------------------------------------------------------------


class TestMultiConcatenate:
    @pytest.mark.parametrize(
        "n_arrays,axis,expected_shape",
        [
            pytest.param(1, 0, (1, 3), id="single_array"),
            pytest.param(3, 0, (3, 3), id="three_arrays_axis0"),
            pytest.param(2, 1, (1, 6), id="two_arrays_axis1"),
        ],
    )
    def test_multi_concatenate_shape(self, n_arrays, axis, expected_shape):
        arrays = [_make_msa_array(["ACG"]) for _ in range(n_arrays)]
        result = MsaArray.multi_concatenate(arrays, axis=axis)
        assert result.msa.shape == expected_shape

    def test_multi_concatenate_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            MsaArray.multi_concatenate([])

    def test_multi_concatenate_invalid_axis(self):
        with pytest.raises(ValueError, match="Axis must be 0"):
            MsaArray.multi_concatenate([_make_msa_array(["ACG"])], axis=2)

    def test_multi_concatenate_axis0_mismatched_cols(self):
        a = _make_msa_array(["ACG"])
        b = _make_msa_array(["AC"])
        with pytest.raises(ValueError, match="same number of columns"):
            MsaArray.multi_concatenate([a, b], axis=0)

    def test_multi_concatenate_axis1_mismatched_rows(self):
        a = _make_msa_array(["AC", "GT"])
        b = _make_msa_array(["AC"])
        with pytest.raises(ValueError, match="same number of rows"):
            MsaArray.multi_concatenate([a, b], axis=1)

    def test_multi_concatenate_axis0_preserves_dataframe_metadata(self):
        a = _make_msa_array(["ACG"], metadata=pd.DataFrame({"s": ["x"]}))
        b = _make_msa_array(["A-G"], metadata=pd.DataFrame({"s": ["y"]}))
        result = MsaArray.multi_concatenate([a, b], axis=0)
        assert isinstance(result.metadata, pd.DataFrame)
        assert len(result.metadata) == 2

    def test_multi_concatenate_axis0_preserves_array_metadata(self):
        a = _make_msa_array(["ACG"], metadata=np.array(["s1"]))
        b = _make_msa_array(["A-G"], metadata=np.array(["s2"]))
        result = MsaArray.multi_concatenate([a, b], axis=0)
        np.testing.assert_array_equal(result.metadata, np.array(["s1", "s2"]))

    def test_multi_concatenate_axis1_drops_metadata(self):
        a = _make_msa_array(["AC"])
        b = _make_msa_array(["GT"])
        result = MsaArray.multi_concatenate([a, b], axis=1)
        assert isinstance(result.metadata, pd.DataFrame)
        assert result.metadata.empty


# ---------------------------------------------------------------------------
# pad
# ---------------------------------------------------------------------------


class TestPad:
    @pytest.mark.parametrize(
        "axis,target,inplace,return_mask",
        [
            pytest.param(0, 5, False, True, id="axis0_not_inplace_with_mask"),
            pytest.param(0, 5, True, True, id="axis0_inplace_with_mask"),
            pytest.param(1, 6, False, True, id="axis1_not_inplace_with_mask"),
            pytest.param(0, 5, False, False, id="axis0_not_inplace_no_mask"),
            pytest.param(0, 5, True, False, id="axis0_inplace_no_mask"),
        ],
    )
    def test_pad_shapes(self, axis, target, inplace, return_mask):
        arr = _make_msa_array(["ACG", "A-G"])  # shape (2, 3)
        result = arr.pad(target, axis=axis, return_mask=return_mask, inplace=inplace)
        if inplace:
            if return_mask:
                ret, mask = result
                assert ret is None
                assert arr.msa.shape[axis] == target
                assert mask.shape == arr.msa.shape
            else:
                assert result is None
                assert arr.msa.shape[axis] == target
        else:
            if return_mask:
                padded, mask = result
                assert padded.msa.shape[axis] == target
                assert mask.shape == padded.msa.shape
            else:
                assert result.msa.shape[axis] == target

    def test_pad_no_op_when_already_target_size(self):
        arr = _make_msa_array(["ACG", "A-G"])
        padded, mask = arr.pad(3, axis=1, return_mask=True, inplace=False)
        np.testing.assert_array_equal(padded.msa, arr.msa)
        np.testing.assert_array_equal(mask, np.ones(arr.msa.shape, dtype=int))

    def test_pad_no_op_inplace(self):
        arr = _make_msa_array(["ACG", "A-G"])
        ret, mask = arr.pad(3, axis=1, return_mask=True, inplace=True)
        assert ret is None
        np.testing.assert_array_equal(mask, np.ones((2, 3), dtype=int))

    def test_pad_values_and_mask(self):
        arr = _make_msa_array(["AC", "GT"])
        padded, mask = arr.pad(
            4, axis=1, pad_value="-", return_mask=True, inplace=False
        )
        # original columns are ones, padded columns are zeros
        expected_mask = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
        np.testing.assert_array_equal(mask, expected_mask)
        # padded values
        np.testing.assert_array_equal(
            padded.msa[:, 2:], np.array([["-", "-"], ["-", "-"]])
        )
        # deletion matrix padded with zeros
        np.testing.assert_array_equal(padded.deletion_matrix[:, 2:], np.zeros((2, 2)))

    def test_pad_negative_raises(self):
        arr = _make_msa_array(["ACG"])
        with pytest.raises(ValueError, match="cannot be padded to a smaller size"):
            arr.pad(1, axis=1)


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_to_dict_keys(self):
        arr = _make_msa_array(["ACG"], metadata=["seq1"])
        d = arr.to_dict()
        assert set(d.keys()) == {"msa", "deletion_matrix", "metadata"}

    def test_to_dict_roundtrip(self):
        arr = _make_msa_array(["ACG", "A-G"])
        d = arr.to_dict()
        np.testing.assert_array_equal(d["msa"], arr.msa)
        np.testing.assert_array_equal(d["deletion_matrix"], arr.deletion_matrix)


# ---------------------------------------------------------------------------
# subset
# ---------------------------------------------------------------------------


class TestSubset:
    @pytest.mark.parametrize(
        "mask,expected_nrows,metadata_type",
        [
            pytest.param(
                np.array([True, False, True]),
                2,
                "dataframe",
                id="dataframe_metadata",
            ),
            pytest.param(
                np.array([True, False, True]),
                2,
                "ndarray",
                id="ndarray_metadata",
            ),
            pytest.param(
                np.array([True, False, True]),
                2,
                "list",
                id="list_metadata",
            ),
            pytest.param(
                np.array([False, False, False]),
                0,
                "dataframe",
                id="all_false",
            ),
            pytest.param(
                np.array([True, True, True]),
                3,
                "dataframe",
                id="all_true",
            ),
        ],
    )
    def test_subset(self, mask, expected_nrows, metadata_type):
        if metadata_type == "dataframe":
            metadata = pd.DataFrame({"species": ["A", "B", "C"]})
        elif metadata_type == "ndarray":
            metadata = np.array(["A", "B", "C"])
        else:
            metadata = ["A", "B", "C"]

        arr = _make_msa_array(["ACG", "A-G", "ACC"], metadata=metadata)
        result = arr.subset(mask)
        assert result.msa.shape[0] == expected_nrows
        assert result.deletion_matrix.shape[0] == expected_nrows

    def test_subset_preserves_data(self):
        arr = _make_msa_array(
            ["ACG", "A-G", "ACC"],
            deletion_matrix=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            metadata=np.array(["s1", "s2", "s3"]),
        )
        mask = np.array([True, False, True])
        result = arr.subset(mask)
        np.testing.assert_array_equal(result.msa, np.array([list("ACG"), list("ACC")]))
        np.testing.assert_array_equal(
            result.deletion_matrix, np.array([[0, 0, 0], [0, 1, 0]])
        )
