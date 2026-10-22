"""GPU gates must also mark tests ``gpu``.

CPU CI skips every GPU-gated test and the GPU CI job selects ``-m "slow or gpu"``,
so a gated test without the marker runs nowhere.
"""

import pytest

from openfold3.tests.utils import compare_utils

GPU_GATES = [
    compare_utils.skip_unless_gpu_available,
    compare_utils.skip_unless_accelerator_available,
    compare_utils.skip_unless_ds4s_installed,
    compare_utils.skip_unless_evo_attn_available,
    compare_utils.skip_unless_cueq_installed,
]


def _mark_names(obj) -> list[str]:
    return [m.name for m in getattr(obj, "pytestmark", [])]


@pytest.mark.parametrize("gate", GPU_GATES, ids=lambda g: g.__name__)
def test_gpu_gate_marks_function_gpu_and_skipif(gate):
    @gate()
    def test_fn():
        pass

    assert "gpu" in _mark_names(test_fn)
    assert "skipif" in _mark_names(test_fn)


@pytest.mark.parametrize("gate", GPU_GATES, ids=lambda g: g.__name__)
def test_gpu_gate_marks_class_gpu_and_skipif(gate):
    @gate()
    class TestCls:
        pass

    assert "gpu" in _mark_names(TestCls)
    assert "skipif" in _mark_names(TestCls)


@pytest.mark.parametrize(
    "gate",
    [compare_utils.skip_unless_triton_installed, compare_utils.skip_if_rocm],
    ids=lambda g: g.__name__,
)
def test_non_gpu_gate_does_not_mark_gpu(gate):
    @gate()
    def test_fn():
        pass

    assert "gpu" not in _mark_names(test_fn)


def test_device_fixture_marks_only_cuda_gpu(request, device):
    has_gpu_mark = request.node.get_closest_marker("gpu") is not None
    assert has_gpu_mark == (device == "cuda")
