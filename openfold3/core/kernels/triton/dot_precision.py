# Copyright 2026 AlQuraishi Laboratory
# Copyright 2026 Advanced Micro Devices, Inc.
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

"""Honors torch's fp32 matmul precision setting in the Triton kernels.

``torch.backends.cuda.matmul.fp32_precision`` has values ``"ieee"`` or
``"tf32"``, but there are other reduced precision dots as well (which Torch
itself uses in its "high" precision mode), so we use the supported mode with at
least the required precision.

Left unset, ``tl.dot`` picks tf32 where available (which on AMD is only gfx942)
and doesn't use other reduced-precision modes (e.g. bf16x3) or respect the Torch
precision setting. So the kernels call ``tl_dot`` instead and pass the mode from
``current_dot_precision``. Ruff bans ``tl.dot`` everywhere else (TID251).
"""

import torch
import triton
import triton.language as tl

IEEE = "ieee"
TF32 = "tf32"

# Triton dot modes for each setting, fastest first. Every chain ends at "ieee",
# which all backends support, so resolution should always succeed.
_MODES = {
    TF32: ("tf32", "bf16x3", "ieee"),
    IEEE: ("ieee",),
}

_ALLOWED_BY_TARGET: dict[object, tuple[str, ...]] = {}


def torch_fp32_precision() -> str:
    """torch's current fp32 matmul precision, as ``IEEE`` or ``TF32``.

    If Torch reports that TF32 is not supported it returns ``IEEE`` immediately.
    Triton will report bf16x3 as available on any target, but it's only faster
    on a few. Torch counts "has a faster bf16x3 path" as support for TF32.

    Uses ``matmul.fp32_precision`` where available. The older
    ``get_float32_matmul_precision()`` or ``matmul.allow_tf32`` raise if a
    process has mixed torch's precision APIs whereas this works no matter how
    the precision was set.
    """
    if not torch.cuda.is_tf32_supported():
        return IEEE
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        return TF32 if matmul.fp32_precision == TF32 else IEEE
    return TF32 if matmul.allow_tf32 else IEEE


def resolve(precision: str, allowed: tuple[str, ...]) -> str:
    """Triton dot mode for ``precision`` on a target supporting ``allowed``."""
    if precision not in _MODES:
        raise ValueError(
            f"unknown precision {precision!r}, expected one of {tuple(_MODES)}"
        )
    for mode in _MODES[precision]:
        if mode in allowed:
            return mode
    # This should never happen because all targets should support ieee and we
    # should always bottom out at that.
    raise RuntimeError(
        f"target supports none of {_MODES[precision]}; it allows only {allowed}"
    )


def _allowed_dot_precisions() -> tuple[str, ...]:
    """Dot input precisions the active Triton target supports, cached per target."""
    target = triton.runtime.driver.active.get_current_target()
    if target not in _ALLOWED_BY_TARGET:
        options = triton.compiler.make_backend(target).parse_options({})
        _ALLOWED_BY_TARGET[target] = tuple(options.allowed_dot_input_precisions)
    return _ALLOWED_BY_TARGET[target]


def current_dot_precision(*, fp32_operands: bool) -> str:
    """``input_precision`` to pass to ``tl_dot`` under torch's current setting.

    Triton ignores the mode unless the operands are fp32, so
    ``fp32_operands=False`` pins to ieee to avoid compiling a second,
    identical kernel.
    """
    if not fp32_operands:
        return IEEE
    return resolve(torch_fp32_precision(), _allowed_dot_precisions())


@triton.jit
def tl_dot(a, b, input_precision: tl.constexpr, acc=None):
    """``tl.dot`` but with ``input_precision`` as a required argument.

    This exists so we can ban ``tl.dot`` calls without an explicit
    ``input_precision`` using a lint rule. Note that because ``triton.jit``
    doesn't support required keyword-only arguments (``, *, ...``) this swaps
    the order of ``input_precision`` and ``acc``. Both should be passed as
    keyword arguments.
    """
    return tl.dot(a, b, acc=acc, input_precision=input_precision)  # noqa: TID251
