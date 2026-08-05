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
#
# Python wrappers modified from PyTorch3D's ball-query Python layer
# (https://github.com/facebookresearch/pytorch3d, Meta Platforms, Inc.,
#  BSD-3-Clause; see
#  https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE)
# by Liang Hong <lhong22@cse.cuhk.edu.hk>: JIT loader plus thin wrappers
# for the cooperative, with-pred, and backward CUDA entry points used by
# the smooth lDDT loss.

"""JIT loader for the smooth lDDT CUDA ball-query extension."""

import importlib.util
import os
import shutil
from pathlib import Path

import torch

EXT_NAME = "openfold3_smooth_lddt_ball_query"
EXT_DIR = Path(__file__).parent / "ball_query_ext"
_EXT_SOURCES = ("binding.cpp", "ball_query.cu")
VERBOSE = bool(int(os.environ.get("OPENFOLD3_SMOOTH_LDDT_VERBOSE", "0")))

INSTALL_HINT = (
    "smooth lDDT ball-query extension is not installed. "
    "Use a pixi CUDA environment (openfold3-cuda12 or openfold3-cuda13), "
    "or install ninja and a matching CUDA toolchain when building from "
    "source, and ensure the CUDA sources under "
    "openfold3/core/kernels/smooth_lddt/ball_query_ext are present."
)

_ext = None


def _ninja_available() -> bool:
    """torch.utils.cpp_extension shells out to the ``ninja`` binary; the pypi
    ``ninja`` wheel provides both binary and Python module, conda ships only
    the binary. Accept either."""
    if shutil.which("ninja") is not None:
        return True
    return importlib.util.find_spec("ninja") is not None


def is_ball_query_installed() -> bool:
    """Check whether the ball-query extension can be built.

    Verifies ninja is available (as a CLI binary or Python module) and the
    CUDA source files ship with the installed package. Does NOT require CUDA
    to be available at runtime, so this is safe to call at import time on
    CPU-only or pypi builds.
    """
    if not _ninja_available():
        return False
    if not EXT_DIR.is_dir():
        return False
    return all((EXT_DIR / src).is_file() for src in _EXT_SOURCES)


def is_ball_query_available() -> bool:
    """Return whether the ball-query extension is installed AND CUDA is usable."""
    return is_ball_query_installed() and torch.cuda.is_available()


def _set_cuda_arch_list() -> None:
    if "TORCH_CUDA_ARCH_LIST" in os.environ or not torch.cuda.is_available():
        return

    try:
        cap = torch.cuda.get_device_capability()
    except Exception:
        return

    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"


def _load():
    global _ext
    if _ext is None:
        if not is_ball_query_installed():
            raise RuntimeError(INSTALL_HINT)
        if not torch.cuda.is_available():
            raise RuntimeError("smooth lDDT ball-query extension requires CUDA")

        # Deferred to keep top-level import cheap and avoid importing
        # torch.utils.cpp_extension on pypi builds that never use the kernel.
        from torch.utils.cpp_extension import load as load_ext

        _set_cuda_arch_list()
        sources = [str(EXT_DIR / src) for src in _EXT_SOURCES]
        _ext = load_ext(
            name=EXT_NAME,
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=VERBOSE,
        )

    return _ext


def ball_query(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: torch.Tensor,
    lengths2: torch.Tensor,
    k: int,
    radius: float,
    skip_points_outside_cube: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CUDA ball-query neighbor indices and squared distances."""
    if not (p1.is_cuda and p2.is_cuda):
        raise RuntimeError("smooth lDDT ball-query extension requires CUDA tensors")

    ext = _load()
    idx, dists = ext.ball_query(
        p1.float(),
        p2.float(),
        lengths1.long(),
        lengths2.long(),
        int(k),
        float(radius),
        bool(skip_points_outside_cube),
    )
    return idx, dists


def ball_query_coop(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: torch.Tensor,
    lengths2: torch.Tensor,
    k: int,
    radius: float,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cooperative ball query with reservoir sampling for unbiased random K selection.

    Uses W=8 threads per query atom for parallel scanning and reservoir sampling
    to select a random subset when more than K neighbors qualify.
    K must be divisible by 8.
    """
    if not (p1.is_cuda and p2.is_cuda):
        raise RuntimeError("smooth lDDT ball-query extension requires CUDA tensors")

    ext = _load()
    idx, dists = ext.ball_query_coop(
        p1.float(),
        p2.float(),
        lengths1.long(),
        lengths2.long(),
        int(k),
        float(radius),
        int(seed),
    )
    return idx, dists


def ball_query_coop_with_pred(
    p1: torch.Tensor,
    p2: torch.Tensor,
    pred: torch.Tensor,
    lengths1: torch.Tensor,
    lengths2: torch.Tensor,
    k: int,
    radius: float,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cooperative ball query that also outputs predicted squared distances.

    GT positions ``p1``/``p2`` are processed in fp32 for neighbor finding;
    predicted positions ``pred`` are read in their native dtype (bf16/fp32),
    distances computed in fp32 internally and stored in ``pred``'s dtype.

    Returns ``(idx, dists_gt, dists_pred)`` of shapes ``[N, P1, K]`` —
    ``idx`` is int64, ``dists_gt`` is fp32, ``dists_pred`` matches ``pred``'s dtype.
    K must be divisible by 8.
    """
    if not (p1.is_cuda and p2.is_cuda and pred.is_cuda):
        raise RuntimeError("smooth lDDT ball-query extension requires CUDA tensors")

    ext = _load()
    idx, dists_gt, dists_pred = ext.ball_query_coop_with_pred(
        p1.float(),
        p2.float(),
        pred.contiguous(),
        lengths1.long(),
        lengths2.long(),
        int(k),
        float(radius),
        int(seed),
    )
    return idx, dists_gt, dists_pred


def ball_query_pred_backward(
    pred: torch.Tensor,
    idxs: torch.Tensor,
    grad_dists: torch.Tensor,
) -> torch.Tensor:
    """Backward pass for predicted squared distances.

    For each (n, i, k): accumulates ``2 * grad_dists[n,i,k] * (pred[n,i] - pred[n,j])``
    into ``grad_pred[n,i]`` and the negation into ``grad_pred[n,j]`` (atomicAdd).

    Returns ``grad_pred [N,P1,3]`` in fp32 (caller may cast back to ``pred.dtype``).
    Nondeterministic due to atomicAdd, matching PyTorch3D's KNN backward.
    """
    if not pred.is_cuda:
        raise RuntimeError("smooth lDDT ball-query extension requires CUDA tensors")

    ext = _load()
    return ext.ball_query_pred_backward(
        pred.contiguous(),
        idxs.long().contiguous(),
        grad_dists.contiguous(),
    )
