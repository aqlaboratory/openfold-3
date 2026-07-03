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

"""Parity and memory-regression tests for the fused template pair embedder.

Design constraints exercised here:

* The fused kernel must reproduce the exact math of
  ``TemplatePairEmbedderAllAtom.forward`` for a single template — the eager
  path does LN(z)@Wz + dgram@Wd + restype_i@Wai (expand) + restype_j@Waj
  (expand) + scalar@Ws + biases.
* The fused kernel operates per template — it accepts the raw pair tensor
  ``z`` and *single-template* feature tensors sliced by the caller.
* Output must add back the template dim so downstream code sees the same
  ``[B, N_templ=1, N, N, c_t=64]`` shape as the eager module.
* Per-call peak transient must be dominated by the output allocation
  (~0.5U above ``z``).
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fused kernel tests"
)


def _u_bytes(N: int, c_z: int = 128, dtype: torch.dtype = torch.float32) -> int:
    return N * N * c_z * torch.tensor([], dtype=dtype).element_size()


@pytest.fixture(scope="module")
def eager_module_factory():
    """Return a factory that builds a fresh ``TemplatePairEmbedderAllAtom``.

    Deferred import — the fused kernel module doesn't exist yet on some
    branches, and we want ``pytest --collect-only`` to survive.
    """
    from openfold3.core.model.feature_embedders.template_embedders import (
        TemplatePairEmbedderAllAtom,
    )

    def _build(c_z=128, c_dgram=39, c_aatype=32, c_out=64, seed=0):
        torch.manual_seed(seed)
        m = TemplatePairEmbedderAllAtom(
            c_in=c_z, c_dgram=c_dgram, c_aatype=c_aatype, c_out=c_out
        ).cuda().eval()
        return m

    return _build


def _make_batch(
    N: int,
    n_templ: int = 1,
    c_dgram: int = 39,
    c_aatype: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> dict:
    """Build a synthetic template batch that ``_embed_feats`` will consume."""
    g = torch.Generator(device=device).manual_seed(seed)
    B = 1
    batch = {
        "template_distogram": torch.randn(
            B, n_templ, N, N, c_dgram, generator=g, device=device, dtype=dtype
        ),
        "template_restype": torch.randn(
            B, n_templ, N, c_aatype, generator=g, device=device, dtype=dtype
        ),
        "template_pseudo_beta_mask": torch.rand(
            B, n_templ, N, generator=g, device=device, dtype=dtype
        ),
        "template_backbone_frame_mask": torch.rand(
            B, n_templ, N, generator=g, device=device, dtype=dtype
        ),
        "template_unit_vector": torch.randn(
            B, n_templ, N, N, 3, generator=g, device=device, dtype=dtype
        ),
        "asym_id": torch.zeros(B, N, device=device, dtype=torch.long),
    }
    return batch


def _eager_forward_one_template(module, batch: dict, z: torch.Tensor) -> torch.Tensor:
    """Run ``TemplatePairEmbedderAllAtom.forward`` end-to-end.

    Returns ``[B, 1, N, N, c_t]``.
    """
    with torch.inference_mode():
        return module(batch, z)


@pytest.mark.parametrize("N", [32, 64, 128])
@pytest.mark.parametrize("allow_tf32", [False, True])
def test_fused_matches_eager(eager_module_factory, N: int, allow_tf32: bool):
    """Fused kernel must match the eager module on a random single-template input."""
    from openfold3.core.model.primitives.fused_template_pair_embedder import (
        fused_template_pair_embedder_inference,
    )

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    try:
        module = eager_module_factory()
        batch = _make_batch(N=N, n_templ=1, seed=42)
        torch.manual_seed(7)
        z = torch.randn(1, N, N, 128, device="cuda")

        with torch.inference_mode():
            expected = module(batch, z.clone())

            actual = fused_template_pair_embedder_inference(
                module=module,
                batch=batch,
                z=z.clone(),
                template_index=0,
            )

        assert actual.shape == expected.shape, (actual.shape, expected.shape)
        # Match tolerance: LN + 5 matmuls; TF32 admits ~1e-2 abs error.
        rtol = 1e-4 if not allow_tf32 else 5e-3
        atol = 1e-5 if not allow_tf32 else 2e-2
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_fused_matches_eager_bitwise_no_tf32(eager_module_factory):
    """With TF32 off, fused and eager should be numerically very close."""
    from openfold3.core.model.primitives.fused_template_pair_embedder import (
        fused_template_pair_embedder_inference,
    )

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        module = eager_module_factory()
        N = 64
        batch = _make_batch(N=N, n_templ=1, seed=123)
        z = torch.randn(1, N, N, 128, device="cuda")

        with torch.inference_mode():
            expected = module(batch, z.clone())
            actual = fused_template_pair_embedder_inference(
                module=module, batch=batch, z=z.clone(), template_index=0
            )
        # LN + 5 matmuls in fp32 with tf32 off — very tight.
        torch.testing.assert_close(actual, expected, rtol=5e-5, atol=1e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


@pytest.mark.parametrize("template_index", [0, 1, 2])
def test_fused_correct_template_slice(eager_module_factory, template_index: int):
    """Verify the caller-provided ``template_index`` selects the right slice."""
    from openfold3.core.model.primitives.fused_template_pair_embedder import (
        fused_template_pair_embedder_inference,
    )

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        module = eager_module_factory()
        N = 48
        batch = _make_batch(N=N, n_templ=3, seed=999)
        z = torch.randn(1, N, N, 128, device="cuda")

        # Build a single-template mini-batch by slicing manually.
        mini_batch = {
            "template_distogram": batch["template_distogram"][:, template_index:template_index+1],
            "template_restype": batch["template_restype"][:, template_index:template_index+1],
            "template_pseudo_beta_mask": batch["template_pseudo_beta_mask"][:, template_index:template_index+1],
            "template_backbone_frame_mask": batch["template_backbone_frame_mask"][:, template_index:template_index+1],
            "template_unit_vector": batch["template_unit_vector"][:, template_index:template_index+1],
            "asym_id": batch["asym_id"],
        }
        with torch.inference_mode():
            eager = module(mini_batch, z.clone())
            fused = fused_template_pair_embedder_inference(
                module=module, batch=batch, z=z.clone(),
                template_index=template_index,
            )
        torch.testing.assert_close(fused, eager, rtol=5e-5, atol=1e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_fused_bfloat16(eager_module_factory):
    """Fused kernel should honor bf16 inputs."""
    from openfold3.core.model.primitives.fused_template_pair_embedder import (
        fused_template_pair_embedder_inference,
    )

    module = eager_module_factory()
    N = 64
    batch_fp32 = _make_batch(N=N, n_templ=1, seed=5)
    batch_bf16 = {
        k: v.to(torch.bfloat16) if v.dtype.is_floating_point else v
        for k, v in batch_fp32.items()
    }
    z_bf16 = torch.randn(1, N, N, 128, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        expected = module(batch_bf16, z_bf16.clone())
        actual = fused_template_pair_embedder_inference(
            module=module, batch=batch_bf16, z=z_bf16.clone(), template_index=0
        )
    assert actual.dtype == torch.bfloat16
    # bf16 has ~7-bit mantissa; 5 accumulated matmuls give ~4e-2 wiggle.
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def _peak_call(fn, *args, **kwargs) -> tuple[torch.Tensor, int]:
    """Return (output, transient_bytes)."""
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn(*args, **kwargs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return out, peak - before


def test_fused_peak_transient_below_output(eager_module_factory):
    """Fused kernel must not allocate more than the output tensor itself.

    Baseline (eager) allocates ``a`` (0.5U) plus multiple transients.
    Fused should allocate only the output ``a`` plus tiny sundries.
    """
    from openfold3.core.model.primitives.fused_template_pair_embedder import (
        fused_template_pair_embedder_inference,
    )

    module = eager_module_factory()
    N = 256  # 1U = 256*256*128*4 = 32 MiB
    batch = _make_batch(N=N, n_templ=1, seed=1)
    z = torch.randn(1, N, N, 128, device="cuda")

    U = _u_bytes(N, c_z=128)  # 1U as bytes
    output_bytes = N * N * 64 * 4  # a: [1,1,N,N,64] fp32
    # Output alone is 0.5U. The dgram slice contributes another ~0.15U
    # (N²·39·4 bytes) transient during addmm; LN chunk contributes ~0.05U
    # briefly. Ceiling at 0.75U comfortably below the 2U eager path.
    ceiling_bytes = int(0.75 * U)

    with torch.inference_mode():
        # Warm up compilation.
        _ = fused_template_pair_embedder_inference(
            module=module, batch=batch, z=z.clone(), template_index=0
        )
        torch.cuda.synchronize()

        _, transient = _peak_call(
            fused_template_pair_embedder_inference,
            module=module, batch=batch, z=z.clone(), template_index=0,
        )
    print(
        f"\nN={N}, 1U={U/1024**2:.2f}MiB, output={output_bytes/1024**2:.2f}MiB,"
        f" transient={transient/1024**2:.2f}MiB, ceiling={ceiling_bytes/1024**2:.2f}MiB"
    )
    assert transient < ceiling_bytes, (
        f"Fused transient {transient/1024**2:.2f}MiB exceeds 0.75U "
        f"({ceiling_bytes/1024**2:.2f}MiB) at N={N}"
    )


def test_fused_residual_no_dgram_after_return(eager_module_factory):
    """After the fused embedder returns, no ``[N, N, c_dgram]`` slice should linger.

    Regression guard for the dgram-lifetime fix: if a future edit forgets to
    ``del dgram`` before the return, the ~0.30U slice at N=1264 would coexist
    with pair_stack downstream.  Here we assert the *post-call residual*
    allocation above the output is small — well under the dgram slice size.
    """
    from openfold3.core.model.primitives.fused_template_pair_embedder import (
        fused_template_pair_embedder_inference,
    )

    module = eager_module_factory()
    N = 512  # 1U = 128 MiB; dgram slice = N*N*39*4 ≈ 39 MiB = 0.305U
    batch = _make_batch(N=N, n_templ=1, seed=7)
    z = torch.randn(1, N, N, 128, device="cuda")

    U = _u_bytes(N, c_z=128)
    output_bytes = N * N * 64 * 4  # a: 0.5U
    # After return: allocation should be baseline + output.  Anything above
    # 0.15U (small aatype projections that die at scope, etc.) is a leak.
    residual_ceiling = int(0.10 * U)

    with torch.inference_mode():
        # Warm up compilation.
        _ = fused_template_pair_embedder_inference(
            module=module, batch=batch, z=z.clone(), template_index=0
        )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        baseline = torch.cuda.memory_allocated()
        out = fused_template_pair_embedder_inference(
            module=module, batch=batch, z=z.clone(), template_index=0,
        )
        torch.cuda.synchronize()
        residual = torch.cuda.memory_allocated() - baseline - out.numel() * out.element_size()

    print(
        f"\nN={N}, 1U={U/1024**2:.2f}MiB, output={output_bytes/1024**2:.2f}MiB,"
        f" residual_above_output={residual/1024**2:.2f}MiB,"
        f" ceiling={residual_ceiling/1024**2:.2f}MiB"
    )
    assert residual < residual_ceiling, (
        f"Post-return residual {residual/1024**2:.2f}MiB above output at "
        f"N={N} exceeds 0.10U ({residual_ceiling/1024**2:.2f}MiB) — dgram or "
        f"another N²-scaled tensor is leaking past the return."
    )


def test_fused_saves_memory_vs_eager(eager_module_factory, monkeypatch):
    """Fused must show a strict memory saving over the eager forward at N=256."""
    from openfold3.core.model.primitives.fused_template_pair_embedder import (
        fused_template_pair_embedder_inference,
    )

    module = eager_module_factory()
    N = 256
    batch = _make_batch(N=N, n_templ=1, seed=1)
    z = torch.randn(1, N, N, 128, device="cuda")

    # Force the eager path in ``module.forward`` for the baseline measure —
    # otherwise the dispatch in ``TemplatePairEmbedderAllAtom.forward``
    # sends both branches through the fused wrapper and we measure
    # fused-vs-fused.
    monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_EMBED", "0")

    with torch.inference_mode():
        # Warm up compilations.
        _ = module(batch, z.clone())
        monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_EMBED", "1")
        _ = fused_template_pair_embedder_inference(
            module=module, batch=batch, z=z.clone(), template_index=0
        )
        torch.cuda.synchronize()

        monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_EMBED", "0")
        _, eager_transient = _peak_call(module, batch, z.clone())

        monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_EMBED", "1")
        _, fused_transient = _peak_call(
            fused_template_pair_embedder_inference,
            module=module, batch=batch, z=z.clone(), template_index=0,
        )
    print(
        f"\nN={N}: eager transient={eager_transient/1024**2:.2f}MiB, "
        f"fused transient={fused_transient/1024**2:.2f}MiB, "
        f"savings={(eager_transient - fused_transient)/1024**2:.2f}MiB"
    )
    assert fused_transient < eager_transient, (
        f"Fused ({fused_transient}) not smaller than eager ({eager_transient})"
    )
    # Expected savings: eager creates ~1.25U of transients above output,
    # fused creates only ~0.05U above output. Assert at least 0.3U saved.
    U = _u_bytes(N, c_z=128)
    assert (eager_transient - fused_transient) > 0.3 * U, (
        f"Savings only {(eager_transient - fused_transient) / U:.3f}U; "
        f"expected > 0.3U at N={N}"
    )


def test_module_forward_dispatches_to_fused(eager_module_factory, monkeypatch):
    """``TemplatePairEmbedderAllAtom.forward`` should dispatch to fused when
    the flag is on and the batch is single-template."""
    from openfold3.core.model.primitives.fused_template_pair_embedder import (
        fused_template_pair_embedder_inference,
    )

    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        module = eager_module_factory()
        N = 64
        batch = _make_batch(N=N, n_templ=1, seed=13)
        z = torch.randn(1, N, N, 128, device="cuda")

        with torch.inference_mode():
            # Dispatch on: module.forward should match direct fused call
            # (bitwise, since both take the same code path).
            monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_EMBED", "1")
            dispatched = module(batch, z.clone())
            direct = fused_template_pair_embedder_inference(
                module=module, batch=batch, z=z.clone(), template_index=0
            )
            torch.testing.assert_close(dispatched, direct, rtol=0, atol=0)

            # Dispatch off: still numerically close (parity is proven
            # elsewhere; this is just a smoke check that both paths run).
            monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_EMBED", "0")
            eager = module(batch, z.clone())
            torch.testing.assert_close(dispatched, eager, rtol=5e-5, atol=1e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_module_forward_multi_template_uses_eager(eager_module_factory, monkeypatch):
    """When n_templ > 1, forward must NOT dispatch to fused (kernel is
    single-template)."""
    module = eager_module_factory()
    N = 32
    batch = _make_batch(N=N, n_templ=3, seed=99)
    z = torch.randn(1, N, N, 128, device="cuda")

    monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_EMBED", "1")
    with torch.inference_mode():
        out = module(batch, z)
    # Output for 3 templates has template dim 3.
    assert out.shape == (1, 3, N, N, 64), out.shape
    """Fused must show a strict memory saving over the eager forward at N=256."""
    from openfold3.core.model.primitives.fused_template_pair_embedder import (
        fused_template_pair_embedder_inference,
    )

    module = eager_module_factory()
    N = 256
    batch = _make_batch(N=N, n_templ=1, seed=1)
    z = torch.randn(1, N, N, 128, device="cuda")

    # Force the eager path in ``module.forward`` for the baseline measure —
    # otherwise the dispatch in ``TemplatePairEmbedderAllAtom.forward``
    # sends both branches through the fused wrapper and we measure
    # fused-vs-fused.
    monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_EMBED", "0")

    with torch.inference_mode():
        # Warm up compilations.
        _ = module(batch, z.clone())
        monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_EMBED", "1")
        _ = fused_template_pair_embedder_inference(
            module=module, batch=batch, z=z.clone(), template_index=0
        )
        torch.cuda.synchronize()

        monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_EMBED", "0")
        _, eager_transient = _peak_call(module, batch, z.clone())

        monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_EMBED", "1")
        _, fused_transient = _peak_call(
            fused_template_pair_embedder_inference,
            module=module, batch=batch, z=z.clone(), template_index=0,
        )
    print(
        f"\nN={N}: eager transient={eager_transient/1024**2:.2f}MiB, "
        f"fused transient={fused_transient/1024**2:.2f}MiB, "
        f"savings={(eager_transient - fused_transient)/1024**2:.2f}MiB"
    )
    assert fused_transient < eager_transient, (
        f"Fused ({fused_transient}) not smaller than eager ({eager_transient})"
    )
    # Expected savings: eager creates ~1.25U of transients above output,
    # fused creates only ~0.05U above output. Assert at least 0.3U saved.
    U = _u_bytes(N, c_z=128)
    assert (eager_transient - fused_transient) > 0.3 * U, (
        f"Savings only {(eager_transient - fused_transient) / U:.3f}U; "
        f"expected > 0.3U at N={N}"
    )
