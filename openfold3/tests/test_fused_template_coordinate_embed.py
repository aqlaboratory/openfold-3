# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import copy
import os
import subprocess
import sys
import tempfile

import pytest
import torch

from openfold3.core.data.primitives.featurization.template import (
    create_template_distogram,
    create_template_unit_vector,
)
from openfold3.core.model.feature_embedders.template_embedders import (
    TemplatePairEmbedderAllAtom,
)
from openfold3.core.model.primitives.fused_template_coordinate import (
    fused_template_coordinate_pair_embedder,
    template_coordinate_projection_add_reference_,
    template_coordinate_projection_reference,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    TemplateSettings,
)


def _coordinates(n_token: int, n_templ: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(n_token, dtype=torch.float32)
    templates = []
    frames = []
    for t in range(n_templ):
        offset = 0.4 * t
        ca = torch.stack(
            (
                index * 3.8,
                torch.sin(index * 0.3 + offset),
                torch.cos(index * 0.2 + offset),
            ),
            dim=-1,
        )
        pseudo_beta = ca + torch.tensor([0.2, -0.3, 0.7])
        n_atom = ca + torch.tensor([-0.7, 1.1, 0.2])
        c_atom = ca + torch.tensor([1.3, 0.2, -0.1])
        templates.append(pseudo_beta)
        frames.append(torch.stack((n_atom, ca, c_atom), dim=-2))
    return torch.stack(templates, dim=0), torch.stack(frames, dim=0)


def _coordinate_batch(n_token: int, device: str = "cpu", n_templ: int = 1) -> dict:
    pseudo_beta, frame = _coordinates(n_token, n_templ=n_templ)
    restype = torch.nn.functional.one_hot(
        torch.arange(n_token) % 32, num_classes=32
    ).to(torch.int32)
    batch = {
        "template_pseudo_beta_coords": pseudo_beta[None].to(device),
        "template_frame_atom_coords": frame[None].to(device),
        "template_pseudo_beta_mask": torch.ones(1, n_templ, n_token, device=device),
        "template_backbone_frame_mask": torch.ones(1, n_templ, n_token, device=device),
        "template_restype": restype[None, None]
        .expand(1, n_templ, -1, -1)
        .contiguous()
        .to(device),
        "asym_id": torch.cat(
            (
                torch.ones(n_token // 2, dtype=torch.int32),
                torch.full((n_token - n_token // 2,), 2, dtype=torch.int32),
            )
        )[None].to(device),
    }
    if n_token > 3:
        batch["template_pseudo_beta_mask"][:, :, 1] = 0
        batch["template_backbone_frame_mask"][:, :, 2] = 0
    return batch


def _precomputed_batch(coordinate_batch: dict) -> dict:
    pseudo_beta = coordinate_batch["template_pseudo_beta_coords"][0].cpu().numpy()
    frame = coordinate_batch["template_frame_atom_coords"][0].cpu().numpy()
    pb_mask = coordinate_batch["template_pseudo_beta_mask"][0].cpu()
    bb_mask = coordinate_batch["template_backbone_frame_mask"][0].cpu()
    asym = coordinate_batch["asym_id"][0].cpu()
    # create_template_* expects [n_templ, N, ...] coords and a broadcastable
    # multichain mask over the template axis.
    multichain = (asym[:, None] == asym[None, :])[None, ..., None]
    return {
        "template_distogram": create_template_distogram(
            pseudo_beta, pb_mask, multichain
        )[None],
        "template_unit_vector": create_template_unit_vector(frame, bb_mask, multichain)[
            None
        ],
        "template_pseudo_beta_mask": coordinate_batch[
            "template_pseudo_beta_mask"
        ].cpu(),
        "template_backbone_frame_mask": coordinate_batch[
            "template_backbone_frame_mask"
        ].cpu(),
        "template_restype": coordinate_batch["template_restype"].cpu(),
        "asym_id": coordinate_batch["asym_id"].cpu(),
    }


@pytest.mark.parametrize("n_token", [8, 17, 31])
def test_coordinate_reference_matches_precomputed_embedder(n_token: int):
    torch.manual_seed(7)
    module = TemplatePairEmbedderAllAtom(128, 39, 32, 64).eval()
    coordinate_batch = _coordinate_batch(n_token)
    precomputed_batch = _precomputed_batch(coordinate_batch)
    z = torch.randn(1, n_token, n_token, 128)

    with torch.inference_mode():
        expected = module(precomputed_batch, z.clone())
        actual = fused_template_coordinate_pair_embedder(
            module=module,
            batch=coordinate_batch,
            z=z.clone(),
            template_index=0,
        )

    assert not {
        "template_distogram",
        "template_unit_vector",
    }.intersection(coordinate_batch)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("n_token,n_templ", [(8, 2), (17, 3)])
def test_template_embedder_coordinate_matches_precomputed(n_token: int, n_templ: int):
    """Full TemplateEmbedderAllAtom path: coords vs precomputed distogram/UV."""
    from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    torch.manual_seed(11)
    of3_config = OF3ProjectEntry().get_model_config_with_presets()
    embedder = TemplateEmbedderAllAtom(of3_config.architecture.template).eval()
    c_z = of3_config.architecture.template.template_pair_embedder.c_in

    coordinate_batch = _coordinate_batch(n_token, n_templ=n_templ)
    precomputed_batch = _precomputed_batch(coordinate_batch)
    # Restype for the pair embedder is float one-hot in the streaming tests.
    coordinate_batch["template_restype"] = coordinate_batch["template_restype"].float()
    precomputed_batch["template_restype"] = precomputed_batch[
        "template_restype"
    ].float()

    z = torch.randn(1, n_token, n_token, c_z)
    pair_mask = torch.ones(1, n_token, n_token)

    with torch.inference_mode():
        expected = embedder(
            batch=precomputed_batch,
            z=z.clone(),
            pair_mask=pair_mask,
            chunk_size=4,
            inplace_safe=True,
        )
        actual = embedder(
            batch=coordinate_batch,
            z=z.clone(),
            pair_mask=pair_mask,
            chunk_size=4,
            inplace_safe=True,
        )

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("checkpoint_mode", ["blocks", "per_template"])
def test_coordinate_training_checkpoint_matches_uncheckpointed(checkpoint_mode: str):
    from openfold3.core.model.latent.template_module import TemplateEmbedderAllAtom
    from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

    torch.manual_seed(23)
    of3_config = OF3ProjectEntry().get_model_config_with_presets()
    reference_config = copy.deepcopy(of3_config.architecture.template)
    reference_config.template_pair_stack.blocks_per_ckpt = None
    reference_config.template_pair_stack.ckpt_per_template = False
    checkpoint_config = copy.deepcopy(reference_config)
    if checkpoint_mode == "blocks":
        checkpoint_config.template_pair_stack.blocks_per_ckpt = 1
    else:
        checkpoint_config.template_pair_stack.ckpt_per_template = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference_module = TemplateEmbedderAllAtom(reference_config).to(device).train()
    with torch.no_grad():
        reference_module.linear_t.weight.normal_(std=0.05)
    checkpoint_module = TemplateEmbedderAllAtom(checkpoint_config).to(device).train()
    checkpoint_module.load_state_dict(reference_module.state_dict())

    n_token = 4
    batch = _coordinate_batch(n_token, device=device, n_templ=2)
    batch["template_restype"] = batch["template_restype"].float()
    c_z = reference_config.template_pair_embedder.c_in
    z_reference = torch.randn(
        1, n_token, n_token, c_z, device=device, requires_grad=True
    )
    z_checkpoint = z_reference.detach().clone().requires_grad_(True)
    pair_mask = torch.ones(1, n_token, n_token, device=device)
    upstream = torch.randn(1, n_token, n_token, c_z, device=device)

    torch.manual_seed(29)
    reference = reference_module(batch, z_reference, pair_mask)
    (reference * upstream).sum().backward()
    torch.manual_seed(29)
    checkpointed = checkpoint_module(batch, z_checkpoint, pair_mask)
    (checkpointed * upstream).sum().backward()

    tolerance = 5e-4 if device == "cuda" else 1e-5
    torch.testing.assert_close(checkpointed, reference, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        z_checkpoint.grad, z_reference.grad, rtol=tolerance, atol=tolerance
    )
    reference_parameters = dict(reference_module.named_parameters())
    for name, parameter in checkpoint_module.named_parameters():
        expected_grad = reference_parameters[name].grad
        if parameter.grad is None or expected_grad is None:
            assert parameter.grad is expected_grad, name
            continue
        torch.testing.assert_close(
            parameter.grad,
            expected_grad,
            rtol=tolerance,
            atol=tolerance,
            msg=lambda msg, name=name: f"{name}: {msg}",
        )


def test_coordinate_embedding_training_eager_batch_and_gradient_accumulation():
    n_token = 6
    batch = _coordinate_batch(n_token)
    batch = {
        key: value.expand(2, *value.shape[1:]).clone() for key, value in batch.items()
    }
    batch["template_restype"] = batch["template_restype"].float()
    module = TemplatePairEmbedderAllAtom(128, 39, 32, 64).train()
    z = torch.randn(2, n_token, n_token, 128, requires_grad=True)

    actual = fused_template_coordinate_pair_embedder(
        module=module,
        batch=batch,
        z=z,
        template_index=0,
    )
    assert actual.shape == (2, 1, n_token, n_token, 64)
    actual.square().mean().backward()
    first_z_grad = z.grad.clone()
    first_weight_grad = module.dgram_linear.weight.grad.clone()

    actual = fused_template_coordinate_pair_embedder(
        module=module,
        batch=batch,
        z=z,
        template_index=0,
    )
    actual.square().mean().backward()
    torch.testing.assert_close(z.grad, first_z_grad * 2)
    torch.testing.assert_close(module.dgram_linear.weight.grad, first_weight_grad * 2)


def test_coordinate_embedding_rejects_coordinate_gradients():
    n_token = 6
    batch = _coordinate_batch(n_token)
    batch["template_pseudo_beta_coords"].requires_grad_(True)
    module = TemplatePairEmbedderAllAtom(128, 39, 32, 64).train()
    z = torch.randn(1, n_token, n_token, 128, requires_grad=True)

    with pytest.raises(ValueError, match="non-differentiable data"):
        fused_template_coordinate_pair_embedder(module, batch, z, 0)


def test_coordinate_embedding_rejects_fused_away_bias():
    n_token = 6
    batch = _coordinate_batch(n_token)
    module = TemplatePairEmbedderAllAtom(128, 39, 32, 64).train()
    module.dgram_linear.bias = torch.nn.Parameter(torch.zeros(64))
    z = torch.randn(1, n_token, n_token, 128, requires_grad=True)

    with pytest.raises(ValueError, match="bias-free"):
        fused_template_coordinate_pair_embedder(module, batch, z, 0)


def test_coordinate_features_reject_nondefault_distogram_settings():
    with pytest.raises(ValueError, match="default"):
        TemplateSettings(
            use_coordinate_pair_features=True,
            distogram={"min_bin": 2.0, "max_bin": 50.75, "n_bins": 39},
        )


def test_coordinate_features_registered_with_quality_control():
    from openfold3.core.data.primitives.quality_control.asserts import (
        FEATURE_OTHER_DTYPES,
        FULL_OTHER_DIM_INDEX_MAP,
        FULL_TEMPLATE_DIM_INDEX_MAP,
        FULL_TOKEN_DIM_INDEX_MAP,
        _assert_template_feature_representation,
    )

    coordinate_features = {
        "template_pseudo_beta_coords": torch.zeros(2, 5, 3),
        "template_frame_atom_coords": torch.zeros(2, 5, 3, 3),
    }
    _assert_template_feature_representation(coordinate_features)
    assert FULL_TOKEN_DIM_INDEX_MAP["template_pseudo_beta_coords"] == [-2]
    assert FULL_TOKEN_DIM_INDEX_MAP["template_frame_atom_coords"] == [-3]
    assert FULL_TEMPLATE_DIM_INDEX_MAP["template_pseudo_beta_coords"] == [-3]
    assert FULL_TEMPLATE_DIM_INDEX_MAP["template_frame_atom_coords"] == [-4]
    assert FULL_OTHER_DIM_INDEX_MAP["template_frame_atom_coords"] == [-1, -2]
    assert FEATURE_OTHER_DTYPES["template_pseudo_beta_coords"] == torch.float32
    assert FEATURE_OTHER_DTYPES["template_frame_atom_coords"] == torch.float32

    with pytest.raises(AssertionError, match="Exactly one"):
        _assert_template_feature_representation(
            {
                **coordinate_features,
                "template_distogram": torch.zeros(2, 5, 5, 39),
                "template_unit_vector": torch.zeros(2, 5, 5, 3),
            }
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coordinate_triton_training_gradients_match_eager(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    n_token = 17
    torch.manual_seed(19)
    reference_module = TemplatePairEmbedderAllAtom(128, 39, 32, 64).cuda().train()
    actual_module = copy.deepcopy(reference_module)
    batch = _coordinate_batch(n_token, device="cuda")
    batch["template_restype"] = batch["template_restype"].float()
    z_reference = torch.randn(
        1, n_token, n_token, 128, device="cuda", requires_grad=True
    )
    z_actual = z_reference.detach().clone().requires_grad_(True)
    upstream = torch.randn(1, 1, n_token, n_token, 64, device="cuda")

    monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_COORD", "0")
    reference = fused_template_coordinate_pair_embedder(
        reference_module, batch, z_reference, 0
    )
    (reference * upstream).sum().backward()

    monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_COORD", "1")
    actual = fused_template_coordinate_pair_embedder(actual_module, batch, z_actual, 0)
    assert (
        "_TemplateCoordinateProjectionFunctionBackward"
        in type(actual.grad_fn.next_functions[0][0]).__name__
    )
    (actual * upstream).sum().backward()

    torch.testing.assert_close(actual, reference, rtol=2e-4, atol=3e-4)
    torch.testing.assert_close(z_actual.grad, z_reference.grad, rtol=5e-4, atol=5e-4)
    reference_parameters = dict(reference_module.named_parameters())
    for name, parameter in actual_module.named_parameters():
        assert parameter.grad is not None, name
        expected_grad = reference_parameters[name].grad
        assert expected_grad is not None, name
        torch.testing.assert_close(
            parameter.grad,
            expected_grad,
            rtol=5e-4,
            atol=5e-4,
            msg=lambda msg, name=name: f"{name}: {msg}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_token", [8, 31, 65, 384])
@pytest.mark.parametrize(
    ("compute_dgram", "compute_scalar"),
    [(True, True), (True, False), (False, True)],
)
def test_coordinate_triton_backward_matches_eager(
    n_token: int,
    compute_dgram: bool,
    compute_scalar: bool,
    monkeypatch,
):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    from openfold3.core.kernels.triton.fused_template_coordinate import (
        template_coordinate_projection_backward,
    )

    torch.manual_seed(41 + n_token)
    batch = _coordinate_batch(n_token, device="cuda")
    pseudo_beta = batch["template_pseudo_beta_coords"][:, 0].contiguous()
    frame = batch["template_frame_atom_coords"][:, 0].contiguous()
    pb_mask = batch["template_pseudo_beta_mask"][:, 0].contiguous()
    bb_mask = batch["template_backbone_frame_mask"][:, 0].contiguous()
    pb_mask[:, ::5] = 0
    bb_mask[:, 2::7] = 0
    asym = batch["asym_id"].contiguous()
    asym[:, n_token // 2 :] = 2
    upstream = torch.randn(1, n_token, n_token, 64, device="cuda")
    if n_token == 31:
        upstream = upstream.transpose(1, 2)
    dgram_weight = torch.randn(64, 39, device="cuda", requires_grad=compute_dgram)
    scalar_weight = torch.randn(64, 5, device="cuda", requires_grad=compute_scalar)
    source = torch.zeros_like(upstream)

    expected = template_coordinate_projection_reference(
        source,
        pseudo_beta,
        frame,
        pb_mask,
        bb_mask,
        asym,
        dgram_weight,
        scalar_weight,
        chunk_rows=7,
    )
    (expected * upstream).sum().backward()
    actual_dgram, actual_scalar = template_coordinate_projection_backward(
        upstream,
        pseudo_beta,
        frame,
        pb_mask,
        bb_mask,
        asym,
        compute_dgram=compute_dgram,
        compute_scalar=compute_scalar,
    )

    if compute_dgram:
        assert dgram_weight.grad is not None
        torch.testing.assert_close(
            actual_dgram, dgram_weight.grad, rtol=5e-4, atol=5e-4
        )
    else:
        assert actual_dgram is None
    if compute_scalar:
        assert scalar_weight.grad is not None
        torch.testing.assert_close(
            actual_scalar, scalar_weight.grad, rtol=5e-4, atol=5e-4
        )
    else:
        assert actual_scalar is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coordinate_triton_backward_tf32_matches_eager(monkeypatch):
    from openfold3.core.kernels.triton.fused_template_coordinate import (
        template_coordinate_projection_backward,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    n_token = 384
    torch.manual_seed(71)
    batch = _coordinate_batch(n_token, device="cuda")
    pseudo_beta = batch["template_pseudo_beta_coords"][:, 0].contiguous()
    frame = batch["template_frame_atom_coords"][:, 0].contiguous()
    pb_mask = batch["template_pseudo_beta_mask"][:, 0].contiguous()
    bb_mask = batch["template_backbone_frame_mask"][:, 0].contiguous()
    asym = batch["asym_id"].contiguous()
    upstream = torch.randn(1, n_token, n_token, 64, device="cuda")
    dgram_weight = torch.randn(64, 39, device="cuda", requires_grad=True)
    scalar_weight = torch.randn(64, 5, device="cuda", requires_grad=True)
    expected = template_coordinate_projection_reference(
        torch.zeros_like(upstream),
        pseudo_beta,
        frame,
        pb_mask,
        bb_mask,
        asym,
        dgram_weight,
        scalar_weight,
        chunk_rows=32,
    )
    (expected * upstream).sum().backward()
    actual = template_coordinate_projection_backward(
        upstream,
        pseudo_beta,
        frame,
        pb_mask,
        bb_mask,
        asym,
    )

    for actual_grad, expected_grad in zip(
        actual, (dgram_weight.grad, scalar_weight.grad), strict=True
    ):
        relative_l2 = (actual_grad - expected_grad).norm() / expected_grad.norm()
        relative_max = (
            actual_grad - expected_grad
        ).abs().max() / expected_grad.abs().max()
        assert relative_l2 < 1e-3
        assert relative_max < 1.5e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coordinate_triton_backward_is_repeatable():
    from openfold3.core.kernels.triton.fused_template_coordinate import (
        template_coordinate_projection_backward,
    )

    n_token = 73
    torch.manual_seed(67)
    batch = _coordinate_batch(n_token, device="cuda")
    args = (
        torch.randn(1, n_token, n_token, 64, device="cuda"),
        batch["template_pseudo_beta_coords"][:, 0].contiguous(),
        batch["template_frame_atom_coords"][:, 0].contiguous(),
        batch["template_pseudo_beta_mask"][:, 0].contiguous(),
        batch["template_backbone_frame_mask"][:, 0].contiguous(),
        batch["asym_id"].contiguous(),
    )
    first = template_coordinate_projection_backward(*args)
    second = template_coordinate_projection_backward(*args)
    for actual, expected in zip(first, second, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coordinate_triton_backward_preserves_open_bin_boundaries():
    from openfold3.core.kernels.triton.fused_template_coordinate import (
        template_coordinate_projection_backward,
    )

    distances = torch.tensor([0.0, 3.25, 4.0, 4.5, 50.75, 60.0], device="cuda")
    n_token = len(distances)
    pseudo_beta = torch.zeros(1, n_token, 3, device="cuda")
    pseudo_beta[0, :, 0] = distances
    _, frame_cpu = _coordinates(n_token)
    frame = frame_cpu.cuda()
    mask = torch.ones(1, n_token, device="cuda")
    asym = torch.ones(1, n_token, dtype=torch.int32, device="cuda")
    upstream = torch.zeros(1, n_token, n_token, 64, device="cuda")
    upstream[0, 0, :, 0] = 1

    grad_dgram, grad_scalar = template_coordinate_projection_backward(
        upstream,
        pseudo_beta,
        frame,
        mask,
        mask,
        asym,
        compute_scalar=False,
    )
    assert grad_scalar is None
    expected = torch.zeros(64, 39, device="cuda")
    expected[0, 0] = 1
    expected[0, 38] = 1
    torch.testing.assert_close(grad_dgram, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coordinate_bfloat16_autocast_uses_triton(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_FUSED_TEMPLATE_COORD", "1")
    n_token = 8
    module = TemplatePairEmbedderAllAtom(128, 39, 32, 64).cuda().train()
    batch = _coordinate_batch(n_token, device="cuda")
    z = torch.randn(1, n_token, n_token, 128, device="cuda", requires_grad=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        actual = fused_template_coordinate_pair_embedder(module, batch, z, 0)
        assert actual.dtype == torch.bfloat16
        assert (
            "_TemplateCoordinateProjectionFunctionBackward"
            in type(actual.grad_fn.next_functions[0][0]).__name__
        )
        actual.float().square().mean().backward()

    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert module.dgram_linear.weight.grad is not None
    assert torch.isfinite(module.dgram_linear.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coordinate_triton_bf16_matches_fp32_eager_oracle(monkeypatch):
    """bf16 Triton fp32-accumulate vs fp32 eager rounded through bf16."""
    from openfold3.core.model.primitives.fused_template_coordinate import (
        _TemplateCoordinateProjectionFunction,
        template_coordinate_projection_reference,
    )

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    n_token = 17
    torch.manual_seed(37)
    batch = _coordinate_batch(n_token, device="cuda")
    pb = batch["template_pseudo_beta_coords"][:, 0].contiguous()
    frame = batch["template_frame_atom_coords"][:, 0].contiguous()
    pb_mask = batch["template_pseudo_beta_mask"][:, 0].contiguous()
    bb_mask = batch["template_backbone_frame_mask"][:, 0].contiguous()
    asym = batch["asym_id"].contiguous()
    dgram_weight = torch.randn(64, 39, device="cuda")
    scalar_weight = torch.randn(64, 5, device="cuda")
    source_bf16 = torch.randn(
        1, n_token, n_token, 64, device="cuda", dtype=torch.bfloat16
    )
    upstream = torch.randn(1, n_token, n_token, 64, device="cuda")

    dgram_ref = dgram_weight.detach().clone().requires_grad_(True)
    scalar_ref = scalar_weight.detach().clone().requires_grad_(True)
    source_ref = source_bf16.float().clone().requires_grad_(True)
    # Cast through bf16 so both paths see the same quantized output grad.
    reference = template_coordinate_projection_reference(
        source_ref, pb, frame, pb_mask, bb_mask, asym, dgram_ref, scalar_ref
    ).to(torch.bfloat16)
    (reference.float() * upstream).sum().backward()

    dgram_act = dgram_weight.detach().clone().requires_grad_(True)
    scalar_act = scalar_weight.detach().clone().requires_grad_(True)
    source_act = source_bf16.detach().clone().requires_grad_(True)
    actual = _TemplateCoordinateProjectionFunction.apply(
        source_act, pb, frame, pb_mask, bb_mask, asym, dgram_act, scalar_act
    )
    (actual.float() * upstream).sum().backward()

    torch.testing.assert_close(actual, reference, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(
        source_act.grad, source_ref.grad.to(torch.bfloat16), rtol=5e-4, atol=5e-4
    )
    torch.testing.assert_close(dgram_act.grad, dgram_ref.grad, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(scalar_act.grad, scalar_ref.grad, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_token", [8, 31, 64, 96])
def test_coordinate_triton_matches_chunked_reference(n_token: int):
    from openfold3.core.kernels.triton.fused_template_coordinate import (
        template_coordinate_projection,
        template_coordinate_projection_add_,
    )

    batch = _coordinate_batch(n_token, device="cuda")
    pseudo_beta = batch["template_pseudo_beta_coords"][:, 0].contiguous()
    frame = batch["template_frame_atom_coords"][:, 0].contiguous()
    pb_mask = batch["template_pseudo_beta_mask"][:, 0].contiguous()
    bb_mask = batch["template_backbone_frame_mask"][:, 0].contiguous()
    asym = batch["asym_id"].contiguous()
    torch.manual_seed(12)
    dgram_weight = torch.randn(64, 39, device="cuda")
    scalar_weight = torch.randn(64, 5, device="cuda")
    source = torch.randn(1, n_token, n_token, 64, device="cuda")
    source_before = source.clone()
    expected = source.clone()

    with torch.no_grad():
        template_coordinate_projection_add_reference_(
            expected,
            pseudo_beta,
            frame,
            pb_mask,
            bb_mask,
            asym,
            dgram_weight,
            scalar_weight,
            chunk_rows=7,
        )
        actual_inplace = source.clone()
        template_coordinate_projection_add_(
            actual_inplace,
            pseudo_beta,
            frame,
            pb_mask,
            bb_mask,
            asym,
            dgram_weight,
            scalar_weight,
        )
        actual_fresh = template_coordinate_projection(
            source,
            pseudo_beta,
            frame,
            pb_mask,
            bb_mask,
            asym,
            dgram_weight,
            scalar_weight,
        )
    torch.cuda.synchronize()

    torch.testing.assert_close(source, source_before, rtol=0, atol=0)
    torch.testing.assert_close(actual_inplace, expected, rtol=2e-4, atol=3e-4)
    torch.testing.assert_close(actual_fresh, expected, rtol=2e-4, atol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coordinate_inplace_launcher_rejects_grad_mode():
    from openfold3.core.kernels.triton.fused_template_coordinate import (
        template_coordinate_projection,
        template_coordinate_projection_add_,
    )

    n_token = 4
    batch = _coordinate_batch(n_token, device="cuda")
    source = torch.zeros(1, n_token, n_token, 64, device="cuda")
    args = (
        batch["template_pseudo_beta_coords"][:, 0],
        batch["template_frame_atom_coords"][:, 0],
        batch["template_pseudo_beta_mask"][:, 0],
        batch["template_backbone_frame_mask"][:, 0],
        batch["asym_id"],
        torch.randn(64, 39, device="cuda"),
        torch.randn(64, 5, device="cuda"),
    )
    with pytest.raises(RuntimeError, match="disabled grad mode"):
        template_coordinate_projection_add_(source, *args)
    with pytest.raises(RuntimeError, match="not autograd-aware"):
        template_coordinate_projection(
            source, *args, out=torch.empty_like(source, requires_grad=True)
        )
    overlapping_out = torch.empty(1, n_token, n_token, 1, device="cuda").expand_as(
        source
    )
    with torch.no_grad(), pytest.raises(ValueError, match="overlap internally"):
        template_coordinate_projection(source, *args, out=overlapping_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_coordinate_kernel_preserves_open_bin_boundaries():
    from openfold3.core.kernels.triton.fused_template_coordinate import (
        template_coordinate_projection_add_,
    )

    distances = torch.tensor([0.0, 3.25, 4.0, 4.5, 50.75, 60.0], device="cuda")
    n_token = len(distances)
    pseudo_beta = torch.zeros(1, n_token, 3, device="cuda")
    pseudo_beta[0, :, 0] = distances
    _, frame_cpu = _coordinates(n_token)
    frame = frame_cpu.cuda()
    mask = torch.ones(1, n_token, device="cuda")
    asym = torch.ones(1, n_token, dtype=torch.int32, device="cuda")
    dgram_weight = (
        torch.arange(1, 40, device="cuda", dtype=torch.float32)[None]
        .expand(64, -1)
        .contiguous()
    )
    scalar_weight = torch.zeros(64, 5, device="cuda")
    actual = torch.zeros(1, n_token, n_token, 64, device="cuda")

    with torch.no_grad():
        template_coordinate_projection_add_(
            actual,
            pseudo_beta,
            frame,
            mask,
            mask,
            asym,
            dgram_weight,
            scalar_weight,
        )
    expected_from_origin = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 39.0], device="cuda")
    torch.testing.assert_close(actual[0, 0, :, 0], expected_from_origin, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_template_coordinate_compile_reuse_across_lengths():
    """One filesystem compile per math mode must serve every sequence length."""
    script = r"""
import json
import os
from pathlib import Path

import torch

from openfold3.core.kernels.triton.fused_template_coordinate import (
    _template_coordinate_projection_kernel,
    _template_coordinate_projection_bwd_partial_kernel,
    _template_coordinate_projection_bwd_reduce_kernel,
    template_coordinate_projection,
    template_coordinate_projection_add_,
    template_coordinate_projection_backward,
)

cache_dir = os.environ["TRITON_CACHE_DIR"]
Path(cache_dir).mkdir(parents=True, exist_ok=True)

def clear():
    _template_coordinate_projection_kernel.device_caches.clear()
    _template_coordinate_projection_bwd_partial_kernel.device_caches.clear()
    _template_coordinate_projection_bwd_reduce_kernel.device_caches.clear()

def count():
    return {
        "forward": len(
            list(Path(cache_dir).rglob("_template_coordinate_projection_kernel.json"))
        ),
        "backward_partial": len(
            list(
                Path(cache_dir).rglob(
                    "_template_coordinate_projection_bwd_partial_kernel.json"
                )
            )
        ),
        "backward_reduce": len(
            list(
                Path(cache_dir).rglob(
                    "_template_coordinate_projection_bwd_reduce_kernel.json"
                )
            )
        ),
    }

torch.set_grad_enabled(False)
w_dgram = torch.randn(64, 39, device="cuda")
w_scalar = torch.randn(64, 5, device="cuda")
for allow_tf32 in (False, True):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    for n in (16, 32, 48, 64, 96):
        source = torch.zeros(1, n, n, 64, device="cuda")
        args = (
            torch.randn(1, n, 3, device="cuda"),
            torch.randn(1, n, 3, 3, device="cuda"),
            torch.ones(1, n, device="cuda"),
            torch.ones(1, n, device="cuda"),
            torch.ones(1, n, dtype=torch.int32, device="cuda"),
            w_dgram,
            w_scalar,
        )
        if n % 32:
            template_coordinate_projection(source, *args)
        else:
            template_coordinate_projection_add_(source, *args)
        grad_output = torch.randn_like(source)
        if n % 32:
            grad_output = grad_output.transpose(1, 2)
        template_coordinate_projection_backward(grad_output, *args[:5])
        clear()

counts = count()
print(json.dumps({"counts": counts}))
assert counts == {
    "forward": 1,
    "backward_partial": 2,
    "backward_reduce": 1,
}, counts
"""
    with tempfile.TemporaryDirectory() as cache_dir:
        env = os.environ.copy()
        env["TRITON_CACHE_DIR"] = cache_dir
        env["OPENFOLD3_FUSED_TEMPLATE_COORD"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert '"backward_partial": 2' in result.stdout
