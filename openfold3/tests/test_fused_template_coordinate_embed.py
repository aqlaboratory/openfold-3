# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import pytest
import torch

from openfold3.core.data.primitives.featurization.template import (
    create_template_distogram,
    create_template_unit_vector,
)
from openfold3.core.model.feature_embedders.template_embedders import (
    TemplatePairEmbedderAllAtom,
)
from openfold3.core.model.primitives.fused_template_pair_embedder import (
    fused_template_coordinate_pair_embedder_inference,
    template_coordinate_projection_add_reference_,
)
from openfold3.projects.of3_all_atom.config.dataset_config_components import (
    TemplateSettings,
)


def _coordinates(n_token: int) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(n_token, dtype=torch.float32)
    ca = torch.stack(
        (index * 3.8, torch.sin(index * 0.3), torch.cos(index * 0.2)), dim=-1
    )
    pseudo_beta = ca + torch.tensor([0.2, -0.3, 0.7])
    n_atom = ca + torch.tensor([-0.7, 1.1, 0.2])
    c_atom = ca + torch.tensor([1.3, 0.2, -0.1])
    frame = torch.stack((n_atom, ca, c_atom), dim=-2)
    return pseudo_beta[None], frame[None]


def _coordinate_batch(n_token: int, device: str = "cpu") -> dict:
    pseudo_beta, frame = _coordinates(n_token)
    restype = torch.nn.functional.one_hot(
        torch.arange(n_token) % 32, num_classes=32
    ).to(torch.int32)
    batch = {
        "template_pseudo_beta_coords": pseudo_beta[None].to(device),
        "template_frame_atom_coords": frame[None].to(device),
        "template_pseudo_beta_mask": torch.ones(1, 1, n_token, device=device),
        "template_backbone_frame_mask": torch.ones(1, 1, n_token, device=device),
        "template_restype": restype[None, None].to(device),
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


@pytest.mark.parametrize("n_token", [8, 17])
def test_coordinate_reference_matches_precomputed_embedder(n_token: int):
    torch.manual_seed(7)
    module = TemplatePairEmbedderAllAtom(128, 39, 32, 64).eval()
    coordinate_batch = _coordinate_batch(n_token)
    precomputed_batch = _precomputed_batch(coordinate_batch)
    z = torch.randn(1, n_token, n_token, 128)

    with torch.inference_mode():
        expected = module(precomputed_batch, z.clone())
        actual = fused_template_coordinate_pair_embedder_inference(
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


def test_coordinate_embedding_rejects_training():
    n_token = 8
    module = TemplatePairEmbedderAllAtom(128, 39, 32, 64).train()
    batch = _coordinate_batch(n_token)
    z = torch.randn(1, n_token, n_token, 128)

    with torch.no_grad(), pytest.raises(RuntimeError, match="inference-only"):
        fused_template_coordinate_pair_embedder_inference(
            module=module,
            batch=batch,
            z=z,
            template_index=0,
        )


def test_coordinate_features_reject_nondefault_distogram_settings():
    with pytest.raises(ValueError, match="default"):
        TemplateSettings(
            use_coordinate_pair_features=True,
            distogram={"min_bin": 2.0, "max_bin": 50.75, "n_bins": 39},
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_token", [8, 31, 64])
def test_coordinate_triton_matches_chunked_reference(n_token: int):
    from openfold3.core.kernels.triton.fused_template_coordinate import (
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
    expected = torch.randn(1, n_token, n_token, 64, device="cuda")
    actual = expected.clone()

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
    template_coordinate_projection_add_(
        actual,
        pseudo_beta,
        frame,
        pb_mask,
        bb_mask,
        asym,
        dgram_weight,
        scalar_weight,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=3e-4)


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
