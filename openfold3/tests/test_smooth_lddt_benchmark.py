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

from pathlib import Path

import numpy as np
import pytest
import torch

from openfold3.core.data.io.structure.atom_array import read_atomarray_from_npz
from openfold3.core.data.resources.residues import (
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES_3,
    STANDARD_RNA_RESIDUES,
)
from openfold3.core.loss.diffusion import (
    smooth_lddt_loss,
    smooth_lddt_loss_ball_query,
)

MINIMAL_TRAINING_ROOT = (
    Path(__file__).parents[2]
    / "data"
    / "pdb_training_minimal"
    / "preprocessed_pdb_data"
    / "standard"
    / "structure_files"
)
MINIMAL_TRAINING_SAMPLE_IDS = ["102m", "12e8", "134d", "17ra"]

pytestmark = [
    pytest.mark.slow,
    pytest.mark.benchmark,
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Ball-query smooth lDDT requires CUDA"
    ),
]


def _minimal_training_sample_path(sample_id: str) -> Path:
    return MINIMAL_TRAINING_ROOT / sample_id / f"{sample_id}.npz"


def _minimal_training_batch(sample_id: str) -> dict:
    sample_path = _minimal_training_sample_path(sample_id)
    if not sample_path.exists():
        pytest.skip(f"Minimal training sample not found: {sample_path}")

    atom_array = read_atomarray_from_npz(sample_path)
    device = torch.device("cuda")
    finite_atom_mask_np = np.isfinite(atom_array.coord).all(axis=-1)
    coords_np = np.nan_to_num(atom_array.coord, nan=0.0, posinf=0.0, neginf=0.0)
    coords = torch.as_tensor(coords_np, device=device, dtype=torch.float32)
    res_names = np.asarray(atom_array.res_name).astype(str)
    is_protein_np = np.isin(res_names, STANDARD_PROTEIN_RESIDUES_3)
    is_rna_np = np.isin(res_names, STANDARD_RNA_RESIDUES)
    is_dna_np = np.isin(res_names, STANDARD_DNA_RESIDUES)
    n_atom = coords.shape[0]

    token_mask = torch.ones((1, n_atom), device=device)
    num_atoms_per_token = torch.ones((1, n_atom), device=device)
    is_dna = torch.as_tensor(is_dna_np, device=device, dtype=torch.float32)[None]
    is_rna = torch.as_tensor(is_rna_np, device=device, dtype=torch.float32)[None]
    is_protein = torch.as_tensor(is_protein_np, device=device, dtype=torch.float32)[
        None
    ]
    is_ligand = 1.0 - torch.clamp(is_protein + is_rna + is_dna, max=1.0)

    return {
        "token_mask": token_mask,
        "num_atoms_per_token": num_atoms_per_token,
        "is_protein": is_protein,
        "is_rna": is_rna,
        "is_dna": is_dna,
        "is_ligand": is_ligand,
        "ground_truth": {
            "atom_resolved_mask": torch.as_tensor(
                finite_atom_mask_np, device=device, dtype=torch.float32
            )[None],
            "atom_positions": coords[None],
        },
    }


def _forward(x, batch, loss_token_mask, backend, top_k):
    if backend == "dense":
        return smooth_lddt_loss(
            x=x, batch=batch, loss_token_mask=loss_token_mask, eps=1e-8
        )
    return smooth_lddt_loss_ball_query(
        x=x,
        batch=batch,
        loss_token_mask=loss_token_mask,
        eps=1e-8,
        top_k=top_k,
    )


@pytest.mark.parametrize(
    "sample_id",
    [
        pytest.param(sample_id, id=f"sample_{sample_id}")
        for sample_id in MINIMAL_TRAINING_SAMPLE_IDS
    ],
)
@pytest.mark.parametrize(
    ("backend", "top_k"),
    [
        pytest.param("dense", None, id="dense"),
        pytest.param("ball_query", 128, id="ball_query_max_neighbors128"),
        pytest.param("ball_query", 256, id="ball_query_max_neighbors256"),
        pytest.param("ball_query", 512, id="ball_query_max_neighbors512"),
        pytest.param("ball_query", 768, id="ball_query_max_neighbors768"),
        pytest.param("ball_query", 1024, id="ball_query_max_neighbors1024"),
        pytest.param("ball_query", 2048, id="ball_query_max_neighbors2048"),
    ],
)
def test_smooth_lddt_loss_benchmark(benchmark, sample_id, backend, top_k):
    torch.manual_seed(0)
    n_sample = 2
    batch = _minimal_training_batch(sample_id)
    x = batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1))
    x = x + 0.05 * torch.randn_like(x)
    loss_token_mask = torch.ones_like(batch["is_protein"])

    def _run():
        torch.cuda.synchronize()
        loss = _forward(x, batch, loss_token_mask, backend, top_k)
        torch.cuda.synchronize()
        return loss

    _run()
    loss = benchmark.pedantic(_run, rounds=1, iterations=1, warmup_rounds=0)
    assert loss.shape == (1, n_sample)


@pytest.mark.parametrize(
    "sample_id",
    [
        pytest.param(sample_id, id=f"sample_{sample_id}")
        for sample_id in MINIMAL_TRAINING_SAMPLE_IDS
    ],
)
@pytest.mark.parametrize(
    ("backend", "top_k"),
    [
        pytest.param("dense", None, id="dense"),
        pytest.param("ball_query", 256, id="ball_query_max_neighbors256"),
        pytest.param("ball_query", 512, id="ball_query_max_neighbors512"),
        pytest.param("ball_query", 1024, id="ball_query_max_neighbors1024"),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float32, id="fp32"),
        pytest.param(torch.bfloat16, id="bf16"),
    ],
)
def test_smooth_lddt_loss_fwd_bwd_benchmark(
    benchmark, sample_id, backend, top_k, dtype
):
    """Forward + backward time, parameterized over dtype to expose bf16 path."""
    torch.manual_seed(0)
    n_sample = 2
    batch = _minimal_training_batch(sample_id)
    x_base = batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1))
    x_base = x_base + 0.05 * torch.randn_like(x_base)
    loss_token_mask = torch.ones_like(batch["is_protein"])

    def _run():
        x = x_base.detach().to(dtype).clone().requires_grad_(True)
        torch.cuda.synchronize()
        loss = _forward(x, batch, loss_token_mask, backend, top_k)
        loss.sum().backward()
        torch.cuda.synchronize()
        return loss, x.grad

    _run()
    loss, grad = benchmark.pedantic(_run, rounds=1, iterations=1, warmup_rounds=0)
    assert loss.shape == (1, n_sample)
    assert grad is not None


@pytest.mark.parametrize(
    "sample_id",
    [
        pytest.param(sample_id, id=f"sample_{sample_id}")
        for sample_id in MINIMAL_TRAINING_SAMPLE_IDS
    ],
)
@pytest.mark.parametrize(
    ("backend", "top_k"),
    [
        pytest.param("dense", None, id="dense"),
        pytest.param("ball_query", 256, id="ball_query_max_neighbors256"),
        pytest.param("ball_query", 512, id="ball_query_max_neighbors512"),
        pytest.param("ball_query", 1024, id="ball_query_max_neighbors1024"),
    ],
)
def test_smooth_lddt_loss_peak_memory(sample_id, backend, top_k):
    """Report fwd/bwd peak GPU memory; not a benchmark fixture, just a measurement."""
    torch.manual_seed(0)
    n_sample = 2
    batch = _minimal_training_batch(sample_id)
    x_base = batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1))
    x_base = x_base + 0.05 * torch.randn_like(x_base)
    loss_token_mask = torch.ones_like(batch["is_protein"])

    # Warm up the JIT extension
    x = x_base.detach().clone().requires_grad_(True)
    loss = _forward(x, batch, loss_token_mask, backend, top_k)
    loss.sum().backward()
    torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = x_base.detach().clone().requires_grad_(True)
    loss = _forward(x, batch, loss_token_mask, backend, top_k)
    fwd_peak = torch.cuda.max_memory_allocated()

    loss.sum().backward()
    torch.cuda.synchronize()
    bwd_peak = torch.cuda.max_memory_allocated()

    n_atom = x.shape[-2]
    print(
        f"\n[memory] sample={sample_id} backend={backend} top_k={top_k} "
        f"n_atom={n_atom} fwd_peak={fwd_peak / 1e6:.1f}MB "
        f"bwd_peak={bwd_peak / 1e6:.1f}MB"
    )
    assert loss.shape == (1, n_sample)
    assert x.grad is not None
