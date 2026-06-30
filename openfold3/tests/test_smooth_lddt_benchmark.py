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
MINIMAL_TRAINING_SAMPLE_IDS = ["102m", "12e8", "134d", "17ra", "1tii", "1xfy"]

BACKENDS = [
    ("dense", None),
    ("ball_query", 128),
    ("ball_query", 256),
    ("ball_query", 512),
    ("ball_query", 1024),
    ("ball_query", 2048),
]

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


def _backend_label(backend, top_k):
    if backend == "dense":
        return "dense"
    return f"bq K={top_k}"


def _time_fwd_bwd(x_base, batch, loss_token_mask, backend, top_k, dtype):
    x = x_base.detach().to(dtype).clone().requires_grad_(True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    loss = _forward(x, batch, loss_token_mask, backend, top_k)
    loss.sum().backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), loss, x.grad


def _measure_peak_memory(x_base, batch, loss_token_mask, backend, top_k):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = x_base.detach().clone().requires_grad_(True)
    loss = _forward(x, batch, loss_token_mask, backend, top_k)
    loss.sum().backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


@pytest.mark.parametrize(
    "sample_id",
    [pytest.param(sid, id=f"sample_{sid}") for sid in MINIMAL_TRAINING_SAMPLE_IDS],
)
def test_smooth_lddt_benchmark(sample_id):
    """Time + memory comparison across all backends for one sample."""
    torch.manual_seed(0)
    n_sample = 2
    batch = _minimal_training_batch(sample_id)
    n_atom = batch["ground_truth"]["atom_positions"].shape[-2]
    x_base = batch["ground_truth"]["atom_positions"].repeat((1, n_sample, 1, 1))
    x_base = x_base + 0.05 * torch.randn_like(x_base)
    loss_token_mask = torch.ones_like(batch["is_protein"])

    # Warmup all backends
    for backend, top_k in BACKENDS:
        _forward(
            x_base.detach().clone().requires_grad_(True),
            batch,
            loss_token_mask,
            backend,
            top_k,
        )
        torch.cuda.synchronize()

    rows = []
    for backend, top_k in BACKENDS:
        label = _backend_label(backend, top_k)
        time_fp32, loss, grad = _time_fwd_bwd(
            x_base,
            batch,
            loss_token_mask,
            backend,
            top_k,
            torch.float32,
        )
        assert loss.shape == (1, n_sample)
        assert grad is not None
        time_bf16, _, _ = _time_fwd_bwd(
            x_base,
            batch,
            loss_token_mask,
            backend,
            top_k,
            torch.bfloat16,
        )
        peak = _measure_peak_memory(
            x_base,
            batch,
            loss_token_mask,
            backend,
            top_k,
        )
        rows.append((label, time_fp32, time_bf16, peak))

    dense_time = rows[0][1]
    dense_mem = rows[0][3]

    header = (
        f"\n{'=' * 78}\n"
        f"  sample={sample_id}  n_atom={n_atom}  n_sample={n_sample}\n"
        f"{'=' * 78}\n"
        f"  {'backend':<12s} {'fp32 (ms)':>10s} {'bf16 (ms)':>10s}"
        f" {'peak mem':>10s} {'speedup':>8s} {'mem save':>9s}\n"
        f"  {'-' * 12} {'-' * 10} {'-' * 10}"
        f" {'-' * 10} {'-' * 8} {'-' * 9}"
    )
    print(header)
    for label, t32, t16, mem in rows:
        speedup = f"{dense_time / t32:.1f}x" if t32 > 0 else "—"
        mem_save = f"{dense_mem / mem:.1f}x" if mem > 0 else "—"
        print(
            f"  {label:<12s} {t32:>10.2f} {t16:>10.2f}"
            f" {mem / 1e6:>9.1f}M {speedup:>8s} {mem_save:>9s}"
        )
    print()
