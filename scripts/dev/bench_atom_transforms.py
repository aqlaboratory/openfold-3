#!/usr/bin/env python
"""Microbenchmark atom broadcast/aggregation paths."""

import time

from openfold3.entry_points.import_utils import _torch_gpu_setup

_torch_gpu_setup()

import torch  # noqa: E402

from openfold3.core.utils.atomize_utils import (  # noqa: E402
    aggregate_atom_feat_to_tokens,
    aggregate_atom_feat_to_tokens_segmented,
    broadcast_token_feat_to_atoms,
    broadcast_token_feat_to_atoms_by_index,
)


def time_ms(fn, reps=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / reps


def main():
    n_token, samples = 1264, 5
    gen = torch.Generator().manual_seed(42)
    lengths = torch.randint(4, 13, (1, n_token), generator=gen)
    atom_index = torch.repeat_interleave(torch.arange(n_token), lengths[0])
    n_atom = atom_index.numel()
    lengths = lengths[:, None].cuda()
    atom_index = atom_index.reshape(1, 1, n_atom).cuda()
    token_mask = torch.ones(1, 1, n_token, device="cuda")
    atom_mask = torch.ones(1, 1, n_atom, device="cuda")
    token_feat = torch.randn(1, samples, n_token, 128, device="cuda")
    atom_feat = torch.randn(1, samples, n_atom, 128, device="cuda")

    def dynamic():
        return broadcast_token_feat_to_atoms(token_mask, lengths, token_feat, -2)

    def indexed():
        return broadcast_token_feat_to_atoms_by_index(
            token_mask, atom_index, atom_mask, token_feat
        )

    def atomic():
        return aggregate_atom_feat_to_tokens(
            token_mask, atom_index, atom_mask, atom_feat, atom_dim=-2
        )

    def segmented():
        return aggregate_atom_feat_to_tokens_segmented(lengths, atom_mask, atom_feat)

    torch.testing.assert_close(indexed(), dynamic())
    torch.testing.assert_close(segmented(), atomic(), rtol=1e-5, atol=1e-6)

    d, i, a, s = (time_ms(fn) for fn in (dynamic, indexed, atomic, segmented))
    print(torch.cuda.get_device_name())
    print(f"broadcast {d:.3f}->{i:.3f} ms | aggregate {a:.3f}->{s:.3f} ms")
    print(f"combined {(2 * d + a) / (2 * i + s):.2f}x")


if __name__ == "__main__":
    main()
