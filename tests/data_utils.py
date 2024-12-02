# Copyright 2021 AlQuraishi Laboratory
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

from random import randint

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from openfold3.core.np.token_atom_constants import (
    DNA_NUCLEOTIDE_TYPES,
    PROTEIN_RESTYPES,
    RNA_NUCLEOTIDE_TYPES,
    TOKEN_NAME_TO_ATOM_NAMES,
    TOKEN_TYPES,
    TOKEN_TYPES_WITH_GAP,
)
from tests.config import consts


def random_asym_ids(n_res, split_chains=True, min_chain_len=4):
    n_chain = randint(1, n_res // min_chain_len) if consts.is_multimer else 1

    if not split_chains:
        return [0] * n_res

    assert n_res >= n_chain

    pieces = []
    asym_ids = []
    final_idx = n_chain - 1
    for idx in range(n_chain - 1):
        n_stop = n_res - sum(pieces) - n_chain + idx - min_chain_len
        if n_stop <= min_chain_len:
            final_idx = idx
            break
        piece = randint(min_chain_len, n_stop)
        pieces.append(piece)
        asym_ids.extend(piece * [idx])
    asym_ids.extend((n_res - sum(pieces)) * [final_idx])

    return np.array(asym_ids).astype(np.float32) + 1


def random_template_feats(n_templ, n, batch_size=None):
    b = []
    if batch_size is not None:
        b.append(batch_size)
    batch = {
        "template_mask": np.random.randint(0, 2, (*b, n_templ)),
        "template_pseudo_beta_mask": np.random.randint(0, 2, (*b, n_templ, n)),
        "template_pseudo_beta": np.random.rand(*b, n_templ, n, 3),
        "template_aatype": np.random.randint(0, 22, (*b, n_templ, n)),
        "template_all_atom_mask": np.random.randint(0, 2, (*b, n_templ, n, 37)),
        "template_all_atom_positions": np.random.rand(*b, n_templ, n, 37, 3) * 10,
        "template_torsion_angles_sin_cos": np.random.rand(*b, n_templ, n, 7, 2),
        "template_alt_torsion_angles_sin_cos": np.random.rand(*b, n_templ, n, 7, 2),
        "template_torsion_angles_mask": np.random.rand(*b, n_templ, n, 7),
    }
    batch = {k: v.astype(np.float32) for k, v in batch.items()}
    batch["template_aatype"] = batch["template_aatype"].astype(np.int64)

    return batch


def random_extra_msa_feats(n_extra, n, batch_size=None):
    b = []
    if batch_size is not None:
        b.append(batch_size)
    batch = {
        "extra_msa": np.random.randint(0, 22, (*b, n_extra, n)).astype(np.int64),
        "extra_has_deletion": np.random.randint(0, 2, (*b, n_extra, n)).astype(
            np.float32
        ),
        "extra_deletion_value": np.random.rand(*b, n_extra, n).astype(np.float32),
        "extra_msa_mask": np.random.randint(0, 2, (*b, n_extra, n)).astype(np.float32),
    }
    return batch


def random_affines_vector(dim):
    prod_dim = 1
    for d in dim:
        prod_dim *= d

    affines = np.zeros((prod_dim, 7)).astype(np.float32)

    for i in range(prod_dim):
        affines[i, :4] = Rotation.random(random_state=42).as_quat()
        affines[i, 4:] = np.random.rand(
            3,
        ).astype(np.float32)

    return affines.reshape(*dim, 7)


def random_affines_4x4(dim):
    prod_dim = 1
    for d in dim:
        prod_dim *= d

    affines = np.zeros((prod_dim, 4, 4)).astype(np.float32)

    for i in range(prod_dim):
        affines[i, :3, :3] = Rotation.random(random_state=42).as_matrix()
        affines[i, :3, 3] = np.random.rand(
            3,
        ).astype(np.float32)

    affines[:, 3, 3] = 1

    return affines.reshape(*dim, 4, 4)


def random_attention_inputs(
    batch_size,
    n_seq,
    n,
    no_heads,
    c_hidden,
    inf=1e9,
    dtype=torch.float32,
    requires_grad=False,
):
    q = torch.rand(
        batch_size, n_seq, n, c_hidden, dtype=dtype, requires_grad=requires_grad
    ).cuda()
    kv = torch.rand(
        batch_size, n_seq, n, c_hidden, dtype=dtype, requires_grad=requires_grad
    ).cuda()

    mask = torch.randint(
        0, 2, (batch_size, n_seq, 1, 1, n), dtype=dtype, requires_grad=False
    ).cuda()
    z_bias = torch.rand(
        batch_size, 1, no_heads, n, n, dtype=dtype, requires_grad=requires_grad
    ).cuda()
    mask_bias = inf * (mask - 1)

    biases = [mask_bias, z_bias]

    return q, kv, mask, biases


def random_af3_features(batch_size, n_token, n_msa, n_templ):
    restypes_flat = torch.randint(0, len(TOKEN_TYPES), (n_token,))
    restypes_names = [TOKEN_TYPES[token_idx] for token_idx in restypes_flat]
    restypes_one_hot = torch.nn.functional.one_hot(
        restypes_flat,
        len(TOKEN_TYPES_WITH_GAP),
    )

    num_atoms_per_token = torch.Tensor(
        [len(TOKEN_NAME_TO_ATOM_NAMES[name]) for name in restypes_names]
    )

    is_protein = torch.Tensor(
        [1.0 if name in PROTEIN_RESTYPES else 0.0 for name in restypes_names]
    )
    is_rna = torch.Tensor(
        [1.0 if name in RNA_NUCLEOTIDE_TYPES else 0.0 for name in restypes_names]
    )
    is_dna = torch.Tensor(
        [1.0 if name in DNA_NUCLEOTIDE_TYPES else 0.0 for name in restypes_names]
    )

    n_atom = torch.max(torch.sum(num_atoms_per_token, dim=-1)).int().item()

    start_atom_index = torch.concat(
        (torch.zeros((1,)), torch.cumsum(num_atoms_per_token, dim=-1)[:-1]), dim=-1
    )

    asym_id = (
        torch.Tensor(random_asym_ids(n_token)).unsqueeze(0).repeat((batch_size, 1))
    )

    return {
        # Input features
        "residue_index": torch.arange(0, n_token)
        .unsqueeze(0)
        .repeat((batch_size, 1))
        .int(),
        "token_index": torch.arange(0, n_token)
        .unsqueeze(0)
        .repeat((batch_size, 1))
        .int(),
        "asym_id": asym_id.int(),
        "entity_id": asym_id.clone().int(),
        "sym_id": torch.ones((batch_size, n_token)).int(),
        "restype": restypes_one_hot.unsqueeze(0).repeat((batch_size, 1, 1)).int(),
        "is_protein": is_protein.unsqueeze(0).repeat((batch_size, 1)).int(),
        "is_dna": is_dna.unsqueeze(0).repeat((batch_size, 1)).int(),
        "is_rna": is_rna.unsqueeze(0).repeat((batch_size, 1)).int(),
        "is_ligand": torch.zeros((batch_size, n_token)).int(),
        "is_atomized": torch.zeros((batch_size, n_token)).int(),
        # Reference conformation features
        "ref_pos": torch.randn((batch_size, n_atom, 3)).float(),
        "ref_mask": torch.ones((batch_size, n_atom)).int(),
        "ref_element": torch.ones((batch_size, n_atom, 119)).int(),
        "ref_charge": torch.ones((batch_size, n_atom)).float(),
        "ref_atom_name_chars": torch.ones((batch_size, n_atom, 4, 64)).int(),
        "ref_space_uid": torch.zeros((batch_size, n_atom)).int(),
        # MSA features
        "msa": torch.ones((batch_size, n_msa, n_token, 32)).int(),
        "has_deletion": torch.ones((batch_size, n_msa, n_token)).float(),
        "deletion_value": torch.ones((batch_size, n_msa, n_token)).float(),
        "profile": torch.ones((batch_size, n_token, 32)).float(),
        "deletion_mean": torch.ones((batch_size, n_token)).float(),
        # Template features
        "template_restype": torch.ones((batch_size, n_templ, n_token, 32)).int(),
        "template_pseudo_beta_mask": torch.ones((batch_size, n_templ, n_token)).float(),
        "template_backbone_frame_mask": torch.ones(
            (batch_size, n_templ, n_token)
        ).float(),
        "template_distogram": torch.ones(
            (batch_size, n_templ, n_token, n_token, 39)
        ).int(),
        "template_unit_vector": torch.ones(
            (batch_size, n_templ, n_token, n_token, 3)
        ).float(),
        # Bond features
        "token_bonds": torch.ones((batch_size, n_token, n_token)).int(),
        # Additional features
        "token_mask": torch.ones((batch_size, n_token)).float(),
        "atom_mask": torch.ones((batch_size, n_atom)).float(),
        "start_atom_index": start_atom_index.unsqueeze(0).repeat((batch_size, 1)).int(),
        "num_atoms_per_token": num_atoms_per_token.unsqueeze(0)
        .repeat((batch_size, 1))
        .int(),
        "msa_mask": torch.ones((batch_size, n_msa, n_token)).float(),
        "num_paired_seqs": torch.randint(
            low=n_msa // 4, high=n_msa // 2, size=(batch_size,)
        ),
        "ground_truth": {
            "atom_positions": torch.randn((batch_size, n_atom, 3)).float(),
            "atom_resolved_mask": torch.ones((batch_size, n_atom)).float(),
        },
        "loss_weights": {
            "bond": torch.Tensor([0.0]).repeat(batch_size),
            "smooth_lddt": torch.Tensor([4.0]).repeat(batch_size),
            "mse": torch.Tensor([4.0]).repeat(batch_size),
            "plddt": torch.Tensor([1e-4]).repeat(batch_size),
            "pde": torch.Tensor([1e-4]).repeat(batch_size),
            "experimentally_resolved": torch.Tensor([1e-4]).repeat(batch_size),
            "pae": torch.Tensor([0.0]).repeat(batch_size),
            "distogram": torch.Tensor([3e-2]).repeat(batch_size),
        },
    }
