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

import math
import unittest

import numpy as np
import torch

from openfold3.core.data.resources.lists import AB_AG_CHAIN_TYPES
from openfold3.core.metrics.quality import (
    _spread_contacts,
    dockq,
    dockq_full_complex,
    drmsd,
    fnat,
    gdt_ha,
    gdt_ts,
    get_ab_ag_metrics,
    get_metrics,
    get_metrics_chunked,
    get_superimpose_metrics,
    interface_lddt,
    lddt,
    rmsd,
)
from openfold3.core.utils.geometry.kabsch_alignment import (
    apply_transformation,
    get_optimal_transformation,
    kabsch_align,
)
from openfold3.core.utils.tensor_utils import tensor_tree_map
from openfold3.tests.config import consts
from openfold3.tests.data_utils import random_of3_features


def random_rotation_translation(structure, factor=100.0):
    """
    Applies random rotations and translations to a given structure
    Args:
        Factor: a multiplier to translation
    Returns:
        new_structure: randomly rotated and translated conformer [*, n_atom, 3]
    """
    # rotation: Rx, Ry, Rz
    x_angle, y_angle, z_angle = torch.randn(3) * 2 * math.pi
    x_rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(x_angle), -math.sin(x_angle)],
            [0.0, math.sin(x_angle), math.cos(x_angle)],
        ]
    ).to(torch.float32)
    y_rotation = torch.tensor(
        [
            [math.cos(y_angle), 0.0, math.sin(y_angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(y_angle), 0.0, math.cos(y_angle)],
        ]
    ).to(torch.float32)
    z_rotation = torch.tensor(
        [
            [math.cos(z_angle), -math.sin(z_angle), 0.0],
            [math.sin(z_angle), math.cos(z_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    ).to(torch.float32)
    xyz_rotation = x_rotation @ y_rotation @ z_rotation

    # 2. translation
    translation = (
        torch.randn(
            size=structure.shape[:-2]
            + (
                1,
                3,
            )
        )
        * factor
    )
    translation = translation.to(torch.float32)
    new_structure = structure @ xyz_rotation + translation
    return new_structure


class TestLDDT(unittest.TestCase):
    def test_lddt(self):
        batch_size = consts.batch_size
        n_atom = consts.n_res

        gt_structure = torch.randn(batch_size, n_atom, 3)  # [batch_size, n_atom, 3]
        predicted_structure = torch.randn(
            batch_size, n_atom, 3
        )  # [batch_size, n_atom, 3]
        atom_mask = torch.ones(batch_size, n_atom).bool()  # [batch_size, n_atom]

        # TODO: write a test that checks intra / inter masking behavior
        intra_mask_filter = torch.ones(
            (batch_size, n_atom)
        ).bool()  # [batch_size, n_atom]
        inter_mask_filter = torch.ones(
            (batch_size, n_atom, n_atom)
        ).bool()  # [batch_size, n_atom, n_atom]

        pair_gt = torch.cdist(
            gt_structure, gt_structure
        )  # [batch_size, n_atom, n_atom]
        pair_pred = torch.cdist(predicted_structure, predicted_structure)
        asym_id = torch.randint(low=0, high=21, size=(n_atom,))  # [n_atom]
        asym_id = asym_id.unsqueeze(0).expand(batch_size, -1)  # [batch_size, n_atom]

        # shape test
        intra_lddt, inter_lddt = lddt(
            pair_pred,
            pair_gt,
            atom_mask,
            intra_mask_filter,
            inter_mask_filter,
            asym_id,
        )
        exp_shape = (batch_size,)
        np.testing.assert_equal(intra_lddt.shape, exp_shape)
        np.testing.assert_equal(inter_lddt.shape, exp_shape)

        # lddt should always be less than one
        np.testing.assert_array_less(intra_lddt, 1.0)
        np.testing.assert_array_less(inter_lddt, 1.0)

        # rototranslation.
        # lddt between gt_structure and gt_structure_rototranslated should give 1.0s
        # note: in the test case with small variance (randn), inter_lddt should be valid
        # but when all inter atom pairs > 15. or 30. (dna/rna), returns nan
        gt_structure_rototranslated = random_rotation_translation(gt_structure)
        exp_outputs = torch.ones(batch_size)
        pair_rototranslated = torch.cdist(
            gt_structure_rototranslated,
            gt_structure_rototranslated,
        )
        intra_lddt_rt, inter_lddt_rt = lddt(
            pair_rototranslated,
            pair_gt,
            atom_mask,
            intra_mask_filter,
            inter_mask_filter,
            asym_id,
        )
        np.testing.assert_allclose(
            intra_lddt_rt,
            exp_outputs,
            atol=consts.eps,
        )
        np.testing.assert_allclose(
            inter_lddt_rt,
            exp_outputs,
            atol=consts.eps,
        )


class TestInterfaceLDDT(unittest.TestCase):
    def test_interface_lddt(self):
        batch_size = consts.batch_size
        n_atom = consts.n_res
        n_atom2 = 5
        gt_structure_1 = torch.randn(batch_size, n_atom, 3)  # [batch_size, n_atom, 3]
        gt_structure_2 = torch.randn(batch_size, n_atom2, 3)  # [batch_size, n_atom, 3]
        predicted_structure_1 = torch.randn(
            batch_size, n_atom, 3
        )  # [batch_size, n_atom, 3]
        predicted_structure_2 = torch.randn(
            batch_size, n_atom2, 3
        )  # [batch_size, n_atom, 3]

        mask1 = torch.ones(batch_size, n_atom).bool()  # [batch_size, n_atom,]
        mask2 = torch.ones(batch_size, n_atom2).bool()  # [batch_size, n_atom,]
        filter_mask = torch.ones(batch_size, n_atom, n_atom2).bool()

        # shape test
        out_interface_lddt = interface_lddt(
            predicted_structure_1,
            predicted_structure_2,
            gt_structure_1,
            gt_structure_2,
            mask1,
            mask2,
            filter_mask,
        )
        exp_shape = (batch_size,)
        np.testing.assert_equal(out_interface_lddt.shape, exp_shape)

        # rototranslation test. should give 1.s
        # rototranslate two structures together
        combined_coordinates = torch.cat((gt_structure_1, gt_structure_2), dim=1)
        combined_coordinates_rt = random_rotation_translation(combined_coordinates)
        # split two molecules back
        p1, p2 = torch.split(combined_coordinates_rt, [n_atom, n_atom2], dim=1)
        # run interface_lddt
        out_interface_lddt = interface_lddt(
            p1, p2, gt_structure_1, gt_structure_2, mask1, mask2, filter_mask
        )
        exp_outputs = torch.ones(batch_size)
        np.testing.assert_allclose(out_interface_lddt, exp_outputs, atol=consts.eps)


class TestDRMSD(unittest.TestCase):
    def test_drmsd(self):
        batch_size = consts.batch_size
        n_atom = consts.n_res

        gt_structure = torch.randn(batch_size, n_atom, 3)  # [batch_size, n_atom, 3]
        predicted_structure = torch.randn(
            batch_size, n_atom, 3
        )  # [batch_size, n_atom, 3]
        mask = torch.ones(batch_size, n_atom).bool()  # [batch_size, n_atom]

        pair_gt = torch.cdist(
            gt_structure,
            gt_structure,
        )  # [batch_size, n_atom, n_atom]]
        pair_pred = torch.cdist(
            predicted_structure,
            predicted_structure,
        )  # [batch_size, n_atom, n_atom]]
        asym_id = torch.randint(low=0, high=21, size=(n_atom,))  # [n_atom]
        asym_id = asym_id.unsqueeze(0).expand(batch_size, -1)  # batch_size, n_atom

        # shape test
        intra_drmsd, inter_drmsd = drmsd(
            pair_pred,
            pair_gt,
            mask,
            asym_id,
        )
        exp_shape = (batch_size,)
        np.testing.assert_equal(intra_drmsd.shape, exp_shape)
        np.testing.assert_equal(inter_drmsd.shape, exp_shape)

        # rototranslation. compuate gt_structure and gt_structure_rototranslated drmsd
        # should give 0.s
        gt_structure_rototranslated = random_rotation_translation(gt_structure)
        pair_gt_rt = torch.cdist(
            gt_structure_rototranslated, gt_structure_rototranslated
        )
        exp_outputs = torch.zeros(batch_size)
        intra_drmsd_rt, inter_drmsd_rt = drmsd(
            pair_gt_rt,
            pair_gt,
            mask,
            asym_id,
        )
        np.testing.assert_allclose(intra_drmsd_rt, exp_outputs, atol=consts.eps)
        np.testing.assert_allclose(inter_drmsd_rt, exp_outputs, atol=consts.eps)


class TestKabschAlign(unittest.TestCase):
    def test_kabsch_align(self):
        batch_size = consts.batch_size
        n_atom = consts.n_res

        gt_structure = torch.randn(batch_size, n_atom, 3)  # [batch_size, n_atom, 3]
        pred_structure = torch.randn(batch_size, n_atom, 3)  # [batch_size, n_atom, 3]
        mask = torch.ones(batch_size, n_atom).bool()

        # shape test
        out_transformation = get_optimal_transformation(
            mobile_positions=pred_structure,
            target_positions=gt_structure,
            positions_mask=mask,
        )

        out_coordinates = apply_transformation(
            positions=pred_structure, transformation=out_transformation
        )

        exp_shape_translation = (batch_size, 1, 3)
        exp_shape_rotation = (batch_size, 3, 3)
        exp_shape_coordinates = (batch_size, n_atom, 3)
        np.testing.assert_equal(
            out_transformation.translation_vector.shape, exp_shape_translation
        )
        np.testing.assert_equal(
            out_transformation.rotation_matrix.shape, exp_shape_rotation
        )
        np.testing.assert_equal(out_coordinates.shape, exp_shape_coordinates)

        # rototranslation test. should give 0.s
        gt_structure_rototranslated = random_rotation_translation(gt_structure)
        exp_outputs = torch.zeros(batch_size)
        out_kabsch = kabsch_align(
            mobile_positions=gt_structure_rototranslated,
            target_positions=gt_structure,
            positions_mask=mask,
        )

        out_rmsd = rmsd(
            pred_positions=out_kabsch,
            target_positions=gt_structure,
            positions_mask=mask,
        )

        np.testing.assert_allclose(out_rmsd, exp_outputs, atol=consts.eps)


class TestGDT(unittest.TestCase):
    def test_gdt(self):
        batch_size = consts.batch_size
        n_atom = consts.n_res

        gt_structure = torch.randn(batch_size, n_atom, 3)
        predicted_structure = torch.randn(batch_size, n_atom, 3)
        mask = torch.ones(batch_size, n_atom).bool()

        # shape test
        pred_superimposed = kabsch_align(
            mobile_positions=predicted_structure,
            target_positions=gt_structure,
            positions_mask=mask,
        )

        out_gdt_ts = gdt_ts(pred_superimposed, gt_structure, mask)
        out_gdt_ha = gdt_ha(pred_superimposed, gt_structure, mask)

        exp_gdt_ts_shape = (batch_size,)
        exp_gdt_ha_shape = (batch_size,)
        np.testing.assert_equal(out_gdt_ts.shape, exp_gdt_ts_shape)
        np.testing.assert_equal(out_gdt_ha.shape, exp_gdt_ha_shape)

        # rototranslation test
        gt_structure = torch.randn(batch_size, n_atom, 3)

        mask = torch.ones(batch_size, n_atom).bool()
        pred = random_rotation_translation(gt_structure)
        pred_superimposed = kabsch_align(
            mobile_positions=pred,
            target_positions=gt_structure,
            positions_mask=mask,
        )

        out_gdt_ts = gdt_ts(pred_superimposed, gt_structure, mask)
        out_gdt_ha = gdt_ha(pred_superimposed, gt_structure, mask)

        exp_gdt_ts_outs = torch.ones(batch_size)
        exp_gdt_ha_outs = torch.ones(batch_size)

        np.testing.assert_allclose(out_gdt_ts, exp_gdt_ts_outs, atol=consts.eps)
        np.testing.assert_allclose(out_gdt_ha, exp_gdt_ha_outs, atol=consts.eps)


class TestGetSuperimposeMetrics(unittest.TestCase):
    def test_get_superimpose_metrics(self):
        batch_size = consts.batch_size
        n_atom = 1000

        coords_pred = torch.randn(batch_size, n_atom, 3)
        coords_gt = torch.randn(batch_size, n_atom, 3)
        all_atom_mask = torch.ones((n_atom,)).bool()

        out = get_superimpose_metrics(coords_pred, coords_gt, all_atom_mask)
        exp_shape = (batch_size,)
        for _, v in out.items():
            np.testing.assert_equal(v.shape, exp_shape)


class TestAllMetrics(unittest.TestCase):
    def test_all_metrics(self):
        no_samples = 5

        batch = random_of3_features(
            batch_size=consts.batch_size,
            n_token=consts.n_res,
            n_msa=consts.n_seq,
            n_templ=consts.n_templ,
            is_eval=True,
        )

        def expand_sample_dim(t: torch.tensor) -> torch.tensor:
            feat_dims = t.shape[2:]
            t = t.expand(-1, no_samples, *((-1,) * len(feat_dims)))
            return t

        batch = tensor_tree_map(lambda t: t.unsqueeze(1), batch)
        batch["ground_truth"] = tensor_tree_map(
            expand_sample_dim, batch["ground_truth"]
        )

        n_atom = batch["ref_pos"].shape[-2]
        outputs = {
            "atom_positions_predicted": torch.randn(
                consts.batch_size, no_samples, n_atom, 3
            )
        }

        # Set extra metrics to False for now in order to skip RASA (needs atom array)
        metrics = get_metrics(batch, outputs, compute_extra_val_metrics=False)
        metrics_chunked = get_metrics_chunked(
            batch, outputs, compute_extra_val_metrics=False
        )

        for name, value in metrics.items():
            chunked_value = metrics_chunked[name]
            assert value.shape == (consts.batch_size, no_samples)
            assert chunked_value.shape == (consts.batch_size, no_samples)
            assert torch.allclose(value, chunked_value)


# =============================================================================
# DockQ Helper Functions
# =============================================================================


def encode_atom_name(name: str) -> torch.Tensor:
    """
    Encode an atom name as a one-hot tensor [4, 64].
    Character encoding: ord(c) - 32 for each position, padded with spaces.
    """
    padded_name = name.ljust(4)[:4]  # Pad or truncate to exactly 4 chars
    one_hot = torch.zeros(4, 64)
    for i, c in enumerate(padded_name):
        code = ord(c) - 32
        if 0 <= code < 64:
            one_hot[i, code] = 1.0
    return one_hot


def build_protein_residue(center: torch.Tensor, res_id: int, asym_id: int) -> dict:
    """
    Build a protein residue with backbone atoms (N, CA, C, O).

    Args:
        center: CA position [3]
        res_id: Residue ID (1-indexed)
        asym_id: Chain ID

    Returns:
        dict with coords, atom_names, res_ids, asym_ids
    """
    # Standard backbone geometry (approximate)
    atom_names = ["N", "CA", "C", "O"]
    offsets = {
        "N": torch.tensor([-1.46, 0.0, 0.0]),
        "CA": torch.tensor([0.0, 0.0, 0.0]),
        "C": torch.tensor([1.52, 0.0, 0.0]),
        "O": torch.tensor([2.36, 1.0, 0.0]),
    }

    coords = torch.stack([center + offsets[name] for name in atom_names])
    return {
        "coords": coords,
        "atom_names": atom_names,
        "res_ids": [float(res_id)] * len(atom_names),
        "asym_ids": [float(asym_id)] * len(atom_names),
    }


def build_two_chain_protein_complex(
    chain_a_length: int = 5,
    chain_b_length: int = 5,
    interface_distance: float = 4.0,
    batch_size: int = 1,
    n_samples: int = 1,
) -> dict:
    """
    Build a two-chain protein complex with a realistic interface.

    Chain A is positioned with residues along the x-axis.
    Chain B is positioned parallel to Chain A, offset in the y-direction.
    The interface_distance controls how close the chains are.

    Returns:
        Dictionary with all atomized features needed for DockQ.
    """
    all_coords = []
    all_atom_names = []
    all_res_ids = []
    all_asym_ids = []

    # Build Chain A (asym_id = 1)
    for i in range(chain_a_length):
        center = torch.tensor([i * 3.8, 0.0, 0.0])
        residue = build_protein_residue(center, res_id=i + 1, asym_id=1)
        all_coords.append(residue["coords"])
        all_atom_names.extend(residue["atom_names"])
        all_res_ids.extend(residue["res_ids"])
        all_asym_ids.extend(residue["asym_ids"])

    # Build Chain B (asym_id = 2), offset in y-direction
    for i in range(chain_b_length):
        center = torch.tensor([i * 3.8, interface_distance, 0.0])
        residue = build_protein_residue(center, res_id=i + 1, asym_id=2)
        all_coords.append(residue["coords"])
        all_atom_names.extend(residue["atom_names"])
        all_res_ids.extend(residue["res_ids"])
        all_asym_ids.extend(residue["asym_ids"])

    # Combine all coordinates
    coords = torch.cat(all_coords, dim=0).float()  # [N_atom, 3]
    n_atom = coords.shape[0]

    # Encode atom names
    ref_atom_name_chars = torch.stack(
        [encode_atom_name(name) for name in all_atom_names]
    )

    # Build feature tensors with correct shapes [B, S, N_atom, ...]
    features = {
        "coords": coords.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, n_samples, -1, -1)
        .clone(),
        "ref_atom_name_chars_atomized": ref_atom_name_chars.unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, n_samples, -1, -1, -1)
        .clone(),
        "asym_id_atomized": torch.tensor(all_asym_ids)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, n_samples, -1)
        .clone(),
        "res_id_atomized": torch.tensor(all_res_ids)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(batch_size, n_samples, -1)
        .clone(),
        "is_protein_atomized": torch.ones(
            batch_size, n_samples, n_atom, dtype=torch.bool
        ),
        "is_rna_atomized": torch.zeros(batch_size, n_samples, n_atom, dtype=torch.bool),
        "is_dna_atomized": torch.zeros(batch_size, n_samples, n_atom, dtype=torch.bool),
        "all_atom_mask": torch.ones(batch_size, n_samples, n_atom),
        "inter_filter_atomized": torch.ones(
            batch_size, n_samples, n_atom, n_atom, dtype=torch.bool
        ),
    }

    return features


class TestSpreadContacts(unittest.TestCase):
    def test_spread_contacts_basic(self):
        """Test that contacts spread to all atoms in the same residue."""
        # 6 atoms: res 1 has atoms 0,1; res 2 has atoms 2,3; res 3 has atoms 4,5
        atom_to_res_id = torch.tensor([1, 1, 2, 2, 3, 3])
        # Only atom 0 (in res 1) and atom 3 (in res 2) have contacts
        is_contact_atom = torch.tensor([True, False, False, True, False, False])

        result = _spread_contacts(is_contact_atom, atom_to_res_id)

        # Expected: All atoms in res 1 and res 2 should be True
        expected = torch.tensor([True, True, True, True, False, False])
        torch.testing.assert_close(result, expected)

    def test_spread_contacts_no_contacts(self):
        """Test with no contacts - all should remain False."""
        atom_to_res_id = torch.tensor([1, 1, 2, 2])
        is_contact_atom = torch.tensor([False, False, False, False])

        result = _spread_contacts(is_contact_atom, atom_to_res_id)

        expected = torch.tensor([False, False, False, False])
        torch.testing.assert_close(result, expected)

    def test_spread_contacts_batch(self):
        """Test batch dimension handling."""
        batch_size, n_samples, n_atoms = 2, 3, 4
        atom_to_res_id = (
            torch.tensor([1, 1, 2, 2]).unsqueeze(0).unsqueeze(0).expand(2, 3, -1)
        )
        is_contact_atom = torch.zeros(batch_size, n_samples, n_atoms, dtype=torch.bool)
        is_contact_atom[0, 0, 0] = True  # batch 0, sample 0, atom 0 (res 1)
        is_contact_atom[1, 2, 3] = True  # batch 1, sample 2, atom 3 (res 2)

        result = _spread_contacts(is_contact_atom, atom_to_res_id)

        # Check shape
        self.assertEqual(result.shape, (batch_size, n_samples, n_atoms))
        # batch 0, sample 0: res 1 atoms (0,1) should be True
        self.assertTrue(result[0, 0, 0].item())
        self.assertTrue(result[0, 0, 1].item())
        # batch 1, sample 2: res 2 atoms (2,3) should be True
        self.assertTrue(result[1, 2, 2].item())
        self.assertTrue(result[1, 2, 3].item())


class TestFnat(unittest.TestCase):
    def test_fnat_perfect(self):
        """Test FNAT=1.0 when all contacts match."""
        contacts_gt = torch.tensor([[True, False], [False, True]])
        contacts_pred = torch.tensor([[True, False], [False, True]])

        result = fnat(contacts_gt, contacts_pred)

        np.testing.assert_allclose(result.item(), 1.0, atol=consts.eps)

    def test_fnat_zero(self):
        """Test FNAT=0.0 when no contacts overlap."""
        contacts_gt = torch.tensor([[True, True], [False, False]])
        contacts_pred = torch.tensor([[False, False], [True, True]])

        result = fnat(contacts_gt, contacts_pred)

        np.testing.assert_allclose(result.item(), 0.0, atol=consts.eps)

    def test_fnat_partial(self):
        """Test partial contact recovery (50%)."""
        # 4 GT contacts
        contacts_gt = torch.zeros(4, 4, dtype=torch.bool)
        contacts_gt[0, 1] = True
        contacts_gt[0, 2] = True
        contacts_gt[1, 2] = True
        contacts_gt[2, 3] = True

        # Prediction recovers 2 of them
        contacts_pred = torch.zeros(4, 4, dtype=torch.bool)
        contacts_pred[0, 1] = True
        contacts_pred[1, 2] = True

        result = fnat(contacts_gt, contacts_pred)

        np.testing.assert_allclose(result.item(), 0.5, atol=consts.eps)

    def test_fnat_no_gt_contacts(self):
        """Test edge case with no GT contacts (should return 0)."""
        contacts_gt = torch.zeros(3, 3, dtype=torch.bool)
        contacts_pred = torch.ones(3, 3, dtype=torch.bool)

        result = fnat(contacts_gt, contacts_pred)

        np.testing.assert_allclose(result.item(), 0.0, atol=consts.eps)


class TestDockQ(unittest.TestCase):
    def test_dockq_perfect_prediction(self):
        """Test DockQ=1.0 when pred_coords == gt_coords."""
        features = build_two_chain_protein_complex(
            chain_a_length=5,
            chain_b_length=5,
            interface_distance=4.0,
        )

        result = dockq(
            pred_coords=features["coords"],
            gt_coords=features["coords"],
            all_atom_mask=features["all_atom_mask"],
            asym_id_atomized=features["asym_id_atomized"],
            res_id_atomized=features["res_id_atomized"],
            ref_atom_name_chars_atomized=features["ref_atom_name_chars_atomized"],
            inter_filter_atomized=features["inter_filter_atomized"],
            is_protein_atomized=features["is_protein_atomized"],
            is_rna_atomized=features["is_rna_atomized"],
            is_dna_atomized=features["is_dna_atomized"],
        )

        # Should have at least one chain pair with contacts
        self.assertGreater(len(result.chain_pair_to_dockq), 0)

        # Perfect prediction should give DockQ = 1.0
        for _chain_pair, dockq_score in result.chain_pair_to_dockq.items():
            np.testing.assert_allclose(dockq_score.item(), 1.0, atol=0.01)

    def test_dockq_no_contacts(self):
        """Test DockQ when chains are far apart (no contacts)."""
        features = build_two_chain_protein_complex(
            chain_a_length=5,
            chain_b_length=5,
            interface_distance=50.0,  # Far apart - no contacts
        )

        result = dockq(
            pred_coords=features["coords"],
            gt_coords=features["coords"],
            all_atom_mask=features["all_atom_mask"],
            asym_id_atomized=features["asym_id_atomized"],
            res_id_atomized=features["res_id_atomized"],
            ref_atom_name_chars_atomized=features["ref_atom_name_chars_atomized"],
            inter_filter_atomized=features["inter_filter_atomized"],
            is_protein_atomized=features["is_protein_atomized"],
            is_rna_atomized=features["is_rna_atomized"],
            is_dna_atomized=features["is_dna_atomized"],
        )

        # No interfaces should be recorded when chains are too far apart
        self.assertEqual(len(result.chain_pair_to_dockq), 0)

    def test_dockq_rototranslation_invariant(self):
        """Test DockQ is invariant to rototranslation of both structures."""
        features = build_two_chain_protein_complex(
            chain_a_length=5,
            chain_b_length=5,
            interface_distance=4.0,
        )

        result_original = dockq(
            pred_coords=features["coords"],
            gt_coords=features["coords"],
            all_atom_mask=features["all_atom_mask"],
            asym_id_atomized=features["asym_id_atomized"],
            res_id_atomized=features["res_id_atomized"],
            ref_atom_name_chars_atomized=features["ref_atom_name_chars_atomized"],
            inter_filter_atomized=features["inter_filter_atomized"],
            is_protein_atomized=features["is_protein_atomized"],
            is_rna_atomized=features["is_rna_atomized"],
            is_dna_atomized=features["is_dna_atomized"],
        )

        # Apply same rototranslation to both gt and pred
        rotated_coords = random_rotation_translation(features["coords"])

        result_rotated = dockq(
            pred_coords=rotated_coords,
            gt_coords=rotated_coords,
            all_atom_mask=features["all_atom_mask"],
            asym_id_atomized=features["asym_id_atomized"],
            res_id_atomized=features["res_id_atomized"],
            ref_atom_name_chars_atomized=features["ref_atom_name_chars_atomized"],
            inter_filter_atomized=features["inter_filter_atomized"],
            is_protein_atomized=features["is_protein_atomized"],
            is_rna_atomized=features["is_rna_atomized"],
            is_dna_atomized=features["is_dna_atomized"],
        )

        # DockQ should be the same after rototranslation
        for chain_pair in result_original.chain_pair_to_dockq:
            np.testing.assert_allclose(
                result_original.chain_pair_to_dockq[chain_pair].item(),
                result_rotated.chain_pair_to_dockq[chain_pair].item(),
                atol=0.01,
            )

    def test_dockq_molecule_types(self):
        """Test that molecule types are correctly identified."""
        features = build_two_chain_protein_complex(
            chain_a_length=5,
            chain_b_length=5,
            interface_distance=4.0,
        )

        result = dockq(
            pred_coords=features["coords"],
            gt_coords=features["coords"],
            all_atom_mask=features["all_atom_mask"],
            asym_id_atomized=features["asym_id_atomized"],
            res_id_atomized=features["res_id_atomized"],
            ref_atom_name_chars_atomized=features["ref_atom_name_chars_atomized"],
            inter_filter_atomized=features["inter_filter_atomized"],
            is_protein_atomized=features["is_protein_atomized"],
            is_rna_atomized=features["is_rna_atomized"],
            is_dna_atomized=features["is_dna_atomized"],
        )

        # All chains are protein
        for _chain_pair, moltype in result.chain_pair_to_moltype.items():
            self.assertEqual(moltype, ("protein", "protein"))


class TestDockQFullComplex(unittest.TestCase):
    def test_dockq_full_complex_single_interface(self):
        """Test dockq_full_complex with a single interface."""
        features = build_two_chain_protein_complex(
            chain_a_length=5,
            chain_b_length=5,
            interface_distance=4.0,
        )

        result = dockq_full_complex(
            pred_coords=features["coords"],
            gt_coords=features["coords"],
            all_atom_mask=features["all_atom_mask"],
            asym_id_atomized=features["asym_id_atomized"],
            res_id_atomized=features["res_id_atomized"],
            ref_atom_name_chars_atomized=features["ref_atom_name_chars_atomized"],
            inter_filter_atomized=features["inter_filter_atomized"],
            is_protein_atomized=features["is_protein_atomized"],
            is_rna_atomized=features["is_rna_atomized"],
            is_dna_atomized=features["is_dna_atomized"],
        )

        # Should have protein-protein metrics
        self.assertIn("dockq_protein_protein_uw", result)
        self.assertIn("dockq_protein_protein_w", result)

        # Perfect prediction: both should be 1.0
        np.testing.assert_allclose(
            result["dockq_protein_protein_uw"].mean().item(), 1.0, atol=0.01
        )
        np.testing.assert_allclose(
            result["dockq_protein_protein_w"].mean().item(), 1.0, atol=0.01
        )

    def test_dockq_full_complex_output_shape(self):
        """Test output shape is [B, S]."""
        n_samples = 3
        features = build_two_chain_protein_complex(
            chain_a_length=5,
            chain_b_length=5,
            interface_distance=4.0,
            batch_size=1,
            n_samples=n_samples,
        )

        result = dockq_full_complex(
            pred_coords=features["coords"],
            gt_coords=features["coords"],
            all_atom_mask=features["all_atom_mask"],
            asym_id_atomized=features["asym_id_atomized"],
            res_id_atomized=features["res_id_atomized"],
            ref_atom_name_chars_atomized=features["ref_atom_name_chars_atomized"],
            inter_filter_atomized=features["inter_filter_atomized"],
            is_protein_atomized=features["is_protein_atomized"],
            is_rna_atomized=features["is_rna_atomized"],
            is_dna_atomized=features["is_dna_atomized"],
        )

        # All outputs should have shape [B, S]
        for key, value in result.items():
            self.assertEqual(value.shape, (1, n_samples), f"{key} has wrong shape")

    def test_dockq_full_complex_no_interfaces(self):
        """Test dockq_full_complex when no interfaces exist."""
        features = build_two_chain_protein_complex(
            chain_a_length=5,
            chain_b_length=5,
            interface_distance=50.0,  # Far apart
        )

        result = dockq_full_complex(
            pred_coords=features["coords"],
            gt_coords=features["coords"],
            all_atom_mask=features["all_atom_mask"],
            asym_id_atomized=features["asym_id_atomized"],
            res_id_atomized=features["res_id_atomized"],
            ref_atom_name_chars_atomized=features["ref_atom_name_chars_atomized"],
            inter_filter_atomized=features["inter_filter_atomized"],
            is_protein_atomized=features["is_protein_atomized"],
            is_rna_atomized=features["is_rna_atomized"],
            is_dna_atomized=features["is_dna_atomized"],
        )

        # Should return empty dict
        self.assertEqual(len(result), 0)


class TestGetAbAgMetrics(unittest.TestCase):
    def test_get_ab_ag_metrics_keys(self):
        """Test that expected output keys are present."""
        n_atom = 30
        batch_size, n_samples = 1, 1

        # Create AB-AG type masks: 10 atoms each for AB-H (1), AB-L (2), AG (3)
        intra_ab_ag_type = torch.cat(
            [
                torch.ones(10) * 1,  # AB-H
                torch.ones(10) * 2,  # AB-L
                torch.ones(10) * 3,  # AG
            ]
        ).view(batch_size, n_samples, n_atom)

        asym_id = torch.cat(
            [
                torch.ones(10) * 1,
                torch.ones(10) * 2,
                torch.ones(10) * 3,
            ]
        ).view(batch_size, n_samples, n_atom)

        gt_coords = torch.randn(batch_size, n_samples, n_atom, 3)
        pred_coords = gt_coords.clone()
        all_atom_mask = torch.ones(batch_size, n_samples, n_atom)

        # Inter type mask (simplified - just zeros for now)
        inter_ab_ag_type = torch.zeros(batch_size, n_samples, n_atom, n_atom)

        result = get_ab_ag_metrics(
            intra_ab_ag_type_atomized=intra_ab_ag_type,
            intra_ab_ag_type_atomized_filtered=intra_ab_ag_type.clone(),
            inter_ab_ag_type_atomized_filtered=inter_ab_ag_type,
            gt_coords=gt_coords,
            pred_coords=pred_coords,
            all_atom_mask=all_atom_mask,
            asym_id_atomized=asym_id,
        )

        # Check intra-chain keys
        for chain_type in AB_AG_CHAIN_TYPES:
            self.assertIn(f"lddt_intra_{chain_type}", result)

    def test_get_ab_ag_metrics_perfect(self):
        """Test lDDT=1.0 for identical coordinates."""
        n_atom = 30
        batch_size, n_samples = 1, 1

        intra_ab_ag_type = torch.cat(
            [
                torch.ones(10) * 1,
                torch.ones(10) * 2,
                torch.ones(10) * 3,
            ]
        ).view(batch_size, n_samples, n_atom)

        asym_id = torch.cat(
            [
                torch.ones(10) * 1,
                torch.ones(10) * 2,
                torch.ones(10) * 3,
            ]
        ).view(batch_size, n_samples, n_atom)

        gt_coords = torch.randn(batch_size, n_samples, n_atom, 3)
        pred_coords = gt_coords.clone()  # Identical
        all_atom_mask = torch.ones(batch_size, n_samples, n_atom)
        inter_ab_ag_type = torch.zeros(batch_size, n_samples, n_atom, n_atom)

        result = get_ab_ag_metrics(
            intra_ab_ag_type_atomized=intra_ab_ag_type,
            intra_ab_ag_type_atomized_filtered=intra_ab_ag_type.clone(),
            inter_ab_ag_type_atomized_filtered=inter_ab_ag_type,
            gt_coords=gt_coords,
            pred_coords=pred_coords,
            all_atom_mask=all_atom_mask,
            asym_id_atomized=asym_id,
        )

        # All intra-chain lDDT should be 1.0 for identical coords
        for key, value in result.items():
            if "intra" in key:
                np.testing.assert_allclose(value.item(), 1.0, atol=consts.eps)


if __name__ == "__main__":
    unittest.main()
