import math
import unittest

import numpy as np
import torch

from openfold3.core.metrics.validation import gdt_ha, gdt_ts, rmsd
from openfold3.core.metrics.validation_all_atom import (
    drmsd,
    get_superimpose_metrics,
    interface_lddt,
    lddt,
)
from openfold3.core.utils.geometry.kabsch_alignment import (
    apply_transformation,
    get_optimal_transformation,
    kabsch_align,
)
from tests.config import consts


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
        intra_mask_filter = torch.ones((batch_size, n_atom))  # [batch_size, n_atom]
        inter_mask_filter = torch.ones(
            (batch_size, n_atom, n_atom)
        )  # [batch_size, n_atom, n_atom]

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
        filter_mask = torch.ones(batch_size, n_atom, n_atom2)

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


if __name__ == "__main__":
    unittest.main()
