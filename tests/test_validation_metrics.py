import math
import unittest

import numpy as np
import torch

from openfold3.core.utils.validation_metrics import (
    batched_kabsch,
    drmsd,
    gdt_ha,
    gdt_ts,
    interface_lddt,
    lddt,
)
from tests.config import consts


def random_rotation_translation(structure, factor = 100.):
    """ 
    Applies random rotations and translations to a given structure
    """ 
    # rotation: Rx, Ry, Rz
    x_angle, y_angle, z_angle = torch.randn(3) * 2 * math.pi
    x_rotation = torch.tensor([[1.,0.,0.], 
                               [0., math.cos(x_angle), -math.sin(x_angle)], 
                               [0., math.sin(x_angle), math.cos(x_angle)]]
                               ).to(torch.float32)
    y_rotation = torch.tensor([[math.cos(y_angle),0., math.sin(y_angle)],
                               [0., 1., 0.], 
                               [-math.sin(y_angle), 0., math.cos(y_angle)]]
                               ).to(torch.float32)
    z_rotation = torch.tensor([[math.cos(z_angle),-math.sin(z_angle), 0.],
                               [math.sin(z_angle), math.cos(z_angle), 0.], 
                               [0., 0., 1.]]
                               ).to(torch.float32)
    xyz_rotation = (x_rotation @ y_rotation @ z_rotation)
    
    #2. translation
    translation = torch.randn(size = structure.shape[:-2] + (1, 3,)) * factor
    translation = translation.to(torch.float32)
    new_structure = structure @ xyz_rotation + translation
    return new_structure

class TestLDDT(unittest.TestCase):
    def test_lddt(self):
        batch_size = consts.batch_size 
        n_atom = consts.n_res

        gt_structure = torch.randn(batch_size, n_atom, 3)
        predicted_structure = torch.randn(batch_size, n_atom, 3)
        mask = torch.ones(batch_size, n_atom)

        #shape test
        out_lddt = lddt(gt_structure, predicted_structure, mask)
        exp_shape = (batch_size,)
        np.testing.assert_equal(out_lddt.shape, exp_shape)

        #rototranslation. lddt should give 1.0s 
        gt_structure_rototranslated = random_rotation_translation(gt_structure)
        exp_outputs = torch.ones(batch_size)
        np.testing.assert_allclose(lddt(gt_structure_rototranslated, 
                                        gt_structure, 
                                        mask), 
                                        exp_outputs, 
                                        atol = 1e-5
                                        )

class TestInterfaceLDDT(unittest.TestCase):
    def test_interface_lddt(self):
        batch_size = consts.batch_size 
        n_atom = consts.n_res
        n_atom2 = 5
        gt_structure_1 = torch.randn(batch_size, n_atom, 3)
        gt_structure_2 = torch.randn(batch_size, n_atom2, 3)
        predicted_structure_1 = torch.randn(batch_size, n_atom, 3)
        predicted_structure_2 = torch.randn(batch_size, n_atom2, 3)
        mask1 = torch.ones(batch_size, n_atom)
        mask2 = torch.ones(batch_size, n_atom2)

        #shape test
        out_interface_lddt = interface_lddt(predicted_structure_1, 
                                            predicted_structure_2, 
                                            gt_structure_1, 
                                            gt_structure_2, 
                                            mask1, 
                                            mask2,
                                            )
        exp_shape = (batch_size,)
        np.testing.assert_equal(out_interface_lddt.shape, exp_shape)

        #rototranslation test
        #should give 1.s 

        #combine coordinates and rototranslate them
        predicted_coordinates = torch.cat((gt_structure_1, gt_structure_2), dim = 1)
        combined_coordinates = random_rotation_translation(predicted_coordinates)
        #split two molecules
        p1, p2 = torch.split(combined_coordinates, 
                             [n_atom, n_atom2], 
                             dim = 1
                             )
        #run interface_lddt
        out_interface_lddt = interface_lddt(p1, 
                                            p2, 
                                            gt_structure_1, 
                                            gt_structure_2, 
                                            mask1, 
                                            mask2
                                            )        
        exp_outputs = torch.ones(batch_size)
        np.testing.assert_allclose(out_interface_lddt, exp_outputs, atol = 1e-5)

class TestDRMSD(unittest.TestCase):
    def test_drmsd(self):
        batch_size = consts.batch_size 
        n_atom = consts.n_res

        gt_structure = torch.randn(batch_size, n_atom, 3)
        predicted_structure = torch.randn(batch_size, n_atom, 3)
        mask = torch.ones(batch_size, n_atom)

        #shape test
        out_drmsd = drmsd(gt_structure, 
                         predicted_structure, 
                         mask
                         )
        exp_shape = (batch_size,)
        np.testing.assert_equal(out_drmsd.shape, exp_shape)

        #rototranslation. should give 0.s 
        gt_structure_rototranslated = random_rotation_translation(gt_structure)
        exp_outputs = torch.zeros(batch_size)
        out_drmsd = drmsd(gt_structure_rototranslated, 
                          gt_structure, 
                          mask,
                          )
        np.testing.assert_allclose(out_drmsd, exp_outputs, atol = 1e-5)


class TestBatchedKabsch(unittest.TestCase):
    def test_batched_kabsch(self):
        batch_size = consts.batch_size 
        n_atom = consts.n_res

        gt_structure = torch.randn(batch_size, n_atom, 3)
        pred_structure = torch.randn(batch_size, n_atom, 3)
        mask = torch.ones(batch_size, n_atom)

        #shape test
        out_translation, out_rotation, out_coordinates = batched_kabsch(pred_structure,
                                                                        gt_structure, 
                                                                        mask,
                                                                        )
        exp_shape_translation = (batch_size, 1, 3)
        exp_shape_rotation = (batch_size, 3, 3)
        exp_shape_coordinates = (batch_size,)
        np.testing.assert_equal(out_translation.shape, exp_shape_translation)
        np.testing.assert_equal(out_rotation.shape, exp_shape_rotation)
        np.testing.assert_equal(out_coordinates.shape, exp_shape_coordinates)

        #rototranslation test. should give 0.s 
        gt_structure_rototranslated = random_rotation_translation(gt_structure)
        exp_outputs = torch.zeros(batch_size)
        out_kabsch = batched_kabsch(gt_structure_rototranslated, gt_structure, mask)[-1]
        np.testing.assert_allclose(out_kabsch, exp_outputs, atol = 1e-5)

class TestGDT(unittest.TestCase):
    def test_gdt(self):
        batch_size = consts.batch_size 
        n_atom = consts.n_res

        gt_structure = torch.randn(batch_size, n_atom, 3)
        predicted_structure = torch.randn(batch_size, n_atom, 3)
        mask = torch.ones(batch_size, n_atom)

        #shape test
        trans, optimal_rotation, rmsd = batched_kabsch(predicted_structure, 
                                                       gt_structure,
                                                       mask,
                                                       )
        
        gt_centered = gt_structure - torch.mean(gt_structure, 
                                                dim = -2, 
                                                keepdim = True
                                                )
        pred_centered = predicted_structure - torch.mean(predicted_structure, 
                                                         dim = -2, 
                                                         keepdim = True
                                                         )
        pred_superimposed = pred_centered @ optimal_rotation.transpose(-1, -2)
        out_gdt_ts = gdt_ts(pred_superimposed, 
                            gt_centered, 
                            mask
                            )
        out_gdt_ha = gdt_ha(pred_superimposed, 
                            gt_centered, 
                            mask
                            )

        exp_gdt_ts_shape = (batch_size,)
        exp_gdt_ha_shape = (batch_size,)
        np.testing.assert_equal(out_gdt_ts.shape, exp_gdt_ts_shape)
        np.testing.assert_equal(out_gdt_ha.shape, exp_gdt_ha_shape)

        #rototranslation test
        gt_structure = torch.randn(batch_size, n_atom, 3)
        gt_structure_centered = gt_structure - torch.mean(gt_structure, 
                                                          dim = -2, 
                                                          keepdim = True
                                                          )

        mask = torch.ones(batch_size, n_atom)        
        pred = random_rotation_translation(gt_structure)
        trans, optimal_rotation, rmsd = batched_kabsch(pred, 
                                                       gt_structure,
                                                       mask,
                                                       )
        pred_centered = pred - torch.mean(pred,
                                          dim = -2, 
                                          keepdim = True,
                                          )
        pred_superimposed = pred_centered @ optimal_rotation.transpose(-1, -2)
        out_gdt_ts = gdt_ts(pred_superimposed, 
                            gt_structure_centered, 
                            mask
                            )
        out_gdt_ha = gdt_ha(pred_superimposed, 
                            gt_structure_centered, 
                            mask
                            )

        exp_gdt_ts_outs = torch.ones(batch_size)
        exp_gdt_ha_outs = torch.ones(batch_size)

        np.testing.assert_allclose(out_gdt_ts, exp_gdt_ts_outs, atol = 1e-5)
        np.testing.assert_allclose(out_gdt_ha, exp_gdt_ha_outs, atol = 1e-5)

if __name__ == "__main__":
    unittest.main()