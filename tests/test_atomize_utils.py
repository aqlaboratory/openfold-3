import unittest

import torch
import torch.nn.functional as F

from openfold3.core.utils.atomize_utils import (
    broadcast_token_feat_to_atoms,
    get_token_atom_index_offset,
    get_token_center_atoms,
    get_token_frame_atoms,
    get_token_representative_atoms,
)


def example1():
    # Standard amino acid residues
    # Protein: ALA GLY
    # NumAtoms: 5 4
    restype = F.one_hot(torch.Tensor([[0, 7]]).long(), num_classes=32).float()

    batch = {
        "restype": restype,
        "asym_id": torch.Tensor([0, 0]),
        "token_mask": torch.ones((1, 2)),
        "is_protein": torch.ones((1, 2)),
        "is_dna": torch.zeros((1, 2)),
        "is_rna": torch.zeros((1, 2)),
        "is_atomized": torch.zeros((1, 2)),
        "start_atom_index": torch.Tensor([[0, 5]]),
        "num_atoms_per_token": torch.Tensor([5, 4]),
    }

    x = torch.randn((1, 9, 3))
    atom_mask = torch.ones((1, 9))

    return batch, x, atom_mask


def example2():
    # Modified amino acid residues
    # Protein: ALA GLY/A
    # NumAtoms: 5 4
    restype = F.one_hot(torch.Tensor([[0, 7, 7, 7, 7]]).long(), num_classes=32).float()

    batch = {
        "restype": restype,
        "token_mask": torch.ones((1, 5)),
        "is_protein": torch.ones((1, 5)),
        "is_dna": torch.zeros((1, 5)),
        "is_rna": torch.zeros((1, 5)),
        "is_atomized": torch.Tensor([[0, 1, 1, 1, 1]]),
        "start_atom_index": torch.Tensor([[0, 5, 6, 7, 8]]),
    }

    x = torch.randn((1, 9, 3))
    atom_mask = torch.ones((1, 9))

    return batch, x, atom_mask


def example3():
    # Standard nucleotide residues
    # Protein 1: A U
    # Protein 2: DG DC
    # NumAtoms 1: 22 20
    # NumAtoms 2: 23 20
    restype = F.one_hot(
        torch.Tensor([[21, 24], [26, 27]]).long(), num_classes=32
    ).float()

    batch = {
        "restype": restype,
        "asym_id": torch.Tensor([[0, 0], [0, 1]]),
        "token_mask": torch.ones((2, 2)),
        "is_protein": torch.zeros((2, 2)),
        "is_dna": torch.Tensor([[0, 0], [1, 1]]),
        "is_rna": torch.Tensor([[1, 1], [0, 0]]),
        "is_atomized": torch.zeros((2, 2)),
        "start_atom_index": torch.Tensor([[0, 22], [0, 23]]),
        "num_atoms_per_token": torch.Tensor([[22, 20], [23, 20]]),
    }

    x = torch.randn((2, 43, 3))
    atom_mask = torch.ones((2, 43))
    atom_mask[0, -1] = 0

    return batch, x, atom_mask


def example4():
    # Modified nucleotide residues
    # Protein 1: A U/A
    # Protein 2: DG/A DC
    # NumAtoms 1: 22 20
    # NumAtoms 2: 23 20
    token_mask = torch.Tensor([[1] * 21 + [0] * 3, [1] * 24])
    restype = (
        F.one_hot(
            torch.Tensor([[21] + [24] * 20 + [31] * 3, [26] * 23 + [27]]).long(),
            num_classes=32,
        ).float()
        * token_mask[..., None]
    )

    batch = {
        "restype": restype,
        "token_mask": token_mask,
        "is_protein": torch.zeros((2, 24)),
        "is_rna": torch.Tensor([[1] * 21 + [0] * 3, [0] * 24]),
        "is_dna": torch.Tensor([[0] * 24, [1] * 24]),
        "is_atomized": torch.Tensor([[0] + [1] * 20 + [0] * 3, [1] * 23 + [0]]),
        "start_atom_index": torch.Tensor(
            [[0] + [i for i in range(22, 42)] + [0] * 3, [i for i in range(24)]]
        ),
    }

    x = torch.randn((2, 43, 3))
    atom_mask = torch.ones((2, 43))
    atom_mask[0, -1] = 0

    return batch, x, atom_mask


def example5():
    # Ligands
    # Ligand 1 + GLY (4 atoms)
    # Ligand 2 + A (22 atoms)
    # Ligand 3 + DG (23 atoms)
    # Ligand 4
    token_mask = torch.ones((4, 4))
    token_mask[-1, -1] = 0
    restype = (
        F.one_hot(
            torch.Tensor(
                [
                    [20, 20, 20, 7],
                    [20, 20, 20, 21],
                    [20, 20, 20, 26],
                    [20, 20, 20, 31],
                ]
            ).long(),
            num_classes=32,
        ).float()
        * token_mask[..., None]
    )

    is_protein = torch.zeros((4, 4))
    is_protein[0, -1] = 1
    is_rna = torch.zeros((4, 4))
    is_rna[1, -1] = 1
    is_dna = torch.zeros((4, 4))
    is_dna[2, -1] = 1

    is_atomized = torch.concat([torch.ones((4, 3)), torch.zeros((4, 1))], dim=-1)

    start_atom_index = torch.arange(4).unsqueeze(0).repeat((4, 1))
    start_atom_index = start_atom_index * token_mask

    batch = {
        "restype": restype,
        "token_mask": token_mask,
        "is_protein": is_protein,
        "is_dna": is_dna,
        "is_rna": is_rna,
        "is_atomized": is_atomized,
        "start_atom_index": start_atom_index,
        "num_atoms_per_token": torch.Tensor(
            [[1, 1, 1, 4], [1, 1, 1, 22], [1, 1, 1, 23], [1, 1, 1, 0]]
        ),
        "asym_id": torch.Tensor(
            [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0]]
        ),
    }

    # Example 1: Ligand atom with valid frames
    ligand1 = torch.Tensor([[0, 0, 0], [1, 0, 0], [0, 2, 0]])

    # Example 2: Ligand atom with failed angle constraint (outward colinear)
    ligand2 = torch.Tensor([[0, 0, 0], [1, 0, 0], [-1, 0.1, 0]])

    # Example 3: Ligand atom with failed angle constraint (inward colinear)
    ligand3 = torch.Tensor([[0, 0, 0], [1, 0, 0], [0.9, 0.1, 0]])

    # Example 4: Ligand atom with failed chain constraint
    ligand4 = torch.Tensor([[0, 0, 0], [1, 0, 0], [0, 2, 0]])

    ligands = torch.stack([ligand1, ligand2, ligand3, ligand4], dim=0)
    x = torch.concat([ligands, torch.randn((4, 23, 3))], dim=1)

    atom_mask = torch.Tensor(
        [[1] * 7 + [0] * 19, [1] * 25 + [0], [1] * 26, [1] * 3 + [0] * 23]
    )

    return batch, x, atom_mask


class TestBroadcastTokenFeatToAtoms(unittest.TestCase):
    def test_with_one_batch_dim(self):
        num_atoms_per_token = torch.Tensor([[3, 6, 2, 5, 1], [4, 7, 1, 3, 5]])

        token_mask = torch.Tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])

        atom_mask = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=token_mask,
        )

        num_atoms = torch.sum(atom_mask, dim=-1).int()
        gt_num_atoms = torch.sum(num_atoms_per_token * token_mask, dim=-1).int()

        self.assertTrue(atom_mask.shape == (2, gt_num_atoms[0]))
        self.assertTrue((num_atoms == gt_num_atoms).all())

    def test_with_two_batch_dim(self):
        num_atoms_per_token = torch.Tensor(
            [[3, 6, 2, 5, 1], [4, 7, 1, 3, 5]]
        ).unsqueeze(1)

        token_mask = torch.Tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]).unsqueeze(1)

        atom_mask = broadcast_token_feat_to_atoms(
            token_mask=token_mask,
            num_atoms_per_token=num_atoms_per_token,
            token_feat=token_mask,
        )

        num_atoms = torch.sum(atom_mask, dim=-1).int()
        gt_num_atoms = torch.sum(num_atoms_per_token * token_mask, dim=-1).int()

        self.assertTrue(atom_mask.shape == (2, 1, gt_num_atoms[0]))
        self.assertTrue((num_atoms == gt_num_atoms).all())


class TestGetTokenAtomIndex(unittest.TestCase):
    def test_with_amino_acid_backbone_residue(self):
        restype = F.one_hot(
            torch.Tensor([[0, 7, 13, 20], [21, 25, 24, 30]]).long(), num_classes=32
        ).float()
        atom_index_offset = get_token_atom_index_offset(
            atom_name="CA", restype=restype
        ).int()
        gt_atom_index_offset = torch.Tensor([[1, 1, 1, -1], [-1, -1, -1, -1]])
        self.assertTrue((atom_index_offset == gt_atom_index_offset).all())

    def test_with_amino_acid_sidechain_residue(self):
        restype = F.one_hot(
            torch.Tensor([[0, 7, 13, 20], [21, 25, 24, 30]]).long(), num_classes=32
        ).float()
        atom_index_offset = get_token_atom_index_offset(
            atom_name="CB", restype=restype
        ).int()
        gt_atom_index_offset = torch.Tensor([[4, -1, 4, -1], [-1, -1, -1, -1]])
        self.assertTrue((atom_index_offset == gt_atom_index_offset).all())

    def test_with_nucleotide_backbone_residue(self):
        restype = F.one_hot(
            torch.Tensor([[0, 7, 13, 20], [21, 25, 24, 30]]).long(), num_classes=32
        ).float()
        atom_index_offset = get_token_atom_index_offset(
            atom_name="C3'", restype=restype
        ).int()
        gt_atom_index_offset = torch.Tensor([[-1, -1, -1, -1], [7, 7, 7, -1]])
        self.assertTrue((atom_index_offset == gt_atom_index_offset).all())

    def test_with_nucleotide_sidechain_residue(self):
        restype = F.one_hot(
            torch.Tensor([[0, 7, 13, 20], [21, 25, 24, 30]]).long(), num_classes=32
        ).float()
        atom_index_offset = get_token_atom_index_offset(
            atom_name="C4", restype=restype
        ).int()
        gt_atom_index_offset = torch.Tensor([[-1, -1, -1, -1], [21, 21, 16, -1]])
        self.assertTrue((atom_index_offset == gt_atom_index_offset).all())


class TestGetTokenCenterAtom(unittest.TestCase):
    def test_with_standard_amino_acid_residues(self):
        batch, x, atom_mask = example1()

        center_x, center_atom_mask = get_token_center_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_center_x = x[:, [1, 6], :]
        gt_center_atom_mask = torch.Tensor([1, 1])
        self.assertTrue((torch.abs(center_x - gt_center_x) < 1e-5).all())
        self.assertTrue((center_atom_mask == gt_center_atom_mask).all())

        atom_mask[0, 6] = 0

        center_x, center_atom_mask = get_token_center_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_center_x = x[:, [1, 6], :]
        gt_center_atom_mask = torch.Tensor([1, 0])
        self.assertTrue((torch.abs(center_x - gt_center_x) < 1e-5).all())
        self.assertTrue((center_atom_mask == gt_center_atom_mask).all())

    def test_with_modified_amino_acid_residues(self):
        batch, x, atom_mask = example2()

        center_x, center_atom_mask = get_token_center_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_center_x = x[:, [1, 5, 6, 7, 8], :]

        gt_center_atom_mask = torch.Tensor([1, 1, 1, 1, 1])
        self.assertTrue((torch.abs(center_x - gt_center_x) < 1e-5).all())
        self.assertTrue((center_atom_mask == gt_center_atom_mask).all())

        atom_mask[0, 8] = 0

        center_x, center_atom_mask = get_token_center_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_center_atom_mask = torch.Tensor([1, 1, 1, 1, 0])
        self.assertTrue((torch.abs(center_x - gt_center_x) < 1e-5).all())
        self.assertTrue((center_atom_mask == gt_center_atom_mask).all())

    def test_with_standard_nucleotide_residues(self):
        batch, x, atom_mask = example3()

        center_x, center_atom_mask = get_token_center_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_center_x = torch.stack([x[0, [11, 33], :], x[1, [11, 34], :]], dim=0)

        gt_center_atom_mask = torch.Tensor([[1, 1], [1, 1]])
        self.assertTrue((torch.abs(center_x - gt_center_x) < 1e-5).all())
        self.assertTrue((center_atom_mask == gt_center_atom_mask).all())

        atom_mask[0, 11] = 0

        center_x, center_atom_mask = get_token_center_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_center_atom_mask = torch.Tensor([[0, 1], [1, 1]])
        self.assertTrue((torch.abs(center_x - gt_center_x) < 1e-5).all())
        self.assertTrue((center_atom_mask == gt_center_atom_mask).all())

    def test_with_modified_nucleotide_residues(self):
        batch, x, atom_mask = example4()

        center_x, center_atom_mask = get_token_center_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_center_x = torch.stack(
            [
                x[0, [11] + [i for i in range(22, 42)] + [0] * 3, :],
                x[1, [i for i in range(23)] + [34], :],
            ],
            dim=0,
        )

        gt_center_atom_mask = torch.Tensor([[1] * 21 + [0] * 3, [1] * 24])
        self.assertTrue((torch.abs(center_x - gt_center_x) < 1e-5).all())
        self.assertTrue((center_atom_mask == gt_center_atom_mask).all())

        atom_mask[1, 11] = 0

        center_x, center_atom_mask = get_token_center_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_center_atom_mask[1, 11] = 0
        self.assertTrue((torch.abs(center_x - gt_center_x) < 1e-5).all())
        self.assertTrue((center_atom_mask == gt_center_atom_mask).all())

    def test_with_ligands(self):
        batch, x, atom_mask = example5()

        center_x, center_atom_mask = get_token_center_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_center_x = torch.stack(
            [
                x[0, [0, 1, 2, 4], :],
                x[1, [0, 1, 2, 14], :],
                x[2, [0, 1, 2, 14], :],
                x[3, [0, 1, 2, 0], :],
            ],
            dim=0,
        )

        gt_center_atom_mask = torch.ones((4, 4))
        gt_center_atom_mask[-1, -1] = 0
        self.assertTrue((torch.abs(center_x - gt_center_x) < 1e-5).all())
        self.assertTrue((center_atom_mask == gt_center_atom_mask).all())

        atom_mask[0, 4] = 0

        center_x, center_atom_mask = get_token_center_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_center_atom_mask[0, -1] = 0
        self.assertTrue((torch.abs(center_x - gt_center_x) < 1e-5).all())
        self.assertTrue((center_atom_mask == gt_center_atom_mask).all())


class TestGetTokenRepresentativeAtom(unittest.TestCase):
    def test_with_standard_amino_acid_residues(self):
        batch, x, atom_mask = example1()

        rep_x, rep_atom_mask = get_token_representative_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_rep_x = x[:, [4, 6], :]
        gt_rep_atom_mask = torch.Tensor([1, 1])
        self.assertTrue((torch.abs(rep_x - gt_rep_x) < 1e-5).all())
        self.assertTrue((rep_atom_mask == gt_rep_atom_mask).all())

        atom_mask[0, 4] = 0

        rep_x, rep_atom_mask = get_token_representative_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_rep_atom_mask = torch.Tensor([0, 1])
        self.assertTrue((torch.abs(rep_x - gt_rep_x) < 1e-5).all())
        self.assertTrue((rep_atom_mask == gt_rep_atom_mask).all())

    def test_with_modified_amino_acid_residues(self):
        batch, x, atom_mask = example2()

        rep_x, rep_atom_mask = get_token_representative_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_rep_x = x[:, [4, 5, 6, 7, 8], :]
        gt_rep_atom_mask = torch.Tensor([1, 1, 1, 1, 1])
        self.assertTrue((torch.abs(rep_x - gt_rep_x) < 1e-5).all())
        self.assertTrue((rep_atom_mask == gt_rep_atom_mask).all())

        atom_mask[0, 7] = 0

        rep_x, rep_atom_mask = get_token_representative_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_rep_atom_mask = torch.Tensor([1, 1, 1, 0, 1])
        self.assertTrue((torch.abs(rep_x - gt_rep_x) < 1e-5).all())
        self.assertTrue((rep_atom_mask == gt_rep_atom_mask).all())

    def test_with_standard_nucleotide_residues(self):
        batch, x, atom_mask = example3()

        rep_x, rep_atom_mask = get_token_representative_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_rep_x = torch.stack([x[0, [21, 35], :], x[1, [22, 36], :]], dim=0)
        gt_rep_atom_mask = torch.ones((2, 2))

        self.assertTrue((torch.abs(rep_x - gt_rep_x) < 1e-5).all())
        self.assertTrue((rep_atom_mask == gt_rep_atom_mask).all())

        atom_mask[1, 35] = 0

        rep_x, rep_atom_mask = get_token_representative_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        self.assertTrue((torch.abs(rep_x - gt_rep_x) < 1e-5).all())
        self.assertTrue((rep_atom_mask == gt_rep_atom_mask).all())

    def test_with_modified_nucleotide_residues(self):
        batch, x, atom_mask = example4()

        rep_x, rep_atom_mask = get_token_representative_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_rep_x = torch.stack(
            [
                x[0, [21] + [i for i in range(22, 42)] + [0] * 3, :],
                x[1, [i for i in range(23)] + [36], :],
            ],
            dim=0,
        )
        gt_rep_atom_mask = torch.Tensor([[1] * 21 + [0] * 3, [1] * 24])

        self.assertTrue((torch.abs(rep_x - gt_rep_x) < 1e-5).all())
        self.assertTrue((rep_atom_mask == gt_rep_atom_mask).all())

        atom_mask[0, 26] = 0

        rep_x, rep_atom_mask = get_token_representative_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_rep_atom_mask[0, 5] = 0
        self.assertTrue((torch.abs(rep_x - gt_rep_x) < 1e-5).all())
        self.assertTrue((rep_atom_mask == gt_rep_atom_mask).all())

    def test_with_ligands(self):
        batch, x, atom_mask = example5()

        rep_x, rep_atom_mask = get_token_representative_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_rep_x = torch.stack(
            [
                x[0, [0, 1, 2, 4], :],
                x[1, [0, 1, 2, 24], :],
                x[2, [0, 1, 2, 25], :],
                x[3, [0, 1, 2, 0], :],
            ],
            dim=0,
        )
        gt_rep_atom_mask = torch.ones((4, 4))
        gt_rep_atom_mask[-1, -1] = 0

        self.assertTrue((torch.abs(rep_x - gt_rep_x) < 1e-5).all())
        self.assertTrue((rep_atom_mask == gt_rep_atom_mask).all())

        atom_mask[2, 1] = 0

        rep_x, rep_atom_mask = get_token_representative_atoms(
            batch=batch, x=x, atom_mask=atom_mask
        )

        gt_rep_atom_mask[2, 1] = 0
        self.assertTrue((torch.abs(rep_x - gt_rep_x) < 1e-5).all())
        self.assertTrue((rep_atom_mask == gt_rep_atom_mask).all())


class TestGetTokenFrameAtom(unittest.TestCase):
    def test_with_standard_amino_acid_residues(self):
        angle_threshold = 25.0
        eps = 1e-8
        inf = 1e9
        batch, x, atom_mask = example1()

        (a, b, c), valid_frame_mask = get_token_frame_atoms(
            batch=batch,
            x=x,
            atom_mask=atom_mask,
            angle_threshold=angle_threshold,
            eps=eps,
            inf=inf,
        )

        gt_a = x[:, [0, 5], :]
        gt_b = x[:, [1, 6], :]
        gt_c = x[:, [2, 7], :]
        gt_valid_frame_mask = torch.Tensor([1, 1])

        self.assertTrue((torch.abs(a - gt_a) < 1e-5).all())
        self.assertTrue((torch.abs(b - gt_b) < 1e-5).all())
        self.assertTrue((torch.abs(c - gt_c) < 1e-5).all())
        self.assertTrue((valid_frame_mask == gt_valid_frame_mask).all())

        atom_mask[0, 7] = 0

        (a, b, c), valid_frame_mask = get_token_frame_atoms(
            batch=batch,
            x=x,
            atom_mask=atom_mask,
            angle_threshold=angle_threshold,
            eps=eps,
            inf=inf,
        )

        gt_valid_frame_mask = torch.Tensor([1, 0])

        self.assertTrue((torch.abs(a - gt_a) < 1e-5).all())
        self.assertTrue((torch.abs(b - gt_b) < 1e-5).all())
        self.assertTrue((torch.abs(c - gt_c) < 1e-5).all())
        self.assertTrue((valid_frame_mask == gt_valid_frame_mask).all())

    def test_with_standard_nucleotide_residues(self):
        angle_threshold = 25.0
        eps = 1e-8
        inf = 1e9
        batch, x, atom_mask = example3()

        (a, b, c), valid_frame_mask = get_token_frame_atoms(
            batch=batch,
            x=x,
            atom_mask=atom_mask,
            angle_threshold=angle_threshold,
            eps=eps,
            inf=inf,
        )

        gt_a = torch.stack(
            [
                x[0, [7, 29], :],
                x[1, [7, 30], :],
            ],
            dim=0,
        )
        gt_b = torch.stack([x[0, [11, 33], :], x[1, [11, 34], :]], dim=0)
        gt_c = torch.stack(
            [
                x[0, [5, 27], :],
                x[1, [5, 28], :],
            ],
            dim=0,
        )
        gt_valid_frame_mask = torch.Tensor([[1, 1], [1, 1]])

        self.assertTrue((torch.abs(a - gt_a) < 1e-5).all())
        self.assertTrue((torch.abs(b - gt_b) < 1e-5).all())
        self.assertTrue((torch.abs(c - gt_c) < 1e-5).all())
        self.assertTrue((valid_frame_mask == gt_valid_frame_mask).all())

        atom_mask[0, 7] = 0

        (a, b, c), valid_frame_mask = get_token_frame_atoms(
            batch=batch,
            x=x,
            atom_mask=atom_mask,
            angle_threshold=angle_threshold,
            eps=eps,
            inf=inf,
        )

        gt_valid_frame_mask = torch.Tensor([[0, 1], [1, 1]])

        self.assertTrue((torch.abs(a - gt_a) < 1e-5).all())
        self.assertTrue((torch.abs(b - gt_b) < 1e-5).all())
        self.assertTrue((torch.abs(c - gt_c) < 1e-5).all())
        self.assertTrue((valid_frame_mask == gt_valid_frame_mask).all())

    def test_with_ligands(self):
        angle_threshold = 25.0
        eps = 1e-8
        inf = 1e9
        batch, x, atom_mask = example5()

        (a, b, c), valid_frame_mask = get_token_frame_atoms(
            batch=batch,
            x=x,
            atom_mask=atom_mask,
            angle_threshold=angle_threshold,
            eps=eps,
            inf=inf,
        )

        gt_a = torch.stack(
            [
                x[0, [1, 0, 0, 3], :],
                x[1, [1, 0, 0, 10], :],
                x[2, [2, 2, 1, 10], :],
                x[3, [1, 0, 0, 0], :],
            ],
            dim=0,
        )
        gt_b = torch.stack(
            [
                x[0, [0, 1, 2, 4], :],
                x[1, [0, 1, 2, 14], :],
                x[2, [0, 1, 2, 14], :],
                x[3, [0, 1, 2, 0], :],
            ],
            dim=0,
        )
        gt_c = torch.stack(
            [
                x[0, [2, 2, 1, 5], :],
                x[1, [2, 2, 1, 8], :],
                x[2, [1, 0, 0, 8], :],
                x[3, [2, 2, 1, 0], :],
            ],
            dim=0,
        )

        # Zero out since no closest neighbors
        a[3, 2, :] = torch.zeros(3)
        c[3, 0, :] = torch.zeros(3)
        c[3, 1, :] = torch.zeros(3)
        c[3, 2, :] = torch.zeros(3)
        gt_a[3, 2, :] = torch.zeros(3)
        gt_c[3, 0, :] = torch.zeros(3)
        gt_c[3, 1, :] = torch.zeros(3)
        gt_c[3, 2, :] = torch.zeros(3)

        gt_valid_frame_mask = torch.Tensor(
            [[1, 1, 1, 1], [0, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 0]]
        )

        self.assertTrue((torch.abs(a - gt_a) < 1e-5).all())
        self.assertTrue((torch.abs(b - gt_b) < 1e-5).all())
        self.assertTrue((torch.abs(c - gt_c) < 1e-5).all())
        self.assertTrue((valid_frame_mask == gt_valid_frame_mask).all())

        atom_mask[2, 2] = 0

        (a, b, c), valid_frame_mask = get_token_frame_atoms(
            batch=batch,
            x=x,
            atom_mask=atom_mask,
            angle_threshold=angle_threshold,
            eps=eps,
            inf=inf,
        )

        gt_valid_frame_mask = torch.Tensor(
            [[1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0]]
        )

        # Zero out since no closest neighbors
        self.assertTrue(
            (
                torch.abs(
                    a * valid_frame_mask[..., None]
                    - gt_a * gt_valid_frame_mask[..., None]
                )
                < 1e-5
            ).all()
        )
        self.assertTrue(
            (
                torch.abs(
                    b * valid_frame_mask[..., None]
                    - gt_b * gt_valid_frame_mask[..., None]
                )
                < 1e-5
            ).all()
        )
        self.assertTrue(
            (
                torch.abs(
                    c * valid_frame_mask[..., None]
                    - gt_c * gt_valid_frame_mask[..., None]
                )
                < 1e-5
            ).all()
        )
        self.assertTrue((valid_frame_mask == gt_valid_frame_mask).all())


if __name__ == "__main__":
    unittest.main()
