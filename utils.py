import torch
from config import ATOM_EQUIV, ATOMS

import torch
from config import N
from rdkit import Chem
from rdkit.Chem import Draw

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import BondType
import numpy as np

# Helper to convert Z to bond type
def get_bond_type(bond_vector):
    bond_types = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
    idx = np.argmax(bond_vector)
    if bond_vector[idx] == 0:
        return None
    return bond_types[idx]

def build_molecule(A, Z, sanitize=True):
    """
    Builds an RDKit molecule from MolGAN-style A and Z tensors (heavy atoms only).
    Hydrogens are inferred by RDKit during sanitization.
    
    Args:
        A (np.ndarray): shape (N, 5), one-hot encoded atom types (C, N, O, F, S)
        Z (np.ndarray): shape (N, N, 4), one-hot bond types
        sanitize (bool): if True, sanitize and infer Hs

    Returns:
        mol (Chem.Mol or None): RDKit molecule or None if invalid
    """
    if isinstance(A, torch.Tensor):
        A = A.cpu().numpy()
    if isinstance(Z, torch.Tensor):
        Z = Z.cpu().numpy()

    mol = Chem.RWMol()
    atom_indices = []

    # Add heavy atoms
    for row in A:
        if np.sum(row) == 0:
            continue  # skip padding
        atom_type = ATOMS[np.argmax(row)]
        atom = Chem.Atom(atom_type)
        idx = mol.AddAtom(atom)
        atom_indices.append(idx)

    # Add bonds
    N = A.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            bond_vec = Z[i, j]
            bond_type = get_bond_type(bond_vec)
            if bond_type is not None:
                try:
                    mol.AddBond(i, j, bond_type)
                except Exception:
                    pass  # invalid bond addition, skip

    if sanitize:
        try:
            mol = mol.GetMol()
            Chem.SanitizeMol(mol)
            mol = Chem.AddHs(mol)  # infer hydrogens
            return mol
        except Exception:
            return None
    else:
        return mol

def check_valid(A, Z):
    mol = build_molecule(A, Z, sanitize=True)
    return mol is not None

def draw(A, Z, filename="image.png"):
    mol = build_molecule(A, Z, sanitize=False)
    if mol:
        Draw.MolToFile(mol, filename)

def print_mol(atoms : torch.Tensor):
    return [ATOM_EQUIV[index.item()] for index in atoms.squeeze()]


# def repr_molecule(mol : torch_geometric.data.data.Data):
#     print(f"Repr shape : {mol.z.size()[0]}x5")
#     print(mol_equiv(mol.z))
# 
#     links = {}
#     for i, element in enumerate(mol.edge_index[0]):
#         element = element.item()
#         if element in links.keys():
#             links[element].append(mol.edge_index[1][i].item())
#         else:
#             links[element] = [mol.edge_index[1][i].item()]
# 
#     for key, values in links.items():
#         print(f"Atom {key} has links with {values}")
