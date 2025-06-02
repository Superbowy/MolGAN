import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger   
from rdkit.Chem.rdchem import BondType

from config import ATOM_EQUIV, ATOMS


def get_bond_type(bond_vector):
    """
    Converts a one-hot bond vector to an RDKit BondType.
    bond_vector: [no bond, single, double, triple]
    """
    if not isinstance(bond_vector, np.ndarray):
        bond_vector = np.array(bond_vector)

    idx = np.argmax(bond_vector)

    if idx == 0 or bond_vector[idx] == 0:
        return None  # no bond

    # Map to RDKit bond type
    bond_types = [None, BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE]
    return bond_types[idx]


def build_molecule(X, A, sanitize=True):
    """
    Converts MolGAN-style (X, A) representation to an RDKit molecule.

    X: Atom features (one-hot) for heavy atoms [C, O, N, F]
    A: Adjacency tensor (N x N x 4) with bond type one-hot: [no, single, double, triple]
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
   
    if X.shape[0] == 1:
        X = X.squeeze()
        X = X.squeeze()
    else:
        print("Caution : trying to use build molecule with batch size > 1")

    mol = Chem.RWMol()
    atom_indices = []

    # Add atoms
    for row in X:
        if row[-1] == 1:
            continue  # skip padding rows
        atom_type = ATOMS[np.argmax(row)]
        atom = Chem.Atom(atom_type)
        idx = mol.AddAtom(atom)
        atom_indices.append(idx)

    N = X.shape[0]

    # Add bonds
    for i in range(N):
        for j in range(i + 1, N):
            bond_vec = A[i, j]
            bond_type = get_bond_type(bond_vec)
            if bond_type is not None:
                try:
                    mol.AddBond(i, j, bond_type)
                except Exception as e:
                    return None

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


def check_valid(X, A):
    mol = build_molecule(X, A, sanitize=True)
    return mol is not None


def draw(X, A, filename="image.png"):
    mol = build_molecule(X, A, sanitize=False)
    if mol:
        Draw.MolToFile(mol, filename)


def print_mol(atoms: torch.Tensor):
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
