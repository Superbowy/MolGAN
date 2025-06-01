import os

import torch
import torch_geometric.datasets

from config import N, T, Y


def load_data():
    print(os.getcwd())
    raw_data = torch_geometric.datasets.QM9(root=os.getcwd())
    return raw_data


def raw_to_AZ(mol: torch_geometric.data.data.Data):
    """Convert PyG molecular graph to (A, Z) format excluding hydrogens (atomic_num = 1)."""

    atom_to_index = {6: 0, 7: 1, 8: 2, 9: 3, 16: 4}  # C, N, O, F, S
    atom_types = mol.z.tolist()

    # Map non-H atoms to new indices
    non_h_indices = [i for i, z in enumerate(atom_types) if z != 1]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(non_h_indices)}
    n = len(non_h_indices)

    # A matrix
    A = torch.zeros((N, 5))
    for new_idx, old_idx in enumerate(non_h_indices):
        atomic_num = atom_types[old_idx]
        if atomic_num in atom_to_index:
            A[new_idx, atom_to_index[atomic_num]] = 1
        else:
            raise ValueError(f"Unsupported atom type: {atomic_num}")

    Z = torch.zeros((n, n, Y))
    edge_index = mol.edge_index
    edge_attr = mol.edge_attr

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if src in index_map and dst in index_map:
            new_src = index_map[src]
            new_dst = index_map[dst]
            Z[new_src, new_dst] = edge_attr[i]

    Z_padded = torch.zeros((N, N, Y))
    Z_padded[:n, :n, :] = Z

    return A, Z_padded
