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

    # Z matrix
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

    # Pad Z to (N, N, Y)
    Z_padded = torch.zeros((N, N, Y))
    Z_padded[:n, :n, :] = Z

    return A, Z_padded



def raw_to_AZ2(mol: torch_geometric.data.data.Data):
    """Raw data from torch geometric to paper format"""
    atom_to_index = {6: 0, 7: 1, 8: 2, 9: 3, 16: 4}  # C N O F S
    nodes = mol.z
    n = nodes.shape[0]
    A = torch.zeros((n, 5))

    for row, i in enumerate(nodes):
        if i.item() == 1:
            continue
        A[row][atom_to_index[i.item()]] = 1

    edges = mol.edge_index
    Z = torch.zeros((n, n, Y))
    for i, atom_index in enumerate(edges[0]):
        if atom_index.item() == 1:
            continue
        Z[atom_index][edges[1][i]] = mol.edge_attr[i]

    # Pad A size NxT with zeros below and on the right
    pad_rows = torch.zeros((N - n, T))
    A = torch.cat([A, pad_rows], dim=0)
    # Pad here Z size NxNxY
    Z_padded = torch.zeros((N, N, Y))
    Z_padded[:n, :n, :] = Z
    Z = Z_padded

    return A, Z
