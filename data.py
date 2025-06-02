import torch
import os
import torch_geometric.datasets

from config import N, T, Y


def load_data():
    raw_data = torch_geometric.datasets.QM9(root=os.getcwd())
    return raw_data


def raw_to_XA(mol: torch_geometric.data.data.Data):
    """Convert PyG molecular graph to (X, A) format excluding hydrogens (atomic_num = 1)."""

    atom_to_index = {6: 0, 8: 1, 7: 2, 9: 3}  # C, O, N, F
    atom_types = mol.z.tolist()

    # Map non-H atoms to new indices
    non_h_indices = [i for i, z in enumerate(atom_types) if z != 1]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(non_h_indices)}
    n = len(non_h_indices)

    # Initialize A (atomic features) with zeros
    X = torch.zeros((N, 5))

    # Set actual atom types
    for new_idx, old_idx in enumerate(non_h_indices):
        atomic_num = atom_types[old_idx]
        if atomic_num in atom_to_index:
            X[new_idx, atom_to_index[atomic_num]] = 1
        else:
            raise ValueError(f"Unsupported atom type: {atomic_num}")

    # **Set padding atoms to one-hot padding vector [0, 0, 0, 0, 1]**
    for pad_idx in range(n, N):
        X[pad_idx, 4] = 1  # Padding atom index is 4

    # Initialize Z (adjacency tensor)
    A = torch.zeros((n, n, Y))
    edge_index = mol.edge_index
    edge_attr = mol.edge_attr

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if src in index_map and dst in index_map:
            new_src = index_map[src]
            new_dst = index_map[dst]

            # edge_attr[i] is one-hot: [single, double, triple, aromatic]
            bond_vector = edge_attr[i].tolist()
            bond_type = bond_vector.index(1)

            one_hot = torch.zeros(Y)
            if bond_type == 0:  # single
                one_hot[1] = 1
            elif bond_type == 1:  # double
                one_hot[2] = 1
            elif bond_type == 2:  # triple
                one_hot[3] = 1
            elif bond_type == 3:  # aromatic → treat as double
                one_hot[2] = 1
                print("Warning: Aromatic bond found. Mapping to double bond.")
            else:
                raise ValueError(f"Invalid bond type encoding: {bond_vector}")

            A[new_src, new_dst] = one_hot

    # Pad Z to original shape
    A_padded = torch.zeros((N, N, Y))
    A_padded[:n, :n, :] = A

    return X, A_padded
