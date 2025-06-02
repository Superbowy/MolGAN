import torch
import torch.nn as nn
import torch.nn.functional as F

from config import T, Y


class MolGANDiscriminator(nn.Module):
    def __init__(self, node_feat_dim=T, hidden_dim=64, num_relations=Y, num_layers=3):
        super(MolGANDiscriminator, self).__init__()
        self.rgcn_layers = nn.ModuleList()

        self.rgcn_layers.append(RGCNLayer(node_feat_dim, hidden_dim, num_relations))
        for _ in range(num_layers - 1):
            self.rgcn_layers.append(RGCNLayer(hidden_dim, hidden_dim, num_relations))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, A, X):
        """
        A: [B, N, N, Y]
        X: [B, N, T]
        Returns: [B, 1] (score per graph)
        """
        h = X
        for rgcn in self.rgcn_layers:
            h = F.relu(rgcn(h, A))

        # Graph-level readout: sum over node embeddings
        g = torch.sum(h, dim=1)  # [B, hidden_dim]
        out = self.mlp(g)  # [B, 1]
        return out


class RGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGCNLayer, self).__init__()
        self.num_relations = num_relations
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.Tensor(num_relations, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, X, A):
        """
        X: [B, N, F_in]
        A: [B, N, N, R]
        Output: [B, N, F_out]
        """
        B, N, _, R = A.shape
        out = torch.zeros(B, N, self.out_features, device=X.device)

        for r in range(R):
            A_r = A[:, :, :, r]  # [B, N, N]
            W_r = self.weight[r]  # [F_in, F_out]
            XW_r = torch.matmul(X, W_r)  # [B, N, F_out]
            out += torch.bmm(A_r, XW_r)  # [B, N, F_out]

        out = out + self.bias  # broadcast over [B, N, F_out]
        return out
