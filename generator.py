import torch.nn as nn
import torch.nn.functional as F

from config import N, T, Y


class MolGANGenerator(nn.Module):
    def __init__(self, z_dim=32, hidden_dims=(128, 256, 512)):
        super().__init__()
        self.N = N
        self.T = T
        self.Y = Y

        h1, h2, h3 = hidden_dims
        # MLP qui part de z_dim, passe par h1,h2,h3, et sort en dimension N*T + N*N*Y
        self.net = nn.Sequential(
            nn.Linear(z_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, h3),
            nn.Tanh(),
            nn.Linear(h3, N * T + N * N * Y),  # → 9*5 + 9*9*4 = 369
        )

    def forward(self, z):
        """
        Prend z: (batch_size, 32)
        Retourne :
          X_probs: (batch_size, N, T)
          A_probs: (batch_size, N, N, Y)
        """
        batch_size = z.size(0)
        out = self.net(z)  # (batch_size, 369)

        # Séparer la partie X et la partie A
        X_logits = out[:, : N * T]  # (batch_size, 45)
        A_logits = out[:, N * T :]  # (batch_size, 324)

        # Reshape
        X_logits = X_logits.view(batch_size, N, T)  # (batch_size, 9, 5)
        A_logits = A_logits.view(batch_size, N, N, Y)  # (batch_size, 9, 9, 4)

        # Softmax sur la dernière dimension
        X_probs = F.softmax(X_logits, dim=-1)  # (batch_size, 9, 5)
        A_probs = F.softmax(A_logits, dim=-1)  # (batch_size, 9, 9, 4)

        return X_probs, A_probs
