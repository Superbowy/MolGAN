import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NZ, N, T, Y


class MolGANGenerator(nn.Module):
    def __init__(self, hidden_dims=(128, 256, 512)):
        super().__init__()
        self.N = N
        self.T = T
        self.Y = Y

        h1, h2, h3 = hidden_dims
        # MLP qui part de z_dim, passe par h1,h2,h3, et sort en dimension N*T + N*N*Y
        self.net = nn.Sequential(
            nn.Linear(NZ, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
            nn.Linear(h2, h3),
            nn.Tanh(),
            nn.Linear(h3, N * T + N * N * Y),  # → 9*5 + 9*9*4 = 369
        )

    def forward(self, z, output: str, tau=0.1):
        """ Takes z ~ N(0;1) and returns X, A)"""
        if output not in ["logits", "gumbel", "argmax"]:
            raise ValueError(
                "Invalid parameter for output. Expected logits, gumbel, argmax."
            )
        batch_size = z.size(0)
        out = self.net(z)  # (batch_size, 369)

        # Séparer la partie X et la partie A
        X_logits = out[:, : N * T]  # (batch_size, 45)
        A_logits = out[:, N * T :]  # (batch_size, 324)

        # Reshape
        X_logits = X_logits.view(batch_size, N, T)  # (batch_size, 9, 5)
        A_logits = A_logits.view(batch_size, N, N, Y)  # (batch_size, 9, 9, 4)

        if output == "logits":
            A = A_logits
            X = X_logits
        elif output == "gumbel":
            # X_probs = F.softmax(X_logits, dim=-1)  # (batch_size, 9, 5)
            # A_probs = F.softmax(A_logits, dim=-1)  # (batch_size, 9, 9, 4)
            A = F.gumbel_softmax(A_logits, dim=-1, tau=tau, hard=False)
            X = F.gumbel_softmax(X_logits, dim=-1, tau=tau, hard=False)
        else:
            A = F.one_hot(torch.argmax(A_logits, dim=-1), num_classes=4).float()
            X = F.one_hot(torch.argmax(X_logits, dim=-1), num_classes=5).float()
        return A, X
