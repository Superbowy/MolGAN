import torch
import torch.nn as nn
from numpy import real
from torch._prims_common import dtype_or_default
from torch.nn.modules import loss

import discriminator
from config import BATCH_SIZE, DEVICE, NUM_WORKERS, NZ
from data import load_data, raw_to_XA
from discriminator import MolGANDiscriminator
from generator import MolGANGenerator
from utils import check_valid, draw, print_mol


def prepare_data():
    data = load_data()
    # transformed_data = [raw_to_XA(x) for x in data]
    transformed_data = []
    for i, x in enumerate(data):
        transformed_data.append(raw_to_XA(x))
        if i == 100:
            break
    data_loader = torch.utils.data.DataLoader(
        transformed_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    return data_loader


def train(
    data_loader: torch.utils.data.DataLoader,
    generator: MolGANGenerator,
    discriminator: MolGANDiscriminator,
    epochs,
    batch_size,
    lrG,
    lrD,
):

    print(f"[!] Using device {DEVICE} for training.")
    losses_G = []
    losses_D = []

    real_label = 0.9
    fake_label = 0.1
    loss_fn = nn.BCELoss()
    optimizerG = torch.optim.Adam(params=generator.parameters(), lr=lrG)
    optimizerD = torch.optim.Adam(params=discriminator.parameters(), lr=lrD)

    for epoch in range(epochs + 1):
        for X, A in data_loader:
            X, A = X.to(DEVICE), A.to(DEVICE)

            # -- Discriminator real -
            discriminator.train()
            label = torch.full(
                (batch_size,), real_label, dtype=torch.float, device=DEVICE
            )
            output = discriminator.forward(A, X).squeeze(1)
            lossD_real = loss_fn(output, label)
            optimizerD.zero_grad()
            lossD_real.backward()
            D_x = output.mean().item()

            # -- Discriminator & generator --
            generator.train()
            noise = torch.randn((batch_size, NZ), device=DEVICE)
            fake_A, fake_X = generator.forward(noise, output="gumbel")
            label.fill_(fake_label)
            output = discriminator.forward(fake_A.detach(), fake_X.detach()).view(-1)
            lossD_fake = loss_fn(output, label)
            optimizerD.zero_grad()
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            # -- Generator --
            optimizerG.zero_grad()
            noise = torch.randn((batch_size, NZ), device=DEVICE)
            fake_A, fake_X = generator.forward(noise, output="gumbel")
            label.fill_(real_label)
            output = discriminator.forward(fake_A.detach(), fake_X.detach()).view(-1)
            lossG = loss_fn(output, label)
            optimizerG.zero_grad()
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            losses_G.append(lossG.item())
            losses_D.append(lossD.item())

        print(
            f"[{epoch}/{EPOCHS}]\tLoss_D: {lossD.item():.4f}\tLoss_G: {lossG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
        )

data_loader = prepare_data()
generator = MolGANGenerator().to(DEVICE)
discriminator = MolGANDiscriminator().to(DEVICE)

train(data_loader, generator, discriminator, 1, BATCH_SIZE, 0.001, 0.001)
