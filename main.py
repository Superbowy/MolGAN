import time

import torch
import torch.nn as nn

from config import BATCH_SIZE, DEVICE, NUM_WORKERS, NZ
from data import load_data, raw_to_XA
from discriminator import MolGANDiscriminator
from generator import MolGANGenerator

torch.manual_seed(1)


def prepare_data():

    data = load_data()
    print("Transforming data...")
    transformed_data = [raw_to_XA(x) for x in data]
    print("Data transformed.")
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
    start_time = time.time()
    counter = 0
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
            output = discriminator.forward(fake_A.detach(), fake_X.detach()).squeeze(1)
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
            output = discriminator.forward(fake_A, fake_X).squeeze(1)
            lossG = loss_fn(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            losses_G.append(lossG.item())
            losses_D.append(lossD.item())

            if counter % 10000 == 0:
                print(
                    f"[{epoch}/{epochs}]\tLoss_D: {lossD.item():.4f}\tLoss_G: {lossG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}. ---- Dt : {time.time() - start_time:.2f}s."
                )
                start_time = time.time()
            counter += 32


data_loader = prepare_data()
generator = MolGANGenerator().to(DEVICE)
discriminator = MolGANDiscriminator().to(DEVICE)

train(data_loader, generator, discriminator, 1, BATCH_SIZE, 0.001, 0.001)
