import torch
from discriminator import MolGANDiscriminator
from config import N, T, Y

def testing():
    #raw_data = load_data()
    print(print_mol(raw_data[10].z))
    A, Z = raw_to_AZ(raw_data[10])
    print(check_valid(A, Z))
    draw(A, Z)

B = 2

X = torch.randn(B, N, T)
A = torch.randint(0, 2, (B, N, N, Y)).float()

discriminator = MolGANDiscriminator()

score = discriminator(A, X)
print(score)
