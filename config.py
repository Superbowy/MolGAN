import torch.nn as nn
import torch.nn.functional as F

# CHEMICAL CONSTANTS
ATOMS = ["C", "N", "O", "S", "F"]
ATOM_EQUIV = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F"}
N = 9  # nombre max de nœuds
T = 5  # nombre de types d’atomes
Y = 4  # nombre de types de liaison


# TRAINING CONSTANTS
nz = 32  # Z size
