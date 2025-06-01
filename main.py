from data import load_data, raw_to_AZ
from utils import draw, check_valid, print_mol


raw_data = load_data()
print(print_mol(raw_data[10].z))
A, Z = raw_to_AZ(raw_data[10])
print(check_valid(A, Z))
draw(A, Z)
