import h5py

def print_h5_keys(name, obj):
    print(name)

path = r"D:\Dataset\ddff-dataset-trainval.h5"

with h5py.File(path, "r") as f:
    f.visititems(print_h5_keys)