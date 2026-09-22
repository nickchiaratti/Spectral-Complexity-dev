import h5py

h5_path = r'C:\satelliteImagery\share2012\SpecTIR\SpecTIR_Ortho_Stack_0920-1706.h5'

def print_datasets(name, node):
    if isinstance(node, h5py.Dataset):
        print(f"Dataset: {name}, Shape: {node.shape}, Dtype: {node.dtype}")

with h5py.File(h5_path, 'r') as f:
    f.visititems(print_datasets)
