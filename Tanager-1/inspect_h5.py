import h5py

file_path = r'C:\satelliteImagery\Tanager\Rochesterv2_SourceData\20250801_165544_61_4001\20250801_165544_61_4001_basic_sr_hdf5.h5'

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: Dataset {obj.shape} {obj.dtype}")
    else:
        print(f"{name}: Group")

try:
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_structure)
except Exception as e:
    print(f"Error reading file: {e}")
