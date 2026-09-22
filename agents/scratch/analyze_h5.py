import h5py
import numpy as np

h5_path = r'C:\satelliteImagery\share2012\SpecTIR\SpecTIR_Ortho_Stack_0920-1706.h5'

def find_datasets(name, node):
    if isinstance(node, h5py.Dataset):
        if name.split('/')[-1] in ['ssd_noise_cov', 'mnf_noise_cov', 'mlr_noise_cov', 'dct_noise_cov']:
            print(f"Found {name}")
            cov = node[:]
            diag = np.diag(cov)
            print(f"\n--- {name} ---")
            print(f"Shape: {cov.shape}")
            print(f"Mean diagonal: {np.mean(diag)}")
            print(f"First 10 diag elements:\n{diag[:10]}")
            if 'mnf' in name:
                off_diag = cov - np.diag(diag)
                print(f"MNF max off-diagonal: {np.max(np.abs(off_diag))}")
                print(f"MNF min diagonal: {np.min(diag)}, max diagonal: {np.max(diag)}")

with h5py.File(h5_path, 'r') as f:
    f.visititems(find_datasets)
