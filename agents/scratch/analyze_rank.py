import h5py
import numpy as np

h5_path = r'C:\satelliteImagery\share2012\SpecTIR\SpecTIR_Ortho_Stack_0920-1706.h5'

def analyze_cube(name, node):
    if isinstance(node, h5py.Dataset):
        if len(node.shape) == 3:
            print(f"Found cube: {name}, shape: {node.shape}, dtype: {node.dtype}")
            
            # Read a spatial chunk to compute covariance and rank
            # Suppose shape is (bands, height, width) or (height, width, bands)
            if node.shape[0] > 100:
                # likely (bands, H, W)
                cube = node[:, 100:200, 100:200].astype(np.float64)
                bands = cube.shape[0]
                pixels = cube.reshape(bands, -1)
            else:
                # likely (H, W, bands)
                cube = node[100:200, 100:200, :].astype(np.float64)
                bands = cube.shape[2]
                pixels = cube.reshape(-1, bands).T
            
            print(f"Pixels shape: {pixels.shape}")
            
            cov = np.cov(pixels)
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.sort(eigvals)[::-1] # descending
            
            print("Top 10 eigenvalues:")
            print(eigvals[:10])
            print("Bottom 10 eigenvalues:")
            print(eigvals[-10:])
            print(f"Condition number: {eigvals[0] / (eigvals[-1] + 1e-12)}")
            
            # Print non-zero eigenvalue count
            non_zero = np.sum(eigvals > 1e-10 * eigvals[0])
            print(f"Estimated rank (threshold 1e-10 * max): {non_zero}")

with h5py.File(h5_path, 'r') as f:
    f.visititems(analyze_cube)
