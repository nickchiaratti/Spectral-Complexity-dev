import h5py
import numpy as np

h5_path = r'C:\satelliteImagery\share2012\SpecTIR\SpecTIR_Ortho_Stack_0920-1706.h5'
with h5py.File(h5_path, 'r') as f:
    cube = f['HDFEOS/GRIDS/SPECTIR/Data Fields/surface_reflectance'][0, :, 100:300, 100:300]

cube_scaled = cube.astype(np.float64) / 10000.0
bands = cube_scaled.shape[0]
pixels = cube_scaled.reshape(bands, -1)

cov = np.cov(pixels)
eigvals = np.linalg.eigvalsh(cov)
eigvals = np.sort(eigvals)[::-1] # descending

print("Top 20 eigenvalues:")
for i, ev in enumerate(eigvals[:20]):
    print(f"{i+1}: {ev:.4e}")
    
print("\nBottom 20 eigenvalues:")
for i, ev in enumerate(eigvals[-20:]):
    print(f"{bands-19+i}: {ev:.4e}")

print(f"\nCondition number (max/min): {eigvals[0] / (eigvals[-1] + 1e-16):.4e}")

# Check how many eigenvalues are > 1e-8
rank = np.sum(eigvals > 1e-8)
print(f"Rank (eigenvalues > 1e-8): {rank} out of {bands}")
