import h5py
import numpy as np
import matplotlib.pyplot as plt

h5_path = r'C:\satelliteImagery\share2012\SpecTIR\SpecTIR_Ortho_Stack_0920-1706.h5'
with h5py.File(h5_path, 'r') as f:
    cube = f['HDFEOS/GRIDS/SPECTIR/Data Fields/surface_reflectance'][0, :, 100:300, 100:300]
    ssd_cov = f['HDFEOS/GRIDS/SPECTIR/Data Fields/ssd_noise_cov'][:]

cube_scaled = cube.astype(np.float64) / 10000.0
bands, H, W = cube_scaled.shape
pixels = cube_scaled.reshape(bands, -1)

# Overall signal variance
signal_var = np.var(pixels, axis=1)

# SSD noise variance from HDF5
ssd_var = np.diag(ssd_cov)

# Calculate ratio
ratio = ssd_var / (signal_var + 1e-8)

print("Correlation between Signal Variance and SSD Variance:")
print(np.corrcoef(signal_var, ssd_var)[0, 1])

print("\nFirst 10 bands - Signal Var:", signal_var[:10])
print("First 10 bands - SSD Var:", ssd_var[:10])
print("Ratio:", ratio[:10])

# To further prove, calculate empirical SSD on this patch
diff_right = cube_scaled[:, :, :-1] - cube_scaled[:, :, 1:]
diff_down = cube_scaled[:, :-1, :] - cube_scaled[:, 1:, :]
empirical_ssd_var = (np.var(diff_right.reshape(bands, -1), axis=1) + np.var(diff_down.reshape(bands, -1), axis=1)) / 2

print("\nCorrelation between Empirical SSD Var and Signal Var:")
print(np.corrcoef(signal_var, empirical_ssd_var)[0, 1])
