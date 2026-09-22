import h5py
import numpy as np

file_path = r"C:\satelliteImagery\share2012\SpecTIR\SpecTIR_Ortho_Stack_0920-1706.h5"
base_path = "HDFEOS/GRIDS/SPECTIR/Data Fields/"
keys = ['ssd_noise_cov', 'mnf_noise_cov', 'mlr_noise_cov', 'dct_noise_cov']

try:
    with h5py.File(file_path, 'r') as f:
        for key in keys:
            full_path = base_path + key
            cov = f[full_path][:]
            diag = np.diag(cov)
            
            # Look at a small window of 20 bands to see if it is artificially smooth
            window = diag[100:120]
            print(f"\n--- {key} ---")
            print(f"Bands 100-119 variance: {window}")
            print(f"Mean absolute difference in window: {np.mean(np.abs(np.diff(window))):.6e}")
except Exception as e:
    print("Error:", e)
