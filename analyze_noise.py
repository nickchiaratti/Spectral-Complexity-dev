import h5py
import numpy as np

file_path = r"C:\satelliteImagery\share2012\SpecTIR\SpecTIR_Ortho_Stack_0920-1706.h5"
base_path = "HDFEOS/GRIDS/SPECTIR/Data Fields/"
keys = ['ssd_noise_cov', 'mnf_noise_cov', 'mlr_noise_cov', 'dct_noise_cov']

try:
    with h5py.File(file_path, 'r') as f:
        for key in keys:
            full_path = base_path + key
            if full_path in f:
                cov = f[full_path][:]
                diag = np.diag(cov)
                print(f"\n--- {key} ---")
                print(f"Shape: {cov.shape}")
                print(f"Min variance: {np.min(diag):.6e}")
                print(f"Max variance: {np.max(diag):.6e}")
                print(f"Mean variance: {np.mean(diag):.6e}")
                print(f"Median variance: {np.median(diag):.6e}")
                print(f"Std of variance across bands: {np.std(diag):.6e}")
                print(f"Coefficient of variation (Std/Mean): {np.std(diag)/np.mean(diag):.4f}")
                
                # Check for "spectrally flat" property
                # Smoothness: mean absolute difference between adjacent bands
                diffs = np.abs(np.diff(diag))
                print(f"Mean absolute diff between adjacent bands: {np.mean(diffs):.6e}")
                
                num_bands = len(diag)
                indices = np.linspace(0, num_bands-1, 10, dtype=int)
                print(f"Sample variances across {num_bands} bands at indices {indices}:")
                print([f"{diag[i]:.6e}" for i in indices])
            else:
                print(f"{key} not found at {full_path}")
except Exception as e:
    print("Error:", e)
