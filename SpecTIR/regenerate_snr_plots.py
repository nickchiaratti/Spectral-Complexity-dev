import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = r"C:\satelliteImagery\share2012\SpecTIR"

def plot_snr_spectrum(X, noise_covariances, wavelengths, run_id=None):
    N, p = X.shape
    
    if wavelengths is None:
        wavelengths = np.arange(1, p + 1)
        x_label = "Spectral Band Number"
    else:
        x_label = "Wavelength (nm)"
        
    data_variances = np.var(X, axis=0, ddof=1)
    
    plt.figure(figsize=(10, 6))
    
    for method_name, cov_matrix in noise_covariances.items():
        noise_variances = np.diag(cov_matrix)
        signal_variances = data_variances - noise_variances
        
        epsilon = 1e-10
        signal_variances = np.clip(signal_variances, a_min=epsilon, a_max=None)
        noise_variances = np.clip(noise_variances, a_min=epsilon, a_max=None)
        
        snr_linear = signal_variances / noise_variances
        
        plt.plot(wavelengths, snr_linear, label=f"{method_name}", linewidth=1.5)
        
    plt.title("Band-by-Band Signal-to-Noise Ratio (SNR) Comparison")
    plt.xlabel(x_label)
    plt.ylabel("SNR (Linear)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.axhline(1, color='black', linewidth=1, linestyle='-')
    
    plt.tight_layout(pad=0.1)
    plt.savefig(os.path.join(OUTPUT_DIR, f"snr_spectrum_{run_id}.png"), dpi=400)
    plt.close()

def main():
    runs_to_process = [
        "0920-1631",
        "0920-1638",
        "0920-1646",
        "0920-1654",
        "0920-1706"
    ]
    
    for run_id in runs_to_process:
        h5_path = os.path.join(OUTPUT_DIR, f"SpecTIR_Ortho_Stack_{run_id}.h5")
        if not os.path.exists(h5_path):
            print(f"Skipping {run_id}, file not found: {h5_path}")
            continue
            
        print(f"Processing {run_id}...")
        with h5py.File(h5_path, 'r') as f:
            grp = f['HDFEOS/GRIDS/SPECTIR/Data Fields']
            sr_ds = grp['surface_reflectance']
            scale_val = sr_ds.attrs['Scale_Factor'][0]
            wavelengths = sr_ds.attrs['wavelengths'][:]
            
            print("  Reading surface_reflectance...")
            sr = sr_ds[0] # (bands, lines, samples)
            sr_float = sr.astype(np.float32) / scale_val
            
            fill_val = 0
            scaled_fill_val = fill_val / scale_val
            valid_mask = np.all(sr_float != scaled_fill_val, axis=0)
            invalid_values = (sr_float < -1.5) | (sr_float > 1.5)
            valid_mask = valid_mask & ~np.any(invalid_values, axis=0)
            
            X_valid = np.transpose(sr_float, (1, 2, 0))[valid_mask]
            
            noise_covs = {
                'SSD': grp['ssd_noise_cov'][:],
                #'MLR': grp['mlr_noise_cov'][:],
                #'DCT': grp['dct_noise_cov'][:],
                'MNF': grp['mnf_noise_cov'][:]
            }
            
            print(f"  Generating plot for {run_id}...")
            plot_snr_spectrum(X_valid, noise_covs, wavelengths, run_id)

if __name__ == '__main__':
    main()
