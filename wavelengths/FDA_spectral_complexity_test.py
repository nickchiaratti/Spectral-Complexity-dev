"""
Functional Data Analysis (FDA) Spectral Complexity Tester
Implements Native Riemann Integral Weighting (Delta Lambda) and 
Unified Tikhonov Regularization for Cross-Sensor Calibration.

Dependencies: h5py, numpy, skfda, rasterio, pyproj, matplotlib
"""

import os
import sys
import h5py
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt

try:
    import skfda
    from skfda.representation.basis import BSplineBasis
except ImportError:
    print("CRITICAL ERROR: skfda not found. Please run: pip install scikit-fda")
    sys.exit(1)
script_dir = os.path.dirname(os.path.realpath(__file__))

# ==========================================
# 1. CONFIGURATION & UNIVERSAL FDA SETUP
# ==========================================
ARD_CUBE_PATH = "C:/satelliteImagery/HLST30/HLST_Tait_Harmonized_SC_EM-7_Norm-bandCount.h5"

# Target spatial coordinates
TARGET_LAT = 43.142856
TARGET_LON = -77.508451

# Feature Toggles
VISUALIZE_FITS = True # Set to True to plot the continuous B-Spline for each sensor

# Universal Domain encompasses 400nm -> 2500nm.
UNIVERSAL_DOMAIN = (400, 2500) 
N_BASIS = 6 

print(f"Initializing Universal B-Spline Basis (Domain: {UNIVERSAL_DOMAIN} nm, Basis Count: {N_BASIS})")
universal_basis = BSplineBasis(domain_range=UNIVERSAL_DOMAIN, n_basis=N_BASIS)
W_MATRIX = universal_basis.inner_product_matrix()


# ==========================================
# 2. CORE MATHEMATICS (RIEMANN WEIGHTED FDA)
# ==========================================

def compute_functional_complexity(spectra_window, wavelengths):
    """
    Computes the Functional Spectral Complexity characteristic length.
    Applies Riemann Integration Weights and Tikhonov Regularization.
    
    Returns:
        characteristic_length (float), coefficients matrix (ndarray)
    """
    if spectra_window.shape[0] < 3:
        return np.nan, None 

    # 1. Evaluate the Universal Basis at the given center wavelengths
    B = np.squeeze(universal_basis(wavelengths))
    if B.ndim == 1:
        B = np.expand_dims(B, 0)
    if B.shape[0] != N_BASIS:
        B = B.T  # Shape: (K, number_of_bands)
        
    # 2. Native Riemann Integral Weighting (The Equalizer)
    # Estimate the spectral bandwidth (Delta Lambda) of each band
    delta_lambdas = np.gradient(wavelengths)
    
    # Normalize so the total weight "mass" is exactly 1.0 for ALL sensors.
    # This prevents Tanager's 426 bands from overpowering Landsat's 8 bands.
    normalized_weights = delta_lambdas / np.sum(delta_lambdas)
    W = np.diag(normalized_weights)
        
    # 3. Unified Tikhonov Regularized Least Squares (Ridge)
    B_W = B @ W
    B_W_B = B_W @ B.T
    
    # L2 Regularization parameter
    lambda_reg = 1e-4
    reg_matrix = np.eye(N_BASIS) * lambda_reg
    
    try:
        # Solve: C^T = (B W B^T + lambda*I)^-1 * (B W Y^T)
        C_T = np.linalg.solve(B_W_B + reg_matrix, B_W @ spectra_window.T)
        C = C_T.T  # Final Coefficients Shape: (N, K)
    except np.linalg.LinAlgError:
        return np.nan, None
    
    # 4. Localize the Coefficients (Geometric Centroiding)
    C_local = C - np.mean(C, axis=0)
    
    # 5. Map to Orthonormal Basis Space (using Cholesky decomp of Inner Product Integral)
    L = np.linalg.cholesky(W_MATRIX)
    C_ortho = C_local @ L  
    
    # 6. Compute Feature Covariance
    Sigma = C_ortho.T @ C_ortho  
    
    # --- DIMENSION-INVARIANT SPREAD (TRACE) ---
    # Using the Trace (sum of variances) rather than the Determinant (product of variances)
    # prevents Landsat's volume from collapsing due to intrinsic rank-deficiency.
    total_variance = np.trace(Sigma)
    
    # Characteristic length is the Root Mean Square of the functional dispersion
    characteristic_length = np.sqrt(total_variance)
    
    return characteristic_length, C


# ==========================================
# 3. VISUALIZATION ENGINE
# ==========================================

def visualize_fda_fit(discrete_spectra, wavelengths, C, universal_basis, sensor_name="Sensor"):
    """
    Visualizes the discrete pixel observations against the fitted continuous B-spline.
    Uses the manually calculated Tikhonov regularized coefficients.
    """
    # Create a dense, continuous domain to evaluate the smooth curve (400nm to 2500nm)
    dense_wavelengths = np.linspace(400, 2500, 500)
    
    # Evaluate basis over dense domain
    B_dense = np.squeeze(universal_basis(dense_wavelengths))
    if B_dense.ndim == 1:
        B_dense = np.expand_dims(B_dense, 0)
    if B_dense.shape[0] != N_BASIS:
        B_dense = B_dense.T # Shape: (K, 500)

    # Compute smooth curves: Y = C @ B
    smooth_curves = C @ B_dense
    
    plt.figure(figsize=(10, 6))
    
    # Plot a maximum of 3 pixels from the window to keep the chart clean
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    num_to_plot = min(3, discrete_spectra.shape[0])
    
    for i in range(num_to_plot):
        # Plot the raw, discrete sensor observations as scatter dots
        plt.scatter(wavelengths, discrete_spectra[i], color=colors[i], 
                    s=50, zorder=3, label=f'Pixel {i+1} (Discrete)')
        
        # Plot the continuous FDA B-Spline fit as a solid line
        plt.plot(dense_wavelengths, smooth_curves[i], color=colors[i], 
                 linewidth=2, zorder=2, alpha=0.8, label=f'Pixel {i+1} (Continuous)')
        
    plt.title(f"Functional Data Analysis: {sensor_name} Regularized B-Spline Fit")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.ylim(-0.1, 1.0) # Standard reflectance bounds
    plt.xlim(300, 2600)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Highlight the typical unobserved SWIR gaps
    plt.axvspan(1360, 1406, color='gray', alpha=0.2, label='Atmospheric Gap')
    plt.axvspan(1800, 1970, color='gray', alpha=0.2)
    
    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir,f'{sensor_name}_fda_fit.png'))


# ==========================================
# 4. HDF5 CUBE ITERATION & EXTRACTION
# ==========================================
def main():
    if not os.path.exists(ARD_CUBE_PATH):
        print(f"Skipping execution: Test file {ARD_CUBE_PATH} not found.")
        return 

    print(f"Opening HDF-EOS5 Harmonized Cube: {ARD_CUBE_PATH}")
    
    with h5py.File(ARD_CUBE_PATH, 'r') as h5:
        harm_grp = h5['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        timestamps = harm_grp['sliding_volume_map'].attrs['acquisition_time']
        sensor_grids = harm_grp['sliding_volume_map'].attrs['source_grid'].astype(str)
        local_indices = harm_grp['sliding_volume_map'].attrs['source_frame_index']
        global_mask = harm_grp['common_mask'][:] 
        
        # ---------------------------------------------------------
        # DYNAMIC ROI DISCOVERY
        # ---------------------------------------------------------
        print("Scanning global mask to dynamically find a valid spatial target...")
        VALID_MASK_VALUE = 1 
        
        clear_pixel_counts = np.sum(global_mask == VALID_MASK_VALUE, axis=0)
        valid_y, valid_x = np.where(clear_pixel_counts > 0)
        
        if len(valid_y) == 0:
            print(f"CRITICAL ERROR: No valid data found in the mask using value {VALID_MASK_VALUE}.")
            return
            
        center_row, center_col = valid_y[len(valid_y)//2], valid_x[len(valid_x)//2]
        print(f"Dynamic Target Selected -> Row: {center_row}, Col: {center_col}")
        
        r_min, r_max = center_row - 1, center_row + 2
        c_min, c_max = center_col - 1, center_col + 2
        
        print("\n--- Initiating Riemann-Weighted Functional Temporal Analysis ---")
        
        valid_epochs = 0
        plotted_sensors = set() # Track which sensors we have already visualized
        
        for t in range(len(timestamps)):
            sensor = sensor_grids[t]
            local_idx = local_indices[t]
            
            mask_window = global_mask[t, r_min:r_max, c_min:c_max]
            valid_pixel_mask = (mask_window == VALID_MASK_VALUE)
            if np.sum(valid_pixel_mask) < 3:
                continue 
                
            native_ds_path = f'/HDFEOS/GRIDS/{sensor}/Data Fields/surface_reflectance'
            native_ds = h5[native_ds_path]
            if sensor == 'TANAGER':
                wavelengths = native_ds.attrs['wavelengths'][:]
            else:
                wavelengths = native_ds.attrs['wavelengths'][:]*1000.0
            
            
            
            # Flatten spatial dimensions -> (9, Bands)
            raw_window = native_ds[local_idx, :, r_min:r_max, c_min:c_max]
            raw_window = np.transpose(raw_window, (1, 2, 0)) 
            flat_window = raw_window.reshape(-1, len(wavelengths))
            valid_spectra = flat_window[valid_pixel_mask.flatten()]
            
            # 3. Compute Riemann-Weighted Functional Spectral Complexity
            fda_volume, C_matrix = compute_functional_complexity(valid_spectra, wavelengths)
            
            dt = datetime.fromtimestamp(timestamps[t], tz=timezone.utc)
            date_str = dt.strftime('%Y-%m-%d %H:%M')
            print(f"[{date_str}] {sensor.ljust(10)} | Bands: {len(wavelengths):>3} | FDA Spread: {fda_volume:.6e}")
            
            # 4. Trigger Visualization (Once per sensor type)
            if VISUALIZE_FITS and sensor not in plotted_sensors and C_matrix is not None:
                visualize_fda_fit(valid_spectra, wavelengths, C_matrix, universal_basis, sensor_name=sensor)
                plotted_sensors.add(sensor)
                
            
            valid_epochs += 1
            
        print(f"\nAnalysis Complete. Processed {valid_epochs} valid epochs utilizing Universal Basis.")
        
        if VISUALIZE_FITS:
            plt.show() # Hold the plots open at the end

if __name__ == "__main__":
    main()