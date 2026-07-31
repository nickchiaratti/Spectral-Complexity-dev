import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import h5py
import numpy as np
import torch
import math
import datetime

# Add Ghaderpour's package to path
ghaderpour_path = os.path.abspath(os.path.join(os.path.dirname(__file__), r'..\..\..\Reference Articles\LSWAVE-SignalProcessing\JUST_PythonPackage_EGhaderpour'))
if ghaderpour_path not in sys.path:
    sys.path.append(ghaderpour_path)

from ALLSSA import ALLSSA
import warnings
warnings.filterwarnings("ignore")

def extract_fractional_years(acq_times):
    """Converts UNIX timestamps into continuous fractional years (t)."""
    frac_years = []
    for dt in acq_times:
        dt_obj = datetime.datetime.fromtimestamp(float(dt), tz=datetime.timezone.utc)
        year = dt_obj.year
        start_of_year = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
        start_of_next = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
        year_duration = (start_of_next - start_of_year).total_seconds()
        elapsed = (dt_obj - start_of_year).total_seconds()
        frac_years.append(year + (elapsed / year_duration))
    return np.array(frac_years)

# 1. Extract data from HDF5
h5_path = 'C:/satelliteImagery/HLST30/HLST_Tait_Harmonized_SC_EM-7_Norm-None.h5'
print(f"Loading data from {h5_path}...")
with h5py.File(h5_path, 'r') as f:
    # Use temporal_z_score as it likely has values (sliding_volume_z_score might be empty for some tests)
    ds = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields/temporal_z_score']
    z_scores = ds[:]
    acq_times = ds.attrs['acquisition_time'][:]
    mask = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'][:]
    
print(f"Dataset shape: {z_scores.shape}")

# Get real time axis
num_frames = z_scores.shape[0]
frac_years = extract_fractional_years(acq_times)

# Sort chronologically
sort_idx = np.argsort(acq_times)
acq_times = acq_times[sort_idx]
frac_years = frac_years[sort_idx]
z_scores = z_scores[sort_idx, ...]
mask = mask[sort_idx, ...]

date_mask = np.ones(num_frames, dtype=bool)

# Find a good pixel
valid_counts = np.sum(np.isfinite(z_scores) & (mask == 0), axis=0)
max_valid_idx = np.unravel_index(np.argmax(valid_counts, axis=None), valid_counts.shape)
print(f"Selected pixel {max_valid_idx} with {valid_counts[max_valid_idx]} valid observations.")

if valid_counts[max_valid_idx] == 0:
    print("Error: No valid observations found in the dataset for any pixel.")
    sys.exit(1)

pixel_mask = (mask[:, max_valid_idx[0], max_valid_idx[1]] == 0)
pixel_data = z_scores[:, max_valid_idx[0], max_valid_idx[1]]

# Filter to valid dates and unmasked pixels
final_mask = date_mask & pixel_mask
t_valid = frac_years[final_mask]
f_valid = pixel_data[final_mask]

# Replace nans and infs if any
valid_data_mask = np.isfinite(f_valid)
t_valid = t_valid[valid_data_mask]
f_valid = f_valid[valid_data_mask]

# Shift time to start at 0 for computational efficiency as recommended by Ghaderpour
t0 = t_valid - t_valid[0]

print(f"Running ALLSSA on {len(t0)} data points.")

# Configuration
K_FREQUENCIES = 2
NDFT_MIN_CPY = 0.2
NDFT_MAX_CPY = 4.0

# Data-driven grid sizing (Rayleigh criterion)
OVERSAMPLING_FACTOR = 4
FREQ_RESOLUTION_FLOOR = 0.02  # CPY

# ============================================================================
# Helper: Single-pixel LSSA-based ALLSSA in PyTorch
# ============================================================================
def pytorch_lssa_allssa(t_shifted, f_values, f_min, f_max, df_step, K):
    """
    Iterative ALLSSA using true LSSA spectrum evaluation.
    
    Implements Ghaderpour (2018) Eq. for the normalized LSS:
        LSS(omega_j) = r^T Phi_j c_hat / ||r||^2
    where c_hat = (N22 - N12^T @ Ninv @ N12)^-1 @ Phi_j @ r
    
    Args:
        t_shifted: numpy array [W], time axis shifted to start at 0
        f_values:  numpy array [W], observation values
        f_min, f_max: frequency range in CPY
        df_step: frequency grid step in CPY
        K: number of frequencies to extract
    
    Returns:
        freqs_cpy: numpy array [K], extracted frequencies in CPY
        cs_coeffs: numpy array [2*K], cosine/sine coefficients (c1,s1,c2,s2,...)
    """
    DEVICE = torch.device('cpu')
    t = torch.from_numpy(t_shifted).double().to(DEVICE)
    f = torch.from_numpy(f_values).double().to(DEVICE)
    W = len(t)
    
    # Build frequency grid
    n_bins = int((f_max - f_min) / df_step) + 1
    f_grid = torch.linspace(f_min, f_max, n_bins, dtype=torch.float64, device=DEVICE)  # CPY
    G = len(f_grid)
    
    # Precompute test basis pairs Phi_j for all grid frequencies: [G, 2, W]
    angles = 2.0 * math.pi * f_grid.unsqueeze(1) * t.unsqueeze(0)  # [G, W]
    Phi_cos = torch.cos(angles)  # [G, W]
    Phi_sin = torch.sin(angles)  # [G, W]
    
    known_freqs = []  # list of extracted frequencies (CPY)
    
    for k in range(K):
        # Build OLS design matrix D from constant trend + known frequencies
        # D columns: [1, cos(2pi*f1*t), sin(2pi*f1*t), cos(2pi*f2*t), sin(2pi*f2*t), ...]
        nc = 1 + 2 * len(known_freqs)
        D = torch.zeros(W, nc, dtype=torch.float64, device=DEVICE)
        D[:, 0] = 1.0  # constant trend
        for i, fk in enumerate(known_freqs):
            D[:, 1 + 2*i]     = torch.cos(2.0 * math.pi * fk * t)
            D[:, 1 + 2*i + 1] = torch.sin(2.0 * math.pi * fk * t)
        
        # OLS fit: coeff = (D^T D)^-1 D^T f
        DtD = D.T @ D          # [nc, nc]
        Ninv = torch.linalg.inv(DtD)  # [nc, nc]
        coeff = Ninv @ (D.T @ f)      # [nc]
        
        # Residual
        r = f - D @ coeff              # [W]
        norm_r_sq = r @ r              # scalar
        
        # Compute normalized LSS at every grid frequency
        # For each omega_j:
        #   Phi_j = [cos(2pi*omega_j*t); sin(2pi*omega_j*t)]  shape [2, W]
        #   N22 = Phi_j @ Phi_j^T   [2, 2]
        #   N12 = D^T @ Phi_j^T     [nc, 2]
        #   Schur = N22 - N12^T @ Ninv @ N12   [2, 2]
        #   c_hat = Schur^-1 @ Phi_j @ r       [2]
        #   LSS = (Phi_j @ r)^T @ c_hat / norm_r_sq
        
        # Vectorized over G grid frequencies:
        # Phi: [G, 2, W]
        Phi = torch.stack([Phi_cos, Phi_sin], dim=1)  # [G, 2, W]
        
        # N22 = Phi @ Phi^T: [G, 2, 2]
        N22 = torch.bmm(Phi, Phi.transpose(1, 2))  # [G, 2, 2]
        
        # N12 = D^T @ Phi^T: need [G, nc, 2]
        # Phi^T is [G, W, 2], D^T is [nc, W]
        # N12[g] = D^T @ Phi[g]^T = (Phi[g] @ D)^T
        PhiD = torch.bmm(Phi, D.unsqueeze(0).expand(G, -1, -1))  # [G, 2, nc]
        N12 = PhiD.transpose(1, 2)  # [G, nc, 2]
        
        # Correction term: N12^T @ Ninv @ N12: [G, 2, 2]
        # Ninv is [nc, nc], shared across G
        Ninv_N12 = torch.matmul(Ninv.unsqueeze(0), N12)  # [G, nc, 2]
        correction = torch.bmm(N12.transpose(1, 2), Ninv_N12)  # [G, 2, 2]
        
        # Schur complement: [G, 2, 2]
        Schur = N22 - correction
        
        # Analytic 2x2 inverse: [[a,b],[c,d]]^-1 = 1/(ad-bc) * [[d,-b],[-c,a]]
        a = Schur[:, 0, 0]
        b = Schur[:, 0, 1]
        c = Schur[:, 1, 0]
        d = Schur[:, 1, 1]
        det = a * d - b * c
        
        Schur_inv = torch.zeros_like(Schur)
        Schur_inv[:, 0, 0] = d / det
        Schur_inv[:, 0, 1] = -b / det
        Schur_inv[:, 1, 0] = -c / det
        Schur_inv[:, 1, 1] = a / det
        
        # Phi @ r: [G, 2]
        Phi_r = torch.matmul(Phi, r.unsqueeze(-1)).squeeze(-1)  # [G, 2]
        
        # c_hat = Schur_inv @ Phi_r: [G, 2]
        c_hat = torch.bmm(Schur_inv, Phi_r.unsqueeze(-1)).squeeze(-1)  # [G, 2]
        
        # LSS = (Phi_r . c_hat) / norm_r_sq: [G]
        LSS = (Phi_r * c_hat).sum(dim=1) / norm_r_sq
        
        # Clamp to [0, 1] (numerical edge cases near singularity)
        LSS = LSS.clamp(0.0, 1.0)
        
        # Mask out already-known frequencies (set their LSS to 0)
        for fk in known_freqs:
            mask_idx = (f_grid - fk).abs() < (df_step * 0.5)
            LSS[mask_idx] = 0.0
        
        # Select peak frequency
        best_idx = torch.argmax(LSS).item()
        best_freq = f_grid[best_idx].item()
        known_freqs.append(best_freq)
    
    # Final OLS fit with all K known frequencies to get coefficients
    nc_final = 1 + 2 * K
    D_final = torch.zeros(W, nc_final, dtype=torch.float64, device=DEVICE)
    D_final[:, 0] = 1.0
    for i, fk in enumerate(known_freqs):
        D_final[:, 1 + 2*i]     = torch.cos(2.0 * math.pi * fk * t)
        D_final[:, 1 + 2*i + 1] = torch.sin(2.0 * math.pi * fk * t)
    
    DtD_final = D_final.T @ D_final
    coeff_final = torch.linalg.inv(DtD_final) @ (D_final.T @ f)
    
    freqs_cpy = np.array(known_freqs)
    # Coefficients are ordered [trend, c1, s1, c2, s2, ...]
    cs_coeffs = coeff_final[1:].cpu().numpy()
    
    return freqs_cpy, cs_coeffs

# ============================================================================
# Multi-pixel validation: compare LSSA spectra directly
# ============================================================================
from LSSA import LSSA

NUM_TEST_PIXELS = 10
np.random.seed(42)

# Find all pixels with enough valid observations
MIN_VALID_OBS = 50
valid_counts = np.sum(np.isfinite(z_scores) & (mask == 0), axis=0)
candidate_yx = np.argwhere(valid_counts >= MIN_VALID_OBS)

if len(candidate_yx) < NUM_TEST_PIXELS:
    print(f"Warning: Only {len(candidate_yx)} pixels with >= {MIN_VALID_OBS} valid obs. Using all.")
    test_indices = np.arange(len(candidate_yx))
else:
    test_indices = np.random.choice(len(candidate_yx), NUM_TEST_PIXELS, replace=False)

test_pixels = candidate_yx[test_indices]

print(f"\n{'='*70}")
print(f"Validating PyTorch LSSA against Ghaderpour LSSA on {len(test_pixels)} pixels")
print(f"{'='*70}")

all_pass = True

for px_idx, (py, px) in enumerate(test_pixels):
    pixel_mask = (mask[:, py, px] == 0)
    pixel_data = z_scores[:, py, px]
    
    final_mask = pixel_mask & np.isfinite(pixel_data)
    t_valid = frac_years[final_mask]
    f_valid = pixel_data[final_mask]
    
    if len(t_valid) < MIN_VALID_OBS:
        print(f"\n[Pixel {px_idx+1}/{len(test_pixels)}] ({py},{px}): Skipped (only {len(t_valid)} valid obs)")
        continue
    
    t0 = t_valid - t_valid[0]
    
    # Data-driven grid
    T_span = t0[-1] - t0[0]
    df_step = max(1.0 / (OVERSAMPLING_FACTOR * T_span), FREQ_RESOLUTION_FLOOR)
    n_bins = int((NDFT_MAX_CPY - NDFT_MIN_CPY) / df_step) + 1
    Omega_numpy = np.linspace(NDFT_MIN_CPY, NDFT_MAX_CPY, n_bins)
    
    print(f"\n[Pixel {px_idx+1}/{len(test_pixels)}] ({py},{px}): {len(t0)} obs, T_span={T_span:.2f}yr, df={df_step:.4f}, G={n_bins}")
    
    # --- Iteration 1: Compare LSSA spectra with no known frequencies ---
    # Ghaderpour LSSA: evaluate spectrum on the same grid with constant trend, no known freqs
    try:
        gh_result = LSSA(t0, f_valid, P=1, Omega=Omega_numpy, level=0.01,
                         trend='constant', slope=False, freq=[])
        gh_spectrum = np.array(gh_result[0])
    except Exception as e:
        print(f"  Ghaderpour LSSA failed: {e}")
        continue
    
    # PyTorch LSSA: evaluate spectrum on the same grid
    DEVICE = torch.device('cpu')
    t_t = torch.from_numpy(t0).double().to(DEVICE)
    f_t = torch.from_numpy(f_valid).double().to(DEVICE)
    W = len(t_t)
    
    f_grid = torch.from_numpy(Omega_numpy).double().to(DEVICE)
    G = len(f_grid)
    
    angles = 2.0 * math.pi * f_grid.unsqueeze(1) * t_t.unsqueeze(0)
    Phi_cos = torch.cos(angles)
    Phi_sin = torch.sin(angles)
    Phi = torch.stack([Phi_cos, Phi_sin], dim=1)  # [G, 2, W]
    
    # D = [1] for constant trend
    D = torch.ones(W, 1, dtype=torch.float64, device=DEVICE)
    DtD = D.T @ D
    Ninv = torch.linalg.inv(DtD)
    coeff = Ninv @ (D.T @ f_t)
    r = f_t - D @ coeff
    norm_r_sq = r @ r
    
    N22 = torch.bmm(Phi, Phi.transpose(1, 2))
    PhiD = torch.bmm(Phi, D.unsqueeze(0).expand(G, -1, -1))
    N12 = PhiD.transpose(1, 2)
    Ninv_N12 = torch.matmul(Ninv.unsqueeze(0), N12)
    correction = torch.bmm(N12.transpose(1, 2), Ninv_N12)
    Schur = N22 - correction
    
    a = Schur[:, 0, 0]; b = Schur[:, 0, 1]
    c = Schur[:, 1, 0]; d = Schur[:, 1, 1]
    det = a * d - b * c
    
    Schur_inv = torch.zeros_like(Schur)
    Schur_inv[:, 0, 0] = d / det
    Schur_inv[:, 0, 1] = -b / det
    Schur_inv[:, 1, 0] = -c / det
    Schur_inv[:, 1, 1] = a / det
    
    Phi_r = torch.matmul(Phi, r.unsqueeze(-1)).squeeze(-1)
    c_hat = torch.bmm(Schur_inv, Phi_r.unsqueeze(-1)).squeeze(-1)
    pt_spectrum = ((Phi_r * c_hat).sum(dim=1) / norm_r_sq).clamp(0.0, 1.0).cpu().numpy()
    
    # Compare spectra
    spectrum_corr = np.corrcoef(gh_spectrum, pt_spectrum)[0, 1]
    spectrum_mae = np.mean(np.abs(gh_spectrum - pt_spectrum))
    
    # Compare peak frequency
    gh_peak_idx = np.argmax(gh_spectrum)
    pt_peak_idx = np.argmax(pt_spectrum)
    gh_peak_freq = Omega_numpy[gh_peak_idx]
    pt_peak_freq = Omega_numpy[pt_peak_idx]
    
    freq_match = (gh_peak_idx == pt_peak_idx)
    
    # Now run the full K-frequency extraction with PyTorch LSSA-ALLSSA
    pytorch_freqs, pytorch_cs = pytorch_lssa_allssa(
        t0, f_valid, NDFT_MIN_CPY, NDFT_MAX_CPY, df_step, K_FREQUENCIES
    )
    
    pixel_pass = freq_match and (spectrum_corr > 0.999) and (spectrum_mae < 1e-6)
    if not pixel_pass:
        all_pass = False
    
    status = "PASS" if pixel_pass else "FAIL"
    print(f"  Spectrum Correlation: {spectrum_corr:.8f}  |  Spectrum MAE: {spectrum_mae:.2e}")
    print(f"  Peak Freq - Ghaderpour: {gh_peak_freq:.4f}  |  PyTorch: {pt_peak_freq:.4f}  |  Match: {freq_match}")
    print(f"  PyTorch ALLSSA Extracted Freqs: {pytorch_freqs}")
    print(f"  [{status}]")

print(f"\n{'='*70}")
print(f"SUMMARY: All Spectra Match: {all_pass}")
print(f"{'='*70}")

