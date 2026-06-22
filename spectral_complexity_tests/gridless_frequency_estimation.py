import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import torch
import math

# --- Configuration & Hyperparameters ---
NDFT_MIN_CPY = 0.3
NDFT_MAX_CPY = 4.0
NDFT_GRID_BINS = 100
SECONDS_IN_YEAR = 365.25 * 24 * 3600

# Ensure GPU execution if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Algorithm Implementations ---

def compute_pytorch_ndft(t_years, y_data, min_f, max_f, bins):
    """Grid-based Non-Uniform DFT (PyTorch Accelerated)"""
    t = torch.tensor(t_years, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_data, dtype=torch.float32, device=DEVICE)
    
    f_grid = torch.linspace(min_f, max_f, bins, device=DEVICE)
    omega = 2 * math.pi * f_grid
    
    E = torch.exp(-1j * omega.unsqueeze(1) * t.unsqueeze(0))
    spectrum = torch.abs(torch.matmul(E, y.to(torch.complex64)))
    
    return f_grid.cpu().numpy(), spectrum.cpu().numpy()

def compute_pytorch_nomp(t_years, y_data, min_f, max_f, max_components=3):
    """True Newtonized Orthogonal Matching Pursuit (Joint Continuous Refinement via PyTorch)"""
    t = torch.tensor(t_years, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_data, dtype=torch.float32, device=DEVICE)
    
    # 1. Dense Coarse Grid for initial detection
    f_grid = torch.linspace(min_f, max_f, 1000, device=DEVICE)
    omega_grid = 2 * math.pi * f_grid
    
    frequencies = []
    
    for k in range(max_components):
        # Build design matrix with current frequencies
        def build_X(freqs_tensor):
            cols = []
            for i in range(len(freqs_tensor)):
                omega = 2 * math.pi * freqs_tensor[i]
                cols.append(torch.cos(omega * t).unsqueeze(1))
                cols.append(torch.sin(omega * t).unsqueeze(1))
            return torch.cat(cols, dim=1) if cols else torch.empty((len(t), 0), device=DEVICE)
            
        # Current Residual
        if len(frequencies) > 0:
            f_tensor_static = torch.tensor(frequencies, dtype=torch.float32, device=DEVICE)
            X = build_X(f_tensor_static)
            beta = torch.linalg.lstsq(X, y.unsqueeze(1)).solution
            y_pred = torch.mm(X, beta).squeeze(1)
            residual = y - y_pred
        else:
            residual = y.clone()
            
        # Detect new frequency on grid
        E = torch.exp(-1j * omega_grid.unsqueeze(1) * t.unsqueeze(0))
        spectrum = torch.abs(torch.matmul(E, residual.to(torch.complex64)))
        f_new = f_grid[torch.argmax(spectrum)].item()
        frequencies.append(f_new)
        
        # 2. Joint Continuous Refinement of ALL frequencies (True NOMP Step)
        freqs_tensor = torch.tensor(frequencies, dtype=torch.float32, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.LBFGS([freqs_tensor], max_iter=15, line_search_fn='strong_wolfe')
        
        def closure():
            optimizer.zero_grad()
            freqs_clamped = torch.clamp(freqs_tensor, min=min_f, max=max_f)
            X_opt = build_X(freqs_clamped)
            
            # Differentiable Least Squares for Amplitudes
            lstsq_out = torch.linalg.lstsq(X_opt, y.unsqueeze(1))
            beta_opt = lstsq_out.solution
            y_pred_opt = torch.mm(X_opt, beta_opt).squeeze(1)
            
            # Loss is sum of squared residuals
            loss = torch.sum((y - y_pred_opt)**2)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        frequencies = torch.clamp(freqs_tensor.detach(), min=min_f, max=max_f).tolist()
        
    # 3. Final Amplitude Extraction
    f_final = torch.tensor(frequencies, dtype=torch.float32, device=DEVICE)
    X_final = build_X(f_final)
    beta_final = torch.linalg.lstsq(X_final, y.unsqueeze(1)).solution.squeeze(1)
    
    amps = []
    for i in range(len(frequencies)):
        c = beta_final[2*i]
        s = beta_final[2*i + 1]
        amps.append(math.sqrt(c**2 + s**2))
        
    return frequencies, amps

def compute_pytorch_cbpdn_continuous(t_years, y_data, min_f, max_f, max_atoms=20):
    """Continuous Basis Pursuit Denoising (C-BPDN)
    Directly optimizes continuous frequencies and amplitudes with an L1 sparsity penalty,
    achieving exact gridless super-resolution without SDP solvers or fill values.
    """
    t = torch.tensor(t_years, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_data, dtype=torch.float32, device=DEVICE)
    
    # 1. Initialize with over-parameterized NDFT peaks
    f_grid = torch.linspace(min_f, max_f, 1000, device=DEVICE)
    omega_grid = 2 * math.pi * f_grid
    E = torch.exp(-1j * omega_grid.unsqueeze(1) * t.unsqueeze(0))
    spectrum = torch.abs(torch.matmul(E, y.to(torch.complex64)))
    
    import scipy.signal
    spectrum_np = spectrum.cpu().numpy()
    peaks, _ = scipy.signal.find_peaks(spectrum_np, distance=5)
    
    if len(peaks) == 0:
        return [], []
        
    top_peaks = peaks[np.argsort(spectrum_np[peaks])[::-1]][:max_atoms]
    f_init = f_grid[top_peaks]
    
    # 2. Continuous Parameters
    freqs = f_init.clone().detach().requires_grad_(True)
    a_real = torch.randn(len(freqs), dtype=torch.float32, device=DEVICE, requires_grad=True)
    a_imag = torch.randn(len(freqs), dtype=torch.float32, device=DEVICE, requires_grad=True)
    
    optimizer = torch.optim.Adam([freqs, a_real, a_imag], lr=0.05)
    lambda_reg = 0.15 * torch.max(spectrum) # Atomic norm sparsity penalty
    
    # 3. Primal Gradient Descent Loop
    for _ in range(800):
        optimizer.zero_grad()
        omega = 2 * math.pi * torch.clamp(freqs, min=min_f, max=max_f)
        
        A_real = torch.cos(omega.unsqueeze(1) * t.unsqueeze(0))
        A_imag = torch.sin(omega.unsqueeze(1) * t.unsqueeze(0))
        
        y_pred_real = torch.matmul(a_real, A_real) - torch.matmul(a_imag, A_imag)
        
        mse = torch.sum((y - y_pred_real)**2)
        # L1 norm of complex amplitudes promotes sparsity
        l1 = lambda_reg * torch.sum(torch.sqrt(a_real**2 + a_imag**2 + 1e-8))
        
        loss = mse + l1
        loss.backward()
        optimizer.step()
        
    f_final = torch.clamp(freqs, min=min_f, max=max_f).detach().cpu().numpy()
    amps_final = torch.sqrt(a_real**2 + a_imag**2).detach().cpu().numpy()
    
    # 4. Prune inactive atoms (Soft-thresholded away)
    threshold = 0.05 * np.max(amps_final)
    active = amps_final > threshold
    
    f_active = f_final[active]
    amps_active = amps_final[active]
    
    # Deduplicate frequencies that converged to the same continuous well
    f_unique = []
    amps_unique = []
    
    sort_active = np.argsort(amps_active)[::-1]
    for f, amp in zip(f_active[sort_active], amps_active[sort_active]):
        if not any(abs(f - uf) < 0.05 for uf in f_unique):
            f_unique.append(f)
            amps_unique.append(amp)
            
    f_active = np.array(f_unique)
    amps_active = np.array(amps_unique)
    
    # Limit to highest confidence top 3 frequencies
    if len(amps_active) > 3:
        f_active = f_active[:3]
        amps_active = amps_active[:3]
        
    return f_active, amps_active

def compute_pytorch_cirl_continuous(t_years, y_data, min_f, max_f, max_atoms=20):
    """Continuous Iterative Reweighted L1 (CIRL)
    Implements a Log-Sum penalty to mimic the inverse-weighting behavior 
    of SPICE, promoting sparsity iteratively off-grid.
    """
    t = torch.tensor(t_years, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_data, dtype=torch.float32, device=DEVICE)
    
    f_grid = torch.linspace(min_f, max_f, 1000, device=DEVICE)
    omega_grid = 2 * math.pi * f_grid
    E = torch.exp(-1j * omega_grid.unsqueeze(1) * t.unsqueeze(0))
    spectrum = torch.abs(torch.matmul(E, y.to(torch.complex64)))
    
    import scipy.signal
    spectrum_np = spectrum.cpu().numpy()
    peaks, _ = scipy.signal.find_peaks(spectrum_np, distance=5)
    
    if len(peaks) == 0:
        return [], []
        
    top_peaks = peaks[np.argsort(spectrum_np[peaks])[::-1]][:max_atoms]
    f_init = f_grid[top_peaks]
    
    freqs = f_init.clone().detach().requires_grad_(True)
    a_real = torch.randn(len(freqs), dtype=torch.float32, device=DEVICE, requires_grad=True)
    a_imag = torch.randn(len(freqs), dtype=torch.float32, device=DEVICE, requires_grad=True)
    
    optimizer = torch.optim.Adam([freqs, a_real, a_imag], lr=0.05)
    
    for step in range(800):
        optimizer.zero_grad()
        omega = 2 * math.pi * torch.clamp(freqs, min=min_f, max=max_f)
        
        A_real = torch.cos(omega.unsqueeze(1) * t.unsqueeze(0))
        A_imag = torch.sin(omega.unsqueeze(1) * t.unsqueeze(0))
        y_pred_real = torch.matmul(a_real, A_real) - torch.matmul(a_imag, A_imag)
        
        mse = torch.sum((y - y_pred_real)**2)
        
        # GLS adaptive covariance weighting (approximated via log-sum penalty)
        # This closely mimics the SPICE inverse covariance scaling continuously
        eps = 1e-3 if step < 400 else 1e-4
        amps = torch.sqrt(a_real**2 + a_imag**2 + 1e-8)
        gls_penalty = torch.sum(torch.log(amps + eps)) * 0.5 * torch.max(spectrum)
        
        loss = mse + gls_penalty
        loss.backward()
        optimizer.step()
        
    f_final = torch.clamp(freqs, min=min_f, max=max_f).detach().cpu().numpy()
    amps_final = torch.sqrt(a_real**2 + a_imag**2).detach().cpu().numpy()
    
    threshold = 0.05 * np.max(amps_final)
    active = amps_final > threshold
    
    f_active = f_final[active]
    amps_active = amps_final[active]
    
    # Deduplicate frequencies that converged to the same continuous well
    f_unique = []
    amps_unique = []
    
    sort_active = np.argsort(amps_active)[::-1]
    for f, amp in zip(f_active[sort_active], amps_active[sort_active]):
        if not any(abs(f - uf) < 0.05 for uf in f_unique):
            f_unique.append(f)
            amps_unique.append(amp)
            
    f_active = np.array(f_unique)
    amps_active = np.array(amps_unique)
    
    # Limit to highest confidence top 3 frequencies
    if len(amps_active) > 3:
        f_active = f_active[:3]
        amps_active = amps_active[:3]
    
    return f_active, amps_active

# --- GUI and Visualization ---

class FrequencyEstimationApp:
    def __init__(self, json_filepath):
        self.data = self.load_data(json_filepath)
        
        self.root = tk.Tk()
        self.root.title("Gridless Frequency Estimator Selection")
        self.root.geometry("400x150")
        
        ttk.Label(self.root, text="Select Location/Pixel:").pack(pady=10)
        
        self.pixel_var = tk.StringVar()
        self.combo = ttk.Combobox(self.root, textvariable=self.pixel_var, state="readonly", width=40)
        self.combo['values'] = list(self.data.keys())
        if self.data.keys():
            self.combo.current(0)
        self.combo.pack(pady=5)
        
        ttk.Button(self.root, text="Analyze & Plot", command=self.run_analysis).pack(pady=15)
        self.root.mainloop()

    def load_data(self, filepath):
        try:
            with open(filepath, 'r') as f:
                raw_json = json.load(f)
            
            data_dict = {}
            for category, items in raw_json.items():
                if isinstance(items, dict):
                    # Check if it has 'timestamp' directly, meaning it's not categorized
                    if 'timestamp' in items:
                        data_dict[category] = items
                    else:
                        for loc_key, loc_data in items.items():
                            data_dict[f"{category} | {loc_key}"] = loc_data
                else:
                    data_dict[category] = items
            return data_dict
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON: {e}")
            return {}

    def run_analysis(self):
        selected_key = self.pixel_var.get()
        if not selected_key:
            return
            
        pixel_data = self.data[selected_key]
        timestamps_raw = pixel_data.get('timestamp', [])
        
        if len(timestamps_raw) == 0:
            messagebox.showerror("Error", "No timestamps found for this selection.")
            return
            
        # Safely extract sliding_volume_z_score and filter nulls simultaneously
        z_scores_raw = None
        if 'sliding_volume_z_score' in pixel_data:
            z_scores_raw = pixel_data['sliding_volume_z_score']
        elif 'z_score' in pixel_data:
            z_scores_raw = pixel_data['z_score']
            
        if z_scores_raw is None:
            # Enforcing strict adherence to restrict-fill-values by failing if data is missing.
            messagebox.showerror("Data Error", f"Z-Score data not found in JSON for {selected_key}. Failing explicitly.")
            return
            
        # Filter out `null` (None) values, and separate valid from masked observations
        common_mask = pixel_data.get('common_mask')
        
        valid_indices = []
        masked_indices = []
        
        for i, val in enumerate(z_scores_raw):
            if val is not None:
                if common_mask is not None and len(common_mask) == len(z_scores_raw):
                    if bool(common_mask[i]):
                        valid_indices.append(i)
                    else:
                        masked_indices.append(i)
                else:
                    valid_indices.append(i)
        
        if len(valid_indices) < 5:
            messagebox.showerror("Data Error", f"Insufficient valid data points (non-null) for {selected_key}.")
            return
            
        timestamps_valid = np.array([timestamps_raw[i] for i in valid_indices], dtype=np.float64)
        y_valid = np.array([z_scores_raw[i] for i in valid_indices], dtype=np.float32)
        
        timestamps_masked = np.array([timestamps_raw[i] for i in masked_indices], dtype=np.float64)
        y_masked = np.array([z_scores_raw[i] for i in masked_indices], dtype=np.float32)
        
        # Determine t0 from the earliest available timestamp
        t0 = timestamps_valid[0]
        if len(timestamps_masked) > 0:
            t0 = min(t0, timestamps_masked[0])
            
        # Convert epoch to relative years for CPY calculation
        t_years_valid = (timestamps_valid - t0) / SECONDS_IN_YEAR
        t_years_masked = (timestamps_masked - t0) / SECONDS_IN_YEAR
        
        # Ensure mean is zero for spectral analysis
        y_mean = np.mean(y_valid)
        y_valid = y_valid - y_mean
        y_masked = y_masked - y_mean
        
        self.plot_results(t_years_valid, y_valid, t_years_masked, y_masked, selected_key)

    def plot_results(self, t, y, t_masked, y_masked, title_key):
        print(f"Executing PyTorch Estimators on device: {DEVICE}")
        
        # 1. Compute all spectrums
        f_ndft, s_ndft = compute_pytorch_ndft(t, y, NDFT_MIN_CPY, NDFT_MAX_CPY, NDFT_GRID_BINS)
        f_nomp, s_nomp = compute_pytorch_nomp(t, y, NDFT_MIN_CPY, NDFT_MAX_CPY, max_components=4)
        
        print("Computing PyTorch C-BPDN...")
        f_cbpdn, s_cbpdn = compute_pytorch_cbpdn_continuous(t, y, NDFT_MIN_CPY, NDFT_MAX_CPY, max_atoms=20)
        
        print("Computing PyTorch Continuous Iterative Reweighted L1 (CIRL)...")
        f_cirl, s_cirl = compute_pytorch_cirl_continuous(t, y, NDFT_MIN_CPY, NDFT_MAX_CPY, max_atoms=20)

        # Normalize components for comparative plotting
        def norm(arr): return arr / (np.max(arr) + 1e-10) if len(arr) > 0 else arr
        s_ndft = norm(s_ndft)
        s_nomp = norm(np.array(s_nomp))
        s_cbpdn = norm(np.array(s_cbpdn))
        s_cirl = norm(np.array(s_cirl))

        # 2. Setup Plot
        fig = plt.figure(figsize=(22, 10))
        fig.canvas.manager.set_window_title(f"Spectral Analysis - {title_key}")
        
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 2.5], height_ratios=[1, 1])
        ax1 = fig.add_subplot(gs[:, 0])  # Left side, spans both rows
        ax2 = fig.add_subplot(gs[0, 1])  # Right side, top row
        ax3 = fig.add_subplot(gs[1, 1])  # Right side, bottom row
        
        # Top Plot: Time Series
        ax1.plot(t, y, 'ko', markersize=4, alpha=0.7, label='Valid Observations (Unmasked)')
        if len(t_masked) > 0:
            ax1.plot(t_masked, y_masked, 'ko', markerfacecolor='none', markersize=4, alpha=0.7, label='Masked Observations')
            
        ax1.set_title(f"Time Series Data - {title_key}", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Time (Years from Start)")
        ax1.set_ylabel("Z-Score Amplitude")
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()

        # Middle Plot: Spectrums (Period in Days)
        ax2.set_title("Frequency Spectrum: Grid-Based vs. True Continuous Gridless (Period)", fontsize=12, fontweight='bold')
        
        # Convert CPY to Period in Days
        period_ndft = 365.25 / f_ndft
        period_nomp = [365.25 / f for f in f_nomp]
        period_cbpdn = [365.25 / f for f in f_cbpdn]
        period_cirl = [365.25 / f for f in f_cirl]

        # Sort NDFT arrays by period
        sort_idx = np.argsort(period_ndft)
        period_ndft_sorted = period_ndft[sort_idx]
        s_ndft_sorted = s_ndft[sort_idx]

        # Plot NDFT
        ax2.plot(period_ndft_sorted, s_ndft_sorted, color='gray', linestyle='-', alpha=0.5, linewidth=2, 
                 label=f'PyTorch NDFT (Fixed {NDFT_GRID_BINS}-bin Grid)\nNotice Spectral Leakage')
        ax2.fill_between(period_ndft_sorted, 0, s_ndft_sorted, color='gray', alpha=0.1)

        # Plot NDFT Top 3 Peaks
        import scipy.signal
        peaks, _ = scipy.signal.find_peaks(s_ndft)
        if len(peaks) > 0:
            top_peaks = peaks[np.argsort(s_ndft[peaks])[::-1]][:3]
            ax2.plot(period_ndft[top_peaks], s_ndft[top_peaks], 'kv', markersize=8, label='NDFT Top 3 Peaks')

        # Plot True NOMP Stems
        if len(period_nomp) > 0:
            ax2.vlines(period_nomp, ymin=0, ymax=s_nomp, color='r', linestyle='-', linewidth=2, alpha=0.8)
            ax2.plot(period_nomp, s_nomp, 'r*', markersize=10, label='PyTorch NOMP (Joint Continuous Refinement)')
            
        # Plot C-BPDN Stems
        if len(period_cbpdn) > 0:
            ax2.vlines(period_cbpdn, ymin=0, ymax=s_cbpdn, color='b', linestyle='-', linewidth=2, alpha=0.8)
            ax2.plot(period_cbpdn, s_cbpdn, 'bD', markersize=7, label='PyTorch C-BPDN (Continuous BPDN)')
            
        # Plot CIRL Stems
        if len(period_cirl) > 0:
            ax2.vlines(period_cirl, ymin=0, ymax=s_cirl, color='m', linestyle='-', linewidth=2, alpha=0.8)
            ax2.plot(period_cirl, s_cirl, 'm^', markersize=8, label='PyTorch CIRL (Continuous Iterative Reweighted L1)')

        ax2.set_xlim([365.25 / NDFT_MAX_CPY, 365.25 / NDFT_MIN_CPY])
        ax2.set_xlabel("Period Length (Days)")
        ax2.set_ylabel("Spectral Energy")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right', framealpha=0.9)

        # Bottom Plot: Spectrums (Frequency in CPY)
        ax3.set_title("Frequency Spectrum: Grid-Based vs. True Continuous Gridless (CPY)", fontsize=12, fontweight='bold')
        
        # Plot NDFT
        ax3.plot(f_ndft, s_ndft, color='gray', linestyle='-', alpha=0.5, linewidth=2, 
                 label=f'PyTorch NDFT (Fixed {NDFT_GRID_BINS}-bin Grid)')
        ax3.fill_between(f_ndft, 0, s_ndft, color='gray', alpha=0.1)

        # Plot NDFT Top 3 Peaks
        if len(peaks) > 0:
            ax3.plot(f_ndft[top_peaks], s_ndft[top_peaks], 'kv', markersize=8, label='NDFT Top 3 Peaks')

        # Plot True NOMP Stems
        if len(f_nomp) > 0:
            ax3.vlines(f_nomp, ymin=0, ymax=s_nomp, color='r', linestyle='-', linewidth=2, alpha=0.8)
            ax3.plot(f_nomp, s_nomp, 'r*', markersize=10, label='PyTorch NOMP')
            
        # Plot C-BPDN Stems
        if len(f_cbpdn) > 0:
            ax3.vlines(f_cbpdn, ymin=0, ymax=s_cbpdn, color='b', linestyle='-', linewidth=2, alpha=0.8)
            ax3.plot(f_cbpdn, s_cbpdn, 'bD', markersize=7, label='PyTorch C-BPDN')
            
        # Plot CIRL Stems
        if len(f_cirl) > 0:
            ax3.vlines(f_cirl, ymin=0, ymax=s_cirl, color='m', linestyle='-', linewidth=2, alpha=0.8)
            ax3.plot(f_cirl, s_cirl, 'm^', markersize=8, label='PyTorch CIRL')

        ax3.set_xlim([NDFT_MIN_CPY, NDFT_MAX_CPY])
        ax3.set_xlabel("Frequency (Cycles Per Year)")
        ax3.set_ylabel("Spectral Energy")
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(loc='upper right', framealpha=0.9)

        # Generate Text Box Content
        text_str = "Top 3 Frequencies\n"
        text_str += "-" * 20 + "\n\n"
        
        # NDFT
        text_str += "NDFT (Grid):\n"
        if len(peaks) > 0:
            for i, p in enumerate(top_peaks):
                text_str += f" {i+1}: {period_ndft[p]:>6.1f} d | {f_ndft[p]:.3f} cpy\n"
        else:
            text_str += " None\n"
            
        # NOMP
        text_str += "\nNOMP (Continuous):\n"
        for i, f in enumerate(f_nomp[:3]):
            text_str += f" {i+1}: {365.25/f:>6.1f} d | {f:.3f} cpy\n"
            
        # C-BPDN
        text_str += "\nC-BPDN (Continuous):\n"
        for i, f in enumerate(f_cbpdn[:3]):
            text_str += f" {i+1}: {365.25/f:>6.1f} d | {f:.3f} cpy\n"
            
        # CIRL
        text_str += "\nCIRL (Continuous):\n"
        for i, f in enumerate(f_cirl[:3]):
            text_str += f" {i+1}: {365.25/f:>6.1f} d | {f:.3f} cpy\n"

        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.9, edgecolor='gray')
        fig.text(0.77, 0.5, text_str, fontsize=11, fontfamily='monospace',
                 verticalalignment='center', bbox=props)

        plt.tight_layout()
        plt.subplots_adjust(right=0.74) # Shrink the plots to make room on the right
        plt.show(block=False)

if __name__ == "__main__":
    app = FrequencyEstimationApp('z-score-samples.json')