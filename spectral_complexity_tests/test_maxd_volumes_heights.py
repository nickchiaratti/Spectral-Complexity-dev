import numpy as np
import torch
import sys
import os

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import SpecComplex
import SpecComplexTorch

def plot_pickleball_heights():
    import matplotlib.pyplot as plt
    from datetime import datetime
    import h5py

    file_path = "C:/satelliteImagery/MGRS30mConstellation/Harmonized_MGRS_Stack_Rochesterv2.h5"
    if not os.path.exists(file_path):
        print(f"Data file not found: {file_path}")
        return

    pixel_y, pixel_x = 706, 771#771,706#
    window_size = 3
    half_w = window_size // 2
    k = window_size**2
    
    target_keys = ['HLSL30', 'HLSS30', 'TANAGER', 'ENMAP', 'DRAGONETTE', 'Other']
    title_map = {
        'HLSL30': "Landsat (HLSL30)",
        'HLSS30': "Sentinel-2 (HLSS30)",
        'TANAGER': "Tanager (TANAGER)",
        'ENMAP': "EnMAP",
        'DRAGONETTE': "Dragonette",
        'Other': "Other Grids"
    }
    
    # Collect data to plot
    plot_data = {key: [] for key in target_keys}
    
    # Process each grid
    with h5py.File(file_path, 'r') as f:
        all_grids = list(f['HDFEOS/GRIDS'].keys())
        for grid_name in all_grids:
            plot_key = 'Other'
            for tk in ['HLSL30', 'HLSS30', 'TANAGER', 'ENMAP', 'DRAGONETTE']:
                if tk in grid_name:
                    plot_key = tk
                    break
            
            try:
                ds = f[f'HDFEOS/GRIDS/{grid_name}/Data Fields/surface_reflectance']
            except KeyError:
                continue
                
            times = ds.attrs.get('acquisition_time', [])
            
            if len(times) == 0:
                continue
            
            # Filter for 2022 through 2026 inclusive
            valid_indices = []
            for i, t_sec in enumerate(times):
                try:
                    dt = datetime.fromtimestamp(t_sec)
                    if 2020 <= dt.year <= 2026:
                        valid_indices.append(i)
                except Exception:
                    pass
            
            if not valid_indices:
                continue
                
            plotted_lines = 0
            for idx in valid_indices:
                # Extract local window for endmember calculation
                patch = SpecComplex.load_scaled_reflectance(ds, np.s_[idx, :, pixel_y-half_w:pixel_y+half_w+1, pixel_x-half_w:pixel_x+half_w+1])
                
                if "TANAGER" in grid_name:
                    gw_mask = ds.attrs.get("all_good_wavelengths")[idx].astype(bool)
                    patch = np.delete(patch, np.where(~gw_mask), axis=0)

                # Rigid failure filter: do not interpolate or fill. Skip if invalid data is present.
                if np.any(patch < 0) or np.any(patch > 1.0) or np.any(np.isnan(patch)):
                    continue
                
                C = patch.shape[0]
                N = patch.shape[1] * patch.shape[2]
                #if N < k:
                #    continue
                
                patch_flat = patch.reshape(1, C, N)
                data_torch = torch.from_numpy(patch_flat)
                
                try:
                    _, v_t, h_t = SpecComplexTorch.maximumDistance_volumes_torch_test(data_torch, k)
                    h_np = h_t[0].cpu().numpy()
                    v_np = v_t[0].cpu().numpy()
                    
                    # Store data along with its acquisition time
                    plot_data[plot_key].append((h_np, v_np, C, times[idx]))
                    plotted_lines += 1
                except Exception as e:
                    # Allows algorithm failures on rank deficient or highly collinear patches
                    continue
                    
            print(f"[{grid_name}] Plotted {plotted_lines} temporal patches.")

    # Separate into top and bottom rows
    top_keys = ['HLSL30', 'HLSS30']
    top_panels = [key for key in top_keys if len(plot_data[key]) > 0]
    if not top_panels: # Fallback just in case
        top_panels = top_keys
        
    bottom_keys = [key for key in target_keys if key not in top_keys]
    bottom_panels = [key for key in bottom_keys if len(plot_data[key]) > 0]

    num_top = len(top_panels)
    num_bottom = len(bottom_panels)
    
    if num_top == 0 and num_bottom == 0:
        print("No valid data found to plot.")
        return

    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates

    # Use fixed day-of-year normalization (1 to 366)
    cmap = plt.get_cmap('hsv')
    norm = mcolors.Normalize(vmin=1, vmax=366)

    # Use max width between the two rows for a balanced figure
    max_cols = max(num_top, num_bottom, 1)
    fig = plt.figure(figsize=(5 * max_cols, 10))
    
    # Plot top row
    for i, plot_key in enumerate(top_panels):
        ax = fig.add_subplot(2, num_top, i + 1)
        ax.set_title(title_map[plot_key])
        ax.set_xlabel("Endmember Index (n)")
        if i == 0:
            ax.set_ylabel("Orthogonal Height ($h_n$)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        for h_vals, v_vals, C_bands, t_sec in plot_data[plot_key]:
            dt = datetime.fromtimestamp(t_sec)
            day_of_year = dt.timetuple().tm_yday
            color = cmap(norm(day_of_year))
            ax.plot(range(k), h_vals, color=color, linewidth=0.5, alpha=0.6)
            ax.set_yscale('log')
            
    # Plot bottom row
    for i, plot_key in enumerate(bottom_panels):
        ax = fig.add_subplot(2, max(num_bottom, 1), max(num_bottom, 1) + i + 1)
        ax.set_title(title_map[plot_key])
        ax.set_xlabel("Endmember Index (n)")
        if i == 0:
            ax.set_ylabel("Orthogonal Height ($h_n$)")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        for h_vals, v_vals, C_bands, t_sec in plot_data[plot_key]:
            dt = datetime.fromtimestamp(t_sec)
            day_of_year = dt.timetuple().tm_yday
            color = cmap(norm(day_of_year))
            ax.plot(range(k), h_vals, color=color, linewidth=0.5, alpha=0.6)
            ax.set_yscale('log')

    plt.tight_layout()
    
    # Add shared colorbar at the bottom or side
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation='horizontal', fraction=0.05, pad=0.1, aspect=40)
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cbar.set_ticks(month_starts)
    cbar.set_ticklabels(month_names)
    cbar.set_label('Seasonality (Month of Year)')
    out_png = os.path.join(os.path.dirname(__file__), "pickleball_heights_trend.png")
    plt.savefig(out_png, dpi=300)
    print(f"Height vs Endmember trend plot saved to {out_png}")

    # --- New Figure 1: Ratio of h_k to V_k / V_{k-1} ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set_title("Empirical Verification of $h_k = V_k / V_{k-1}$")
    ax2.set_xlabel("Endmember Index (k)")
    ax2.set_ylabel("$h_k / (V_k / V_{k-1})$")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # --- New Figure 2: Ratio of log(h_k) to log(V_k) - log(V_{k-1}) ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.set_title("Empirical Verification of $\\log(h_k) = \\log(V_k) - \\log(V_{k-1})$")
    ax3.set_xlabel("Endmember Index (k)")
    ax3.set_ylabel("$\\log(h_k) / (\\log(V_k) - \\log(V_{k-1}))$")
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    for plot_key, data_list in plot_data.items():
        if not data_list: continue
        
        all_k1 = []
        all_ratio1 = []
        all_k2 = []
        all_ratio2 = []
        
        for h_vals, v_vals, C_bands, t_sec in data_list:
            # i starts at 2 (since index 0 is origin, index 1 is first height)
            for i in range(2, len(h_vals)):
                if v_vals[i-1] > 1e-12 and h_vals[i] > 1e-12: # Avoid div by zero
                    v_ratio = v_vals[i] / v_vals[i-1]
                    if v_ratio > 1e-12:
                        all_k1.append(i)
                        all_ratio1.append(h_vals[i])# / v_ratio)
                        
                        log_h = np.log(h_vals[i])
                        log_v_ratio = np.log(v_vals[i]) - np.log(v_vals[i-1])
                        
                        if abs(log_v_ratio) > 1e-12:
                            all_k2.append(i)
                            all_ratio2.append(log_h)# / log_v_ratio)
                    
        if all_k1:
            # Add some jitter to x-axis to separate the sensor points slightly for visibility
            jitter = np.random.uniform(-0.1, 0.1, size=len(all_k1))
            ax2.scatter(np.array(all_k1), all_ratio1, label=title_map.get(plot_key, plot_key), alpha=0.5, s=10)
        if all_k2:
            jitter = np.random.uniform(-0.1, 0.1, size=len(all_k2))
            ax3.scatter(np.array(all_k2), all_ratio2, label=title_map.get(plot_key, plot_key), alpha=0.5, s=10)

    ax2.legend()
    fig2.tight_layout()
    out_png2 = os.path.join(os.path.dirname(__file__), "marginal_orthogonal_contributions.png")
    fig2.savefig(out_png2, dpi=300)
    print(f"Marginal Orthogonal Contributions plot saved to {out_png2}")

    ax3.legend()
    fig3.tight_layout()
    out_png3 = os.path.join(os.path.dirname(__file__), "log_volume_growth_rate.png")
    fig3.savefig(out_png3, dpi=300)
    print(f"Log-Volume Growth Rate plot saved to {out_png3}")

    # --- New Figure 3: Normalized k-th Root Volume ---
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.set_title(r"k-th Root Normalized Volume:  = (V_k)^{1/k} / \sqrt{N}$")
    ax4.set_xlabel("Endmember Index (k)")
    ax4.set_ylabel("Normalized Volume ($)")
    ax4.grid(True, linestyle='--', alpha=0.6)

    for plot_key, data_list in plot_data.items():
        if not data_list: continue
        all_k3 = []
        all_c_k = []
        for h_vals, v_vals, C_bands, t_sec in data_list:
            for i in range(1, len(v_vals)):
                if v_vals[i] > 1e-12:
                    all_k3.append(i)
                    c_k = (v_vals[i] ** (1.0/i)) / np.sqrt(C_bands)
                    all_c_k.append(c_k)
        if all_k3:
            jitter = np.random.uniform(-0.1, 0.1, size=len(all_k3))
            ax4.scatter(np.array(all_k3) + jitter, all_c_k, label=title_map.get(plot_key, plot_key), alpha=0.5, s=10)

    ax4.legend()
    fig4.tight_layout()
    out_png4 = os.path.join(os.path.dirname(__file__), "normalized_kth_root_volume.png")
    fig4.savefig(out_png4, dpi=300)
    print(f"Normalized k-th Root Volume plot saved to {out_png4}")

    # --- New Figure 4: Exponential Decay and Curvature ---
    fig5, (ax_logh, ax_kappa) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    
    ax_logh.set_title("Log of Orthogonal Heights (Linearizing the Exponential Decay)")
    ax_logh.set_ylabel("$\\log(h_k)$")
    ax_logh.grid(True, linestyle='--', alpha=0.6)
    
    ax_kappa.set_title("Second Derivative (Maximum Curvature Stopping Criterion)")
    ax_kappa.set_xlabel("Endmember Index (k)")
    ax_kappa.set_ylabel("Curvature $\\kappa(k)$")
    ax_kappa.grid(True, linestyle='--', alpha=0.6)

    for plot_key, data_list in plot_data.items():
        if not data_list: continue
        
        all_k = []
        all_log_h = []
        
        all_k_kappa = []
        all_kappa = []
        
        for h_vals, v_vals, C_bands, t_sec in data_list:
            log_h_list = []
            valid_k = []
            
            # Extract valid log(h_k)
            for i in range(1, len(h_vals)):
                if h_vals[i] > 1e-12:
                    log_h_list.append(np.log(h_vals[i]))
                    valid_k.append(i)
                    
            if len(log_h_list) > 0:
                all_k.extend(valid_k)
                all_log_h.extend(log_h_list)
                
            # Calculate curvature (second derivative)
            for i in range(1, len(log_h_list) - 1):
                kappa = log_h_list[i+1] - 2 * log_h_list[i] + log_h_list[i-1]
                all_k_kappa.append(valid_k[i])
                all_kappa.append(kappa)

        if all_k:
            jitter1 = np.random.uniform(-0.1, 0.1, size=len(all_k))
            ax_logh.scatter(np.array(all_k) + jitter1, all_log_h, label=title_map.get(plot_key, plot_key), alpha=0.5, s=10)
            
        if all_k_kappa:
            jitter2 = np.random.uniform(-0.1, 0.1, size=len(all_k_kappa))
            ax_kappa.scatter(np.array(all_k_kappa) + jitter2, all_kappa, alpha=0.5, s=10)

    ax_logh.legend()
    fig5.tight_layout()
    out_png5 = os.path.join(os.path.dirname(__file__), "exponential_decay_curvature.png")
    fig5.savefig(out_png5, dpi=300)
    print(f"Exponential Decay Curvature plot saved to {out_png5}")

if __name__ == "__main__":
    plot_pickleball_heights()
