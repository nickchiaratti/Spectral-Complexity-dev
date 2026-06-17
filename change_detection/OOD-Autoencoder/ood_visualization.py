import h5py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os
import torch
import torch.nn as nn
from matplotlib.cm import Reds
import matplotlib.patches as patches
import matplotlib.dates as mdates
from pnpxai.explainers import IntegratedGradients
from torch.utils.data import DataLoader

# Attempt to use scienceplots if available
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

from dataset import TimeSeriesH5Dataset
from models import FrequencyAutoencoder

LOCATION = "Malibu"
H5_PATH = rf"E:\satelliteImagery\HLST30\HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
DATASET_NAME = "HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score"
OOD_MAP_PATH = f"E:/satelliteImagery/HLST30/OOD/{LOCATION}/{LOCATION}_ood_results.h5"
MODEL_PATH = f"E:/satelliteImagery/HLST30/OOD/{LOCATION}/{LOCATION}_ood_model.pth"

LATENT_DIM = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_ortho(source_h5_path, target_date_str="2025-09-12"):
    with h5py.File(source_h5_path, 'r') as f:
        harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        acq_time = harm_grp['sliding_volume_z_score'].attrs['acquisition_time'][:]
        dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
        
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).date()
        diffs = [abs((d.date() - target_date).days) for d in dates]
        base_idx = np.argmin(diffs)
        
        spc = harm_grp['sliding_volume_z_score'].attrs['source_spacecraft'][base_idx]
        spc = spc.decode('utf-8') if isinstance(spc, bytes) else str(spc)
        
        o = harm_grp['ortho_visual'][base_idx]
        o = np.transpose(o, (1, 2, 0)).astype(np.float32) / 255.0
        
        valid_mask = np.all(o > 0, axis=-1)
        o[~valid_mask] = 0.0 # Set NoData to black
        
        return o, spc, dates[base_idx], acq_time

def main():
    print("Loading dataset in memory (this may take a moment)...")
    dataset = TimeSeriesH5Dataset(h5_path=H5_PATH, dataset_name=DATASET_NAME)
    
    print("Loading trained PyTorch model...")
    model = FrequencyAutoencoder(sequence_length=dataset.time_steps, latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # PnPXAI wrapper removed here, dynamically constructed in onclick

    print("Computing mean dataset frequency representation...")
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_freq_amps = []
    with torch.no_grad():
        for pts, vals, _ in loader:
            pts, vals = pts.to(device), vals.to(device)
            f_amps, _ = model(pts, vals)
            all_freq_amps.append(f_amps.cpu())
    mean_frame_freq = torch.cat(all_freq_amps, dim=0).mean(dim=0).numpy().flatten()

    print("Loading inference maps...")
    with h5py.File(OOD_MAP_PATH, 'r') as f:
        ood_map = f['ood_map'][:].astype(bool)
        ood_time_map = f['ood_time_map'][:]
        
    print("Loading common mask...")
    with h5py.File(H5_PATH, 'r') as f:
        common_mask_volume = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'][:]
        spacecraft_bytes = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score'].attrs['source_spacecraft'][:]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in spacecraft_bytes]
        spacecrafts_arr = np.array(spacecrafts)
        
    base_frame, base_sg, base_date, acq_time = get_ortho(H5_PATH)
    dates = [datetime.fromtimestamp(ts, timezone.utc) for ts in acq_time]
    dates_arr = np.array(dates)
    total_days = (np.max(acq_time) - np.min(acq_time)) / 86400.0
    
    H, W = ood_map.shape
    
    fig, (ax_img, ax_ts, ax_freq) = plt.subplots(1, 3, figsize=(24, 8))
    ax2 = ax_ts.twinx()
    ax2.set_yticks([])
    fig.canvas.manager.set_window_title('OOD Anomaly Explorer & Explainability Viewer')
    
    # 1. Plot Base Image
    ax_img.imshow(base_frame)
    ax_img.set_title(f"True Color Backdrop ({base_sg}): {base_date.strftime('%Y-%m-%d')} UTC")
    
    # 2. Overlay Temporal OOD Map
    if np.any(ood_map):
        # Mask where not OOD, or where OOD but driven by synthetic data (-1)
        masked_time = np.ma.masked_where((~ood_map) | (ood_time_map < 0), ood_time_map)
        
        cmap = plt.cm.jet
        cmap.set_bad(color='white', alpha=0)
        
        im = ax_img.imshow(masked_time, cmap=cmap, alpha=0.7, vmin=0, vmax=len(dates_arr)-1)
        
        # Add a colorbar mapped to dates
        cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
        cbar.set_label('Date of Maximum Anomaly Attribution')
        
        # Format colorbar ticks as dates
        num_ticks = 6
        tick_locs = np.linspace(0, len(dates_arr)-1, num_ticks, dtype=int)
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels([dates_arr[i].strftime('%Y-%m') for i in tick_locs])
        
    rect = patches.Rectangle((-1, -1), 1, 1, linewidth=2, edgecolor='cyan', facecolor='none', visible=False)
    ax_img.add_patch(rect)
    
    ax_ts.text(0.5, 0.5, 'Click a pixel on the spatial map\nto dynamically run PnPXAI explainability.', 
               horizontalalignment='center', verticalalignment='center', transform=ax_ts.transAxes, fontsize=12)

    def onclick(event):
        if event.inaxes != ax_img: return
        x, y = int(event.xdata), int(event.ydata)
        if x < 0 or x >= W or y < 0 or y >= H: return
        
        print(f"Evaluating pixel X:{x}, Y:{y}...")
        
        rect.set_xy((x - 0.5, y - 0.5))
        rect.set_visible(True)
        
        # Load the raw z-scores for this pixel directly from the HDF5 file
        with h5py.File(H5_PATH, 'r') as f:
            z_scores = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score'][:, y, x]
            
        # Create a new mask that ONLY excludes NaNs, ignoring the common_mask
        # This allows the NUFFT to see the "outliers" which were previously masked
        invalid_mask = np.isnan(z_scores)
        raw_values = np.where(invalid_mask, 0.0, z_scores)
        
        points = dataset.points.to(device)
        values = torch.tensor(raw_values, dtype=torch.float32).to(device)
        
        # Dynamically run explainability
        points_input = points.unsqueeze(0)
        values_input = values.unsqueeze(0)
        values_input.requires_grad_()
        
        # Wrapper to attribute wrt values only
        class PnPXAIWrapper(nn.Module):
            def __init__(self, ae_model, fixed_points):
                super().__init__()
                self.ae = ae_model
                self.fixed_points = fixed_points
            def forward(self, vals):
                true_amps, rec_amps = self.ae(self.fixed_points, vals)
                return torch.mean(torch.abs(true_amps - rec_amps), dim=1, keepdim=True)
                
        wrapper = PnPXAIWrapper(model, points_input).to(device)
        wrapper.eval()
        explainer = IntegratedGradients(wrapper)
        
        attributions = explainer.attribute(values_input, targets=0)
        attrs_np = attributions.cpu().detach().numpy().flatten()
        
        # Mask out attributions on invalid points so they are not selected as peak
        attrs_np[invalid_mask] = -99999.0
        
        # Extract True Frequencies and Reconstructed Frequencies from the model directly
        true_amps, rec_amps = model(points_input, values_input)
        freq_true_np = true_amps.cpu().detach().numpy().flatten()
        freq_rec_np = rec_amps.cpu().detach().numpy().flatten()
        
        # z_scores is already loaded above
        
        ax_ts.clear()
        ax2.clear()
        
        # Plot Time Series Data
        pixel_cmask = common_mask_volume[:, y, x]
        
        # There is no synthetic data anymore, just valid and masked
        cmask = pixel_cmask > 0
        real_mask = ~cmask
        valid_mask = real_mask & ~np.isnan(z_scores)
        
        for marker_type, sc_keyword in [('s', 'Sentinel'), ('o', 'Landsat'), ('D', 'Tanager')]:
            sc_mask = np.array([sc_keyword.lower() in str(sc).lower() for sc in spacecrafts_arr])
            
            idx_valid = valid_mask & sc_mask
            if np.any(idx_valid):
                ax_ts.plot(dates_arr[idx_valid], z_scores[idx_valid], color='k', marker=marker_type, linestyle='None', markersize=4, label=f'Real Obs ({sc_keyword})')
                
            idx_cmask = cmask & sc_mask
            if np.any(idx_cmask):
                ax_ts.plot(dates_arr[idx_cmask], z_scores[idx_cmask], color='gray', markerfacecolor='none', marker=marker_type, linestyle='None', markersize=4, label=f'QA Masked ({sc_keyword})')
            
        # Highlight the date of maximum attribution
        max_attr_idx = np.argmax(attrs_np)
        ax_ts.axvline(x=dates_arr[max_attr_idx], color='red', linestyle='--', linewidth=2, alpha=0.8, zorder=0, label='Max Attribution (Peak Anomaly)')
            
        ax_ts.set_title(f"Out-of-Distribution Autoencoder Profile | X:{x}, Y:{y}\nOOD Flagged: {'YES (Anomalous)' if ood_map[y, x] else 'NO (Normal)'}", fontweight="bold")
        ax_ts.set_ylabel("Spectral Complexity (Z-Score)")
        ax_ts.set_xlabel("Acquisition Date")
        ax_ts.grid(True, linestyle='--', alpha=0.3)
        ax_ts.set_ylim(-5, 5)
        
        # Plot Attributions on Secondary Axis
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        ax2.set_ylabel("PnPXAI Attribution Weight (Contribution to Anomaly Score)", color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        
        # Calculate absolute max to make symmetrical Y axis
        # We must ignore the invalid mask which is set to -99999.0
        valid_attrs = np.where(valid_mask, attrs_np, 0.0)
        max_attr = np.max(np.abs(valid_attrs))
        if max_attr == 0: max_attr = 1
        # Set bounds artificially wide so the bars don't swamp the dot plot
        ax2.set_ylim(-max_attr * 1.5, max_attr * 1.5) 
        
        # Plot Attribution Bars
        width_days = 8 # Width of bars in matplotlib datetime units
        if np.any(valid_mask):
            # Only plot positive/valid attribution bars (invalid were masked to -99999)
            valid_attrs = np.where(valid_mask, attrs_np, 0.0)
            ax2.bar(dates_arr[valid_mask], valid_attrs[valid_mask], width=width_days, color='purple', alpha=0.5, label='Attribution (Real)')
        
        # Consolidate Legends
        lines_1, labels_1 = ax_ts.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax_ts.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        
        ax_ts.xaxis.set_major_locator(mdates.YearLocator())
        
        # Plot Frequency Domain Data
        ax_freq.clear()
        
        # Remove twin axis if it exists
        if hasattr(ax_freq, 'twin_period_ax'):
            try:
                ax_freq.twin_period_ax.remove()
            except:
                pass
            
        # Exclude DC (Bin 0) to avoid infinite period issues
        freq_bins = np.arange(1, len(freq_true_np))
        ax_freq.plot(freq_bins, freq_true_np[1:], color='blue', label='True Signal Spectrum')
        ax_freq.plot(freq_bins, freq_rec_np[1:], color='orange', linestyle='--', label='Reconstructed Spectrum')
        ax_freq.plot(freq_bins, mean_frame_freq[1:], color='green', alpha=0.6, linestyle='-', label='Mean Frame Spectrum')
        ax_freq.set_title(f"Frequency Domain Representation (Excluding DC)", fontweight="bold")
        ax_freq.set_xlabel("Cycle Length (Days)")
        ax_freq.set_ylabel("Amplitude")
        ax_freq.legend(loc='upper right')
        ax_freq.grid(True, linestyle='--', alpha=0.3)
        
        # Use a FuncFormatter to translate the linear frequency bins into Period in Days for the x-axis labels
        import matplotlib.ticker as ticker
        def format_fn(tick_val, tick_pos):
            if tick_val <= 0: return ""
            return f"{total_days / tick_val:.0f}"
            
        ax_freq.xaxis.set_major_formatter(ticker.FuncFormatter(format_fn))
        
        # Exclude DC (Bin 0) from the top peaks statistics
        top3_px_idx = np.argsort(freq_true_np[1:])[-3:][::-1] + 1
        top3_px_vals = freq_true_np[top3_px_idx]
        
        top3_fr_idx = np.argsort(mean_frame_freq[1:])[-3:][::-1] + 1
        top3_fr_vals = mean_frame_freq[top3_fr_idx]
        
        def bin_to_days(k):
            return "DC" if k == 0 else f"{total_days / k:.1f}d"
            
        px_text = "Pixel Top 3 Freqs:\n" + "\n".join([f"Bin {k} ({bin_to_days(k)}): {v:.2f}" for k, v in zip(top3_px_idx, top3_px_vals)])
        fr_text = "Frame Top 3 Freqs:\n" + "\n".join([f"Bin {k} ({bin_to_days(k)}): {v:.2f}" for k, v in zip(top3_fr_idx, top3_fr_vals)])
        
        text_str = px_text + "\n\n" + fr_text
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        # Place text box on ax_freq inside the plot near bottom right or center right
        ax_freq.text(0.95, 0.5, text_str, transform=ax_freq.transAxes, fontsize=10,
                     verticalalignment='center', horizontalalignment='right', bbox=props)
        
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(OOD_MAP_PATH) or not os.path.exists(MODEL_PATH):
        print("Required files not found. Please run OOD_autoencoder_main.py first.")
    else:
        main()
