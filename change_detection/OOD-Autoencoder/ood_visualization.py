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

# Attempt to use scienceplots if available
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

from dataset import TimeSeriesH5Dataset
from models import FrequencyAutoencoder

LOCATION = "Tait"
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
    
    # Wrapper for PnPXAI to output a scalar MAE loss
    class MAEWrapper(nn.Module):
        def __init__(self, ae_model):
            super(MAEWrapper, self).__init__()
            self.ae = ae_model
            
        def forward(self, x):
            true_amps, rec_amps = self.ae(x)
            mae = torch.mean(torch.abs(true_amps - rec_amps), dim=1, keepdim=True)
            return mae
            
    wrapper_model = MAEWrapper(model).to(device)
    wrapper_model.eval()
    explainer = IntegratedGradients(wrapper_model)

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
    
    H, W = ood_map.shape
    
    fig, (ax_img, ax_ts) = plt.subplots(1, 2, figsize=(18, 8))
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
        
        # Calculate flat index and retrieve data
        idx = y * W + x
        pixel_ts, _, _ = dataset[idx]
        pixel_ts = pixel_ts.to(device)
        interp_mask = dataset.interpolation_mask[idx]
        
        # Dynamically run explainability
        pixel_ts_input = pixel_ts.unsqueeze(0)
        pixel_ts_input.requires_grad_()
        
        attributions = explainer.attribute(pixel_ts_input, targets=0)
        attrs_np = attributions.cpu().detach().numpy().flatten()
        z_scores = pixel_ts.cpu().detach().numpy()
        
        ax_ts.clear()
        ax2.clear()
        
        # Plot Time Series Data
        pixel_cmask = common_mask_volume[:, y, x]
        
        syn_mask = interp_mask
        real_mask = ~syn_mask
        cmask = (pixel_cmask > 0) & real_mask
        valid_mask = real_mask & (~cmask)
        
        for marker_type, sc_keyword in [('s', 'Sentinel'), ('o', 'Landsat'), ('D', 'Tanager')]:
            sc_mask = np.array([sc_keyword.lower() in str(sc).lower() for sc in spacecrafts_arr])
            
            idx_valid = valid_mask & sc_mask
            if np.any(idx_valid):
                ax_ts.plot(dates_arr[idx_valid], z_scores[idx_valid], color='k', marker=marker_type, linestyle='None', markersize=4, label=f'Real Obs ({sc_keyword})')
                
            idx_cmask = cmask & sc_mask
            if np.any(idx_cmask):
                ax_ts.plot(dates_arr[idx_cmask], z_scores[idx_cmask], color='gray', marker=marker_type, linestyle='None', markersize=4, label=f'QA Masked ({sc_keyword})')
                
            idx_syn = syn_mask & sc_mask
            if np.any(idx_syn):
                ax_ts.plot(dates_arr[idx_syn], z_scores[idx_syn], color='w', markeredgecolor='gray', marker=marker_type, linestyle='None', markersize=4, label=f'Interpolated ({sc_keyword})')
            
        # Highlight the date of maximum attribution
        max_attr_idx = np.argmax(attrs_np)
        ax_ts.axvline(x=dates_arr[max_attr_idx], color='red', linestyle='--', linewidth=2, alpha=0.8, label='Max Attribution (Peak Anomaly)')
            
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
        max_attr = np.max(np.abs(attrs_np))
        if max_attr == 0: max_attr = 1
        # Set bounds artificially wide so the bars don't swamp the dot plot
        ax2.set_ylim(-max_attr * 1.5, max_attr * 1.5) 
        
        # Plot Attribution Bars
        width_days = 8 # Width of bars in matplotlib datetime units
        if np.any(real_mask):
            ax2.bar(dates_arr[real_mask], attrs_np[real_mask], width=width_days, color='purple', alpha=0.5, label='Attribution (Real)')
        if np.any(syn_mask):
            ax2.bar(dates_arr[syn_mask], attrs_np[syn_mask], width=width_days, color='orange', alpha=0.7, hatch='///', label='Attribution (Synthetic Risk)')
        
        # Consolidate Legends
        lines_1, labels_1 = ax_ts.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax_ts.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        
        ax_ts.xaxis.set_major_locator(mdates.YearLocator())
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(OOD_MAP_PATH) or not os.path.exists(MODEL_PATH):
        print("Required files not found. Please run OOD_autoencoder_main.py first.")
    else:
        main()
