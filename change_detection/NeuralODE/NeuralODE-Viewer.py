import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
import sys
import warnings

# ==========================================
# CONFIGURATION
# ==========================================
Location = "Rochester"
Frame_Reg = "WRS16" # Added to prevent NameError in SOURCE_H5 definition
ODE_H5 = f"C:/satelliteImagery/LANDSAT/{Location}/CCD_ODE_RNN_{Location}.h5"
SOURCE_H5 = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025_{Frame_Reg}_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

# ==========================================
# SCIENTIFIC VISUALIZATION RATIONALE:
# - Panel 1 & 2: Spatial distribution of structural breaks (anomalies) and their timings.
#                *Upgraded: Now rendered as an overlay on top of the SOURCE_H5 geographic landscape.
# - Panel 3: Temporal Domain validation matching continuous ODE predictions against STRICTLY VALID, un-imputed discrete observations.
# - Panel 4: Phase Space Topology. Visualizes the continuous latent manifold (hidden states). 
#            Cyclical natural phenology appears as stable orbital attractors. Anomalies appear as trajectory divergences.
#            (Ref: Rubanova, Y., et al. 2019. Latent ODEs for Irregularly-Sampled Time Series)
# ==========================================

def load_h5_data(ode_path, source_path):
    """
    Loads spatial maps from ODE_H5 and geographic context from SOURCE_H5.
    Fails loudly if required data is missing to ensure analytical rigor.
    """
    data = {}
    try:
        # 1. Load ODE Trained Parameters & Temporal Arrays
        with h5py.File(ode_path, 'r') as f:
            data['change_detected_map'] = f['change_detected_map'][:]
            data['change_date_map'] = f['change_date_map'][:]
            data['features'] = f['features'][:]
            data['masks'] = f['masks'][:]
            data['time_steps'] = f['time_steps'][:]
            data['ode_predictions'] = f['ode_predictions'][:]
            data['hidden_states'] = f['hidden_states'][:] 
            
        # 2. Load Source Visualization Dataset
        with h5py.File(source_path, 'r') as f:
            data_grp = f['/HDFEOS/GRIDS/LANDSAT/Data Fields']
            if 'ortho_visual' in data_grp:
                raw_ortho = data_grp['ortho_visual'][:]
                
                # Compute a cloud-free true color composite using the temporal median
                # This mathematically filters out transient clouds/shadows without imputation
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    rgb_composite = np.nanmedian(raw_ortho, axis=0)
                    
                # Reorder axes for Matplotlib (Height, Width, Channels) if necessary
                if rgb_composite.shape[0] in [3, 4]:
                    rgb_composite = np.transpose(rgb_composite, (1, 2, 0))
                    
                # Retain only RGB channels (drop Alpha or NIR if appended)
                if rgb_composite.shape[-1] >= 3:
                    rgb_composite = rgb_composite[:, :, :3]
                    
                # Robust 2%-98% contrast stretch for optimal visual clarity
                # Required because raw reflectance data is rarely natively scaled 0.0-1.0
                p2, p98 = np.nanpercentile(rgb_composite, (2, 98), axis=(0, 1))
                for i in range(3):
                    if p98[i] > p2[i]:
                        rgb_composite[:, :, i] = np.clip((rgb_composite[:, :, i] - p2[i]) / (p98[i] - p2[i]), 0, 1)
                    else:
                        rgb_composite[:, :, i] = 0 # Fallback for uniform dead bands
                        
                data['background_map'] = rgb_composite
            else:
                raise KeyError("CRITICAL ERROR: 'ortho_visual' not found in SOURCE_H5")
                
    except KeyError as e:
        print(f"CRITICAL ERROR: Missing expected dataset in HDF5 file: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: File not found: {e}")
        sys.exit(1)
        
    return data

def setup_dashboard(data):
    """
    Sets up the Matplotlib interactive dashboard with a 4-panel layout.
    """
    fig = plt.figure(figsize=(18, 9))
    fig.canvas.manager.set_window_title("Neural ODE Spatial-Temporal & Phase Space Viewer")

    # Grid layout: Left side maps (2x2), Right side plots (2x2)
    gs = fig.add_gridspec(2, 4)
    ax_map_det = fig.add_subplot(gs[0, :2])
    ax_map_date = fig.add_subplot(gs[1, :2])
    ax_ts = fig.add_subplot(gs[0, 2:])
    ax_phase = fig.add_subplot(gs[1, 2:])

    # --- Panel 1: Anomaly Detection Map Overlay ---
    # Plot true color background landscape for context (removed cmap='gray' since data is natively RGB)
    ax_map_det.imshow(data['background_map'], alpha=0.7, interpolation='none')
    
    # Mask out non-anomalous pixels so only the breaks are overlaid
    masked_det = np.ma.masked_where(data['change_detected_map'] == 0, data['change_detected_map'])
    cmap_binary = ListedColormap(['#e74c3c']) # Red for anomaly
    
    ax_map_det.imshow(masked_det, cmap=cmap_binary, interpolation='none')
    ax_map_det.set_title('Spatial Distribution of Detected Anomalies', fontsize=12, fontweight='bold')
    ax_map_det.axis('off')
    
    import matplotlib.patches as mpatches
    anom_patch = mpatches.Patch(color='#e74c3c', label='Anomaly Detected')
    ax_map_det.legend(handles=[anom_patch], loc='upper right')

    # --- Panel 2: Anomaly Date Map Overlay ---
    ax_map_date.imshow(data['background_map'], alpha=0.7, interpolation='none')
    
    date_map = data['change_date_map']
    masked_date_map = np.ma.masked_where(data['change_detected_map'] == 0, date_map)
    
    cmap_date = plt.cm.viridis
    cmap_date.set_bad(color='white', alpha=0) # Make missing data completely transparent
    
    date_img = ax_map_date.imshow(masked_date_map, cmap=cmap_date, interpolation='none')
    ax_map_date.set_title('Temporal Distribution (Date of Anomaly)', fontsize=12, fontweight='bold')
    ax_map_date.axis('off')
    
    cbar = plt.colorbar(date_img, ax=ax_map_date, fraction=0.046, pad=0.04)
    cbar.set_label('Fractional Year')

    # --- Panel 3 & 4 Initialization ---
    ax_ts.set_title('Temporal Dynamics (Click a pixel on the maps)', fontsize=10, style='italic')
    ax_ts.grid(True, linestyle='--', alpha=0.6)
    
    ax_phase.set_title('Latent Phase Space Topology (h_0 vs h_1)', fontsize=10, style='italic')
    ax_phase.grid(True, linestyle='--', alpha=0.6)

    dashboard_state = {
        'ax_ts': ax_ts,
        'ax_phase': ax_phase,
        'data': data,
        'fig': fig
    }

    def onclick(event):
        if event.inaxes not in [ax_map_det, ax_map_date]:
            return
        x, y = int(event.xdata), int(event.ydata)
        height, width = data['change_detected_map'].shape
        if not (0 <= y < height and 0 <= x < width):
            return
        update_plots(x, y, dashboard_state)

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()

def update_plots(x, y, state):
    """
    Updates both the temporal and phase space plots.
    Strictly applies boolean masking to prevent visualizing imputed sensor data.
    """
    ax_ts = state['ax_ts']
    ax_phase = state['ax_phase']
    data = state['data']
    
    ax_ts.clear()
    ax_phase.clear()
    ax_ts.grid(True, linestyle='--', alpha=0.6)
    ax_phase.grid(True, linestyle='--', alpha=0.6)
    
    is_anomaly = data['change_detected_map'][y, x]
    anomaly_date = data['change_date_map'][y, x]
    
    title_color = 'red' if is_anomaly else 'black'
    status_text = f"ANOMALOUS (Date: {anomaly_date:.2f})" if is_anomaly else "NORMAL"
    
    # Check if pixel was actually processed (avoid all-zero uncomputed borders)
    if np.all(data['ode_predictions'][:, 0, y, x] == 0):
        ax_ts.text(0.5, 0.5, "Pixel excluded during training (insufficient clear obs)", 
                   ha='center', va='center', transform=ax_ts.transAxes, color='red')
        state['fig'].canvas.draw_idle()
        return

    # --- 1. Temporal Dynamics Update ---
    ax_ts.set_title(f'Temporal Domain: Pixel (X:{x}, Y:{y}) - {status_text}', color=title_color, fontsize=11, fontweight='bold')
    ax_ts.set_xlabel('Time (Fractional Year)')
    ax_ts.set_ylabel('Standardized Target Metric')

    pixel_features = data['features'][:, 0, y, x] # Target metric is dim 0
    pixel_masks = data['masks'][:, y, x]
    times = data['time_steps']
    
    valid_idx = pixel_masks.astype(bool)
    valid_times = times[valid_idx]
    valid_obs = pixel_features[valid_idx]
    
    # Plot true, physically observed data points only
    ax_ts.scatter(valid_times, valid_obs, color='black', label='Valid Sensor Obs.', zorder=3)
    
    # --- NEW VISUALIZATION: MASKED OBSERVATIONS ---
    # Expose the QA-masked data points (e.g., clouds/shadows) as open circles.
    # This provides visual validation that the mask correctly intercepted severe optical noise.
    invalid_idx = ~valid_idx
    invalid_times = times[invalid_idx]
    invalid_obs = pixel_features[invalid_idx]
    
    # We strictly filter out infinite values before plotting.
    # If an 'inf' is plotted, Matplotlib will expand the y-axis bounds to infinity, flattening the valid trajectory.
    finite_mask = np.isfinite(invalid_obs)
    if np.any(finite_mask):
        ax_ts.scatter(invalid_times[finite_mask], invalid_obs[finite_mask], 
                      facecolors='none', edgecolors='gray', alpha=0.6, 
                      label='Masked (Cloud/Shadow)', zorder=1)
    
    # Plot continuous ODE representation
    pixel_preds = data['ode_predictions'][:, 0, y, x]
    ax_ts.plot(times, pixel_preds, color='#1f77b4', linewidth=2, label='ODE Continuous Trajectory', zorder=2)
    
    if is_anomaly:
        ax_ts.axvline(x=anomaly_date, color='red', linestyle='--', linewidth=1.5, label='Anomaly Onset')
    ax_ts.legend(loc='best')
    ax_ts.set_ylim(-4, 4)

    # --- 2. Phase Space Topology Update ---
    ax_phase.set_title('Latent Phase Space (Vector Field Attractor)', color=title_color, fontsize=11, fontweight='bold')
    ax_phase.set_xlabel('Latent Dimension 0 (h_0)')
    ax_phase.set_ylabel('Latent Dimension 1 (h_1)')

    # Extract first two latent dimensions
    h0 = data['hidden_states'][:, 0, y, x]
    h1 = data['hidden_states'][:, 1, y, x]

    # Create a continuous colored line to show temporal evolution in phase space
    points = np.array([h0, h1]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Color mapping based on time
    norm = plt.Normalize(times.min(), times.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm, alpha=0.8, linewidth=1.5)
    lc.set_array(times)
    ax_phase.add_collection(lc)
    
    # Auto-scale phase space
    ax_phase.set_xlim(h0.min() - 0.1, h0.max() + 0.1)
    ax_phase.set_ylim(h1.min() - 0.1, h1.max() + 0.1)

    # Highlight the divergence point if an anomaly occurred
    if is_anomaly:
        # Find index closest to anomaly date
        idx = np.argmin(np.abs(times - anomaly_date))
        ax_phase.scatter(h0[idx], h1[idx], color='red', s=100, marker='X', zorder=5, label='Trajectory Divergence Break')
        ax_phase.legend(loc='best')

    # Add colorbar for time dimension mapping in phase space
    if not hasattr(state, 'cbar_phase'):
        state['cbar_phase'] = state['fig'].colorbar(lc, ax=ax_phase, fraction=0.046, pad=0.04)
        state['cbar_phase'].set_label('Time (Fractional Year)')
    else:
        state['cbar_phase'].update_normal(lc)

    state['fig'].canvas.draw_idle()

if __name__ == "__main__":
    print(f"Loading ODE parameters from:\n  {ODE_H5}")
    print(f"Loading geographic source context from:\n  {SOURCE_H5}")
    
    plot_data = load_h5_data(ODE_H5, SOURCE_H5)
    
    print("Initializing Phase Space dashboard...")
    print("--> INSTRUCTIONS: Click on any spatial pixel to view its underlying vector field dynamics.")
    setup_dashboard(plot_data)