import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timezone
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import warnings

# Import data extraction from the user's framework
from data_loader import load_merged_datacube, landsat_path, tanager_path

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = r"C:\satelliteImagery\transformerModel\spatiotemporal_transformer_mask_aware.pth"
ANOMALY_THRESHOLD = 2.5
CONSECUTIVE_REQUIRED = 3

# ==========================================
# DUPLICATED ARCHITECTURE FOR STANDALONE LOADING
# ==========================================
class SpatialEmbedding(nn.Module):
    def __init__(self, h, w, d_model):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        conv_out_h = math.ceil(h / 4)
        conv_out_w = math.ceil(w / 4)
        self.linear = nn.Linear(32 * conv_out_h * conv_out_w, d_model)

    def forward(self, x):
        b, t, h, w = x.shape
        x = x.view(b * t, 1, h, w)
        x = self.conv(x)
        x = self.linear(x)
        x = x.view(b, t, -1)
        return x

class Time2Vec(nn.Module):
    """
    Continuous temporal embedding (Kazemi et al., 2019).
    Maps scalar fractional years into a d_model dimensional vector.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Linear trend component (catches long-term drift)
        self.w_linear = nn.Parameter(torch.randn(1))
        self.b_linear = nn.Parameter(torch.randn(1))
        
        # Periodic components (catches seasonality)
        freqs = torch.randn(d_model - 1)
        
        # Inductive Physical Bias: Target core ecological phenology harmonics
        if d_model > 4:
            freqs[0] = 2.0 * math.pi * 1.0  # 1-Year Cycle
            freqs[1] = 2.0 * math.pi * 2.0  # 6-Month Cycle
            freqs[2] = 2.0 * math.pi * 3.0  # 4-Month Cycle
            
        self.w_periodic = nn.Parameter(freqs)
        self.b_periodic = nn.Parameter(torch.randn(d_model - 1))

    def forward(self, t):
        # t shape: [Batch, Sequence_Length]
        t = t.unsqueeze(-1) # [B, S, 1]
        
        # Time2Vec mapping
        linear = self.w_linear * t + self.b_linear # [B, S, 1]
        periodic = torch.sin(self.w_periodic * t + self.b_periodic) # [B, S, d_model - 1]
        
        # Concatenate to form the full d_model vector
        return torch.cat([linear, periodic], dim=-1) # [B, S, d_model]

class SpatialReconstruction(nn.Module):
    def __init__(self, d_model, h, w):
        super().__init__()
        self.h = h
        self.w = w
        self.linear = nn.Linear(d_model, h * w)

    def forward(self, x):
        b, t, _ = x.shape
        x = self.linear(x)
        x = x.view(b, t, self.h, self.w)
        return x

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, h, w, d_model=128, nhead=8, num_encoder_layers=4, num_decoder_layers=4):
        super().__init__()
        self.embedding = SpatialEmbedding(h, w, d_model)
        
        # Replace discrete positional encoding with continuous Time2Vec
        self.time_encoder = Time2Vec(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers, batch_first=True
        )
        self.reconstruction = SpatialReconstruction(d_model, h, w)

    def forward(self, src_safe, tgt_safe, src_time, tgt_time):
        src_emb = self.embedding(src_safe) + self.time_encoder(src_time)
        tgt_emb = self.embedding(tgt_safe) + self.time_encoder(tgt_time)
        output = self.transformer(src_emb, tgt_emb)
        return self.reconstruction(output)

def get_fractional_year(dt_utc):
    """Converts a UTC datetime object into a continuous fractional year."""
    year = dt_utc.year
    start_of_year = datetime(year, 1, 1, tzinfo=timezone.utc)
    start_of_next = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    
    year_duration = (start_of_next - start_of_year).total_seconds()
    elapsed = (dt_utc - start_of_year).total_seconds()
    
    return float(year) + (elapsed / year_duration)

class SpatioTemporalDataset(Dataset):
    def __init__(self, data_list, seq_length):
        self.seq_length = seq_length
        arrays = [frame['sliding_volume_z_score_masked'] for frame in data_list]
        self.data_cube = np.stack(arrays, axis=0)
        self.valid_mask = ~np.isnan(self.data_cube)
        safe_cube = np.nan_to_num(self.data_cube, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.tensor_cube = torch.tensor(safe_cube, dtype=torch.float32)
        self.mask_cube = torch.tensor(self.valid_mask, dtype=torch.bool)
        
        # Extract continuous chronological time
        frac_years = [get_fractional_year(frame['datetime_utc']) for frame in data_list]
        self.tensor_times = torch.tensor(frac_years, dtype=torch.float32)

    def __len__(self):
        return len(self.tensor_cube) - self.seq_length

    def __getitem__(self, idx):
        src_safe = self.tensor_cube[idx : idx + self.seq_length]
        tgt_safe = self.tensor_cube[idx + 1 : idx + self.seq_length + 1]
        tgt_mask = self.mask_cube[idx + 1 : idx + self.seq_length + 1]
        
        src_time = self.tensor_times[idx : idx + self.seq_length]
        tgt_time = self.tensor_times[idx + 1 : idx + self.seq_length + 1]
        
        return src_safe, tgt_safe, tgt_mask, src_time, tgt_time

def extract_background_rgb(eval_frames):
    print("Generating temporal median RGB composite for background context...")
    rgb_stack = []
    for frame in eval_frames:
        rgba = frame['ortho_visual']
        if rgba.shape[-1] >= 3:
            rgb_stack.append(rgba[:, :, :3])
            
    rgb_cube = np.stack(rgb_stack, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bg_map = np.nanmedian(rgb_cube, axis=0)
        
    p2, p98 = np.nanpercentile(bg_map, (2, 98), axis=(0, 1))
    for i in range(3):
        if p98[i] > p2[i]:
            bg_map[:, :, i] = np.clip((bg_map[:, :, i] - p2[i]) / (p98[i] - p2[i]), 0, 1)
        else:
            bg_map[:, :, i] = 0
            
    return bg_map

# ==========================================
# INTERACTIVE VIEWER CLASS
# ==========================================
class TransformerInteractiveViewer:
    def __init__(self):
        print("Ingesting merged multi-sensor datacube...")
        cube, h5_l, h5_t = load_merged_datacube(landsat_path, tanager_path)
        
        print("Applying spatial QA masks to spectral volumes...")
        for frame in cube:
            frame['raw_z_score'] = frame['sliding_volume_z_score_masked'].copy()
            frame['sliding_volume_z_score_masked'][~frame['qa_mask']] = np.nan
        
        split_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        eval_frames = [f for f in cube if f['datetime_utc'] >= split_date]
        
        if len(eval_frames) == 0:
            raise RuntimeError("No evaluation frames found (>= 2024).")

        self.SEQUENCE_LENGTH = 10 
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model checkpoint missing at {MODEL_PATH}")

        checkpoint = torch.load(MODEL_PATH)
        self.H, self.W = checkpoint['spatial_dims']
        
        print(f"Loading trained Transformer weights for spatial grid [{self.H}x{self.W}]...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SpatioTemporalTransformer(h=self.H, w=self.W, d_model=128, nhead=8).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.eval_dataset = SpatioTemporalDataset(eval_frames, seq_length=self.SEQUENCE_LENGTH)
        eval_loader = DataLoader(self.eval_dataset, batch_size=1, shuffle=False)

        # Build comprehensive data cache for fast plotting
        self.N_frames = len(eval_frames)
        self.timestamps = np.array([f['datetime_utc'] for f in eval_frames])
        self.raw_data = np.stack([f['raw_z_score'] for f in eval_frames], axis=0)
        self.qa_masks = np.stack([f['qa_mask'] for f in eval_frames], axis=0)
        
        self.pred_cube = np.full((self.N_frames, self.H, self.W), np.nan)
        self.anomaly_breaches = np.zeros((self.N_frames, self.H, self.W), dtype=bool)

        consecutive_anomaly_counts = np.zeros((self.H, self.W), dtype=int)
        self.confirmed_anomaly_map = np.zeros((self.H, self.W), dtype=bool)

        print("Evaluating autoregressive trajectory and applying temporal consistency filter...")
        with torch.no_grad():
            for batch_idx, (src_safe, tgt_safe, tgt_mask, src_time, tgt_time) in enumerate(eval_loader):
                src_safe = src_safe.to(self.device)
                tgt_safe = tgt_safe.to(self.device)
                tgt_mask = tgt_mask.to(self.device)
                src_time = src_time.to(self.device)
                tgt_time = tgt_time.to(self.device)
                
                predicted_tgt = self.model(src_safe, tgt_safe, src_time, tgt_time)
                
                pred_frame = predicted_tgt[0, -1, :, :].cpu().numpy()
                true_frame = tgt_safe[0, -1, :, :].cpu().numpy()
                mask_frame = tgt_mask[0, -1, :, :].cpu().numpy()
                
                # Align prediction with specific chronological target frame
                target_idx = batch_idx + self.SEQUENCE_LENGTH
                self.pred_cube[target_idx] = pred_frame
                
                residual_map = np.zeros((self.H, self.W), dtype=np.float32)
                residual_map[mask_frame] = np.abs(pred_frame[mask_frame] - true_frame[mask_frame])
                
                is_anomalous_frame = residual_map > ANOMALY_THRESHOLD
                self.anomaly_breaches[target_idx] = is_anomalous_frame & mask_frame
                
                # CCDC-style consistency filter
                valid_and_anomalous = mask_frame & is_anomalous_frame
                consecutive_anomaly_counts[valid_and_anomalous] += 1
                
                valid_and_normal = mask_frame & ~is_anomalous_frame
                consecutive_anomaly_counts[valid_and_normal] = 0
                
                newly_confirmed = consecutive_anomaly_counts >= CONSECUTIVE_REQUIRED
                self.confirmed_anomaly_map |= newly_confirmed

        self.bg_map = extract_background_rgb(eval_frames)
        h5_l.close()
        h5_t.close()

        self._init_ui()

    def _init_ui(self):
        print("\nRendering Interactive Spatial Anomaly Interface...")
        self.fig, (self.ax_map, self.ax_plot) = plt.subplots(1, 2, figsize=(18, 8))
        self.fig.canvas.manager.set_window_title("Transformer Predictive Anomaly Viewer")

        # Setup Map Panel
        self.ax_map.imshow(self.bg_map, alpha=0.8, interpolation='none')
        masked_anomaly_map = np.ma.masked_where(~self.confirmed_anomaly_map, self.confirmed_anomaly_map)
        cmap_binary = ListedColormap(['#e74c3c']) 
        
        self.ax_map.imshow(masked_anomaly_map, cmap=cmap_binary, interpolation='none')
        self.ax_map.set_title(f'Confirmed Anomalies (\u2265{CONSECUTIVE_REQUIRED} Obs > {ANOMALY_THRESHOLD} Z)', 
                              fontsize=14, fontweight='bold')
        self.ax_map.axis('off')
        
        anom_patch = mpatches.Patch(color='#e74c3c', label='Structural Break')
        self.ax_map.legend(handles=[anom_patch], loc='upper right', fontsize=11)

        # Highlight marker for clicks
        self.marker_circle, = self.ax_map.plot([], [], marker='o', color='orange', markersize=12, fillstyle='none', mew=2)
        
        # Setup Plot Panel
        self.ax_plot.set_title("Temporal Dynamics (Click a pixel to analyze)", fontsize=11, style='italic')
        self.ax_plot.grid(True, linestyle='--', alpha=0.6)

        # Connect event
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        
        # Load center pixel as default
        self._plot_pixel(self.W // 2, self.H // 2)
        
        plt.tight_layout()

    def onclick(self, event):
        if event.inaxes == self.ax_map:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= x < self.W and 0 <= y < self.H:
                self._plot_pixel(x, y)

    def _plot_pixel(self, x, y):
        self.ax_plot.clear()
        self.marker_circle.set_data([x], [y])
        
        valid = self.qa_masks[:, y, x]
        pixel_raw = self.raw_data[:, y, x]
        pixel_preds = self.pred_cube[:, y, x]
        pixel_breaches = self.anomaly_breaches[:, y, x]

        # Ensure we only plot finite mathematical values (reject absolute NaNs from edges)
        finite_mask = np.isfinite(pixel_raw)
        valid_idx = valid & finite_mask
        invalid_idx = ~valid & finite_mask

        # Plot 1: Valid Observations
        if np.any(valid_idx):
            self.ax_plot.scatter(self.timestamps[valid_idx], pixel_raw[valid_idx], 
                                 c='black', label='Valid Obs.', zorder=4)

        # Plot 2: Rejected / Masked Observations
        if np.any(invalid_idx):
            self.ax_plot.scatter(self.timestamps[invalid_idx], pixel_raw[invalid_idx], 
                                 facecolors='none', edgecolors='gray', alpha=0.6, 
                                 label='Masked (Cloud/Shadow)', zorder=3)

        # Plot 3: Autoregressive Predictions & Error Bounds
        valid_pred_mask = ~np.isnan(pixel_preds)
        if np.any(valid_pred_mask):
            pred_times = self.timestamps[valid_pred_mask]
            pred_vals = pixel_preds[valid_pred_mask]
            
            self.ax_plot.plot(pred_times, pred_vals, c='#1f77b4', linewidth=2, label='Transformer Pred.', zorder=2)
            
            # Error Bounds defined by threshold
            self.ax_plot.fill_between(pred_times, 
                                      pred_vals - ANOMALY_THRESHOLD, 
                                      pred_vals + ANOMALY_THRESHOLD, 
                                      color='#1f77b4', alpha=0.15, 
                                      label=f'Normal Bounds (\u00B1{ANOMALY_THRESHOLD} Z)', zorder=1)

        # Plot 4: Flagged Breaches (When Valid Obs exceed Error Bounds)
        if np.any(pixel_breaches):
            self.ax_plot.scatter(self.timestamps[pixel_breaches], pixel_raw[pixel_breaches], 
                                 c='red', marker='X', s=80, label='Anomaly Breach', zorder=5)

        # Style & Formatting
        is_confirmed = self.confirmed_anomaly_map[y, x]
        title_color = 'red' if is_confirmed else 'black'
        status_text = "STRUCTURAL BREAK CONFIRMED" if is_confirmed else "STABLE DYNAMICS"
        
        self.ax_plot.set_title(f"Evaluation Trajectory | Pixel ({x}, {y}) | {status_text}", 
                               color=title_color, fontsize=12, fontweight='bold')
        self.ax_plot.set_ylabel("Spectral Complexity (Z-Score)")
        self.ax_plot.grid(True, linestyle='--', alpha=0.6)
        
        # Let Matplotlib handle datetime formatting nicely
        self.fig.autofmt_xdate()
        self.ax_plot.legend(loc='best')
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    viewer = TransformerInteractiveViewer()
    plt.show()