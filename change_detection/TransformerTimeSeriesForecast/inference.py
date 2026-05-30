import os
import warnings

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore", message=".*nested tensors.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

import torch
import torch.nn as nn
import h5py
import numpy as np
from datetime import datetime, timezone
import math
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import rasterio.transform
from pyproj import Transformer, CRS

# ==========================================
# 1. CONFIGURATION & ABLATION SETTINGS
# ==========================================
H5_PATH = "C:/satelliteImagery/LANDSAT/Rochester/LANDSAT_Stack_Rochester_GEE_2015_2025_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

TRAINED_METRICS = ['sliding_volume_z_score']#, 'evi_map']  
POSITIONAL_ENCODING = ['DOY_ratio','xPos_ratio','yPos_ratio'] #,'DOY_sin','DOY_cos'
METRIC_SCALARS = {
    'evi_map': 1.0,        
    'sliding_volume_map': 1.0,   
    'msd_map': 1.0,  
    'sliding_volume_z_score': 1/3, 
    'sliding_volume_local_z_score': 1.0, 
}

METRIC_TRANSFORMS = {
    'evi_map': ('z_score_dynamic', None, None),   
    'sliding_volume_map': ('log10_shift', 0, 1/math.log10(3.5e-4)), 
    'msd_map': ('linear', None, None), 
    'sliding_volume_z_score': ('linear', None, None), 
    'sliding_volume_local_z_score': ('linear', None, None), 
}

run_name = "-".join(POSITIONAL_ENCODING)
run_name += "-".join(TRAINED_METRICS)
MODEL_WEIGHTS = f"C:/satelliteImagery/LANDSAT/Rochester/temporal_autoencoder_{run_name}.pth"

MAX_SEQ_LEN = 300 
D_MODEL = 64
N_HEADS = 4
NUM_LAYERS = 4
DIM_FEEDFORWARD = 256

OUTPUT_PLOT = f"C:/satelliteImagery/LANDSAT/Rochester/Trajectory_Analysis_{run_name}_2015_2025.png"
INFERENCE_START_YEAR = 2024 

TS_LOCATIONS = [
    {'latlon': (43.142856, -77.508451), 'label': "West Tait Forest",     'color': 'tab:green'},
    {'latlon': (43.144861, -77.501176), 'label': "East Tait Forest",             'color': 'tab:olive'},
    {'latlon': (43.136910, -77.469462), 'label': "Artificial turf football field",  'color': 'tab:blue'},
    {'latlon': (43.138241, -77.470873), 'label': "Recently added artificial turf",  'color': 'tab:cyan'},
    {'latlon': (43.141297, -77.506256), 'label': "Tait Parking Lot",                'color': 'tab:red'},
    {'latlon': (43.139411, -77.504005), 'label': "ROCX NITE Tarp",                  'color': 'tab:purple'},
]

SUN_ELEVATION_THRESHOLD = 0
CLOUD_DILATION = 0
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_FILTER = 'medium'
AEROSOL_LEVELS = {
    'low': [2, 4, 32, 66, 68, 96, 100],
    'medium': [130, 132, 160, 164],
    'high': [192, 194, 196, 224, 228]
}

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class TemporalAttentionAutoencoder(nn.Module):
    def __init__(self, input_channels, output_channels, d_model=128, n_heads=8, num_layers=6, dim_feedforward=512):
        super().__init__()
        
        # True MAE Architecture (Must match training script exactly)
        self.num_metrics = output_channels
        self.num_context = input_channels - output_channels 
        
        self.phys_proj = nn.Linear(self.num_metrics, d_model)
        self.context_proj = nn.Linear(self.num_context, d_model)
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, 
            batch_first=True, dropout=0.0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, output_channels)
        
    def forward(self, x, padding_mask, rand_mask=None):
        phys_x = x[:, :, :self.num_metrics]
# Keep AEROSOL_LEVELS in config
AEROSOL_LEVELS = {
    'low': [2, 4, 32, 66, 68, 96, 100],
    'medium': [130, 132, 160, 164],
    'high': [192, 194, 196, 224, 228]
}

# ==========================================
# 3. SPATIAL MASKING UTILITY
# ==========================================
def get_advanced_mask(data_grp, valid_indices, height, width):
    num_frames = len(valid_indices)
    valid_mask = np.ones((num_frames, height, width), dtype=bool)
    sun_elev_arr = data_grp['surface_reflectance'].attrs.get('sun_elevation')
    kernel = np.ones((3, 3), dtype=bool)
    
    if 'QUALITY_L2_AEROSOL' in data_grp:
        raw_aerosol = data_grp['QUALITY_L2_AEROSOL'][...]

    for new_idx, original_idx in enumerate(valid_indices):
        if sun_elev_arr is not None and original_idx < len(sun_elev_arr):
            if sun_elev_arr[original_idx] < SUN_ELEVATION_THRESHOLD:
                valid_mask[new_idx] = False; continue

        if 'QUALITY_L1_PIXEL' in data_grp:
            qa_pixel = data_grp['QUALITY_L1_PIXEL'][original_idx, ...]
            bad_qa_mask = (qa_pixel & QA_REJECT_MASK) != 0
            if CLOUD_DILATION > 0: bad_qa_mask = ndimage.binary_dilation(bad_qa_mask, structure=kernel, iterations=CLOUD_DILATION)
            valid_mask[new_idx] &= ~bad_qa_mask

        if 'RADIOMETRIC_SATURATION' in data_grp:
            bad_radsat = data_grp['RADIOMETRIC_SATURATION'][original_idx, ...] != RADSAT_ACCEPT_VALUE
            if CLOUD_DILATION > 0: bad_radsat = ndimage.binary_dilation(bad_radsat, structure=kernel, iterations=CLOUD_DILATION)
            valid_mask[new_idx] &= ~bad_radsat

        if 'QUALITY_L2_AEROSOL' in data_grp:
            frame_aerosol = raw_aerosol[original_idx, ...]
            good_aerosol_mask = np.isin(frame_aerosol, AEROSOL_LEVELS[AEROSOL_FILTER])
            valid_mask[new_idx] &= good_aerosol_mask

    return valid_mask

# ==========================================
# 4. INFERENCE ENGINE 
# ==========================================
def main():
    # Safety mapping: Links the new config variable name to the legacy script references
    ACTIVE_METRICS = TRAINED_METRICS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Temporal Autoencoder on {device}...")
    
    num_metrics = len(ACTIVE_METRICS)
    num_context = len(POSITIONAL_ENCODING)
    in_channels = num_metrics + num_context
    out_channels = num_metrics
    
    model = TemporalAttentionAutoencoder(
        input_channels=in_channels, output_channels=out_channels,
        d_model=D_MODEL, n_heads=N_HEADS, num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        model.eval() 
        print(f"Model weights loaded successfully: {MODEL_WEIGHTS}")
    except Exception as e:
        print(f"Failed to load weights. Error: {e}")
        return

    print("\nScanning HDF5 for complete Time Series Data...")
    with h5py.File(H5_PATH, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/LANDSAT/Data Fields']
        sr_ds = data_grp['surface_reflectance']
        
        sr_attrs = sr_ds.attrs
        geo_transform = sr_attrs.get('GeoTransform')
        spatial_ref = sr_attrs.get('spatial_ref')
        
        if geo_transform is not None and spatial_ref is not None:
            if isinstance(spatial_ref, bytes): spatial_ref = spatial_ref.decode('utf-8')
            crs = CRS.from_wkt(spatial_ref)
            transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            affine = rasterio.transform.Affine(*geo_transform)
            inv_affine = ~affine
            
            for loc in TS_LOCATIONS:
                lat, lon = loc['latlon']
                proj_x, proj_y = transformer.transform(lon, lat)
                px, py = inv_affine * (proj_x, proj_y)
                loc['yx'] = (int(round(py)), int(round(px)))
        else:
            raise ValueError("Missing spatial metadata in HDF5.")

        acq_times = sr_ds.attrs.get('acquisition_time')
        all_indices = list(range(len(acq_times)))
        
        julian_days = []
        datetime_objs = []
        
        for dt_ts in acq_times:
            dt_obj = datetime.fromtimestamp(float(dt_ts), tz=timezone.utc)
            datetime_objs.append(dt_obj)
            julian_days.append(dt_obj.timetuple().tm_yday)
            
        julian_days = np.array(julian_days)
        
        height, width = sr_ds.shape[2], sr_ds.shape[3]
        
        raw_data = {}
        if 'evi_map' in ACTIVE_METRICS:
            raw_data['evi_map'] = data_grp['evi_map'][...]
        if 'sliding_volume_map' in ACTIVE_METRICS:
            raw_data['sliding_volume_map'] = data_grp['sliding_volume_map'][...]
        if 'msd_map' in ACTIVE_METRICS:
            raw_data['msd_map'] = data_grp['msd_map'][...]
        if 'sliding_volume_z_score' in ACTIVE_METRICS:
            raw_data['sliding_volume_z_score'] = data_grp['sliding_volume_z_score'][...]
        if 'sliding_volume_local_z_score' in ACTIVE_METRICS:
            raw_data['sliding_volume_local_z_score'] = data_grp['sliding_volume_local_z_score'][...]
            
        print("Generating spatial validity masks for the full 10-year stack...")
        valid_mask = get_advanced_mask(data_grp, all_indices, height, width)
        
        print("Evaluating targeted pixels...")
        results = {}
        
        with torch.no_grad():
            for loc in TS_LOCATIONS:
                y, x = loc['yx']
                label = loc['label']
                
                pixel_valid = valid_mask[:, y, x]
                if np.sum(pixel_valid) == 0:
                    continue
                    
                x_ratio = 2.0 * (x / max(1, width - 1)) - 1.0
                y_ratio = 2.0 * (y / max(1, height - 1)) - 1.0
                    
                pixel_days = julian_days[pixel_valid]
                pixel_dates = [datetime_objs[i] for i, valid in enumerate(pixel_valid) if valid]
                
                # Transform data into the latent representation space the model expects
                latent_inputs = {}
                for metric in ACTIVE_METRICS:
                    raw_vals = raw_data[metric][pixel_valid, y, x] * METRIC_SCALARS[metric]
                    trans_type, offset, scale = METRIC_TRANSFORMS[metric]
                    
                    if trans_type == 'log10_shift':
                        latent_inputs[metric] = (np.log10(np.clip(raw_vals, 1e-12, None)) + offset) / scale
                    elif trans_type == 'log1p':
                        latent_inputs[metric] = np.log1p(np.clip(raw_vals, 0, None))
                    else:
                        latent_inputs[metric] = raw_vals
                    
                total_obs = len(pixel_days)
                pred_latent = {m: [] for m in ACTIVE_METRICS}
                
                for i in range(0, total_obs, MAX_SEQ_LEN):
                    chunk_days = pixel_days[i:i+MAX_SEQ_LEN]
                    seq_len = len(chunk_days)
                    
                    features = np.zeros((1, MAX_SEQ_LEN, in_channels), dtype=np.float32)
                    DOY_ratio = (chunk_days / 365.25)
                    
                    for m_idx, metric in enumerate(ACTIVE_METRICS):
                        features[0, :seq_len, m_idx] = latent_inputs[metric][i:i+MAX_SEQ_LEN]
                    
                    # Dynamically inject the requested contextual features
                    for p_idx, enc_type in enumerate(POSITIONAL_ENCODING):
                        feature_idx = num_metrics + p_idx
                        
                        if enc_type == 'DOY_ratio':
                            features[0, :seq_len, feature_idx] = 2.0 * DOY_ratio - 1.0 
                        elif enc_type == 'DOY_sin':
                            features[0, :seq_len, feature_idx] = np.sin(2.0 * math.pi * DOY_ratio)
                        elif enc_type == 'DOY_cos':
                            features[0, :seq_len, feature_idx] = np.cos(2.0 * math.pi * DOY_ratio)
                        elif enc_type == 'xPos_ratio':
                            features[0, :seq_len, feature_idx] = x_ratio
                        elif enc_type == 'yPos_ratio':
                            features[0, :seq_len, feature_idx] = y_ratio
                        else:
                            raise ValueError(f"Unknown POSITIONAL_ENCODING configuration: {enc_type}")
                    
                    padding_mask = np.ones((1, MAX_SEQ_LEN), dtype=bool)
                    padding_mask[0, :seq_len] = False
                    
                    b_feat = torch.from_numpy(features).to(device)
                    b_pad = torch.from_numpy(padding_mask).to(device)
                    
                    predictions = model(b_feat, b_pad).cpu().numpy()[0] 
                    
                    for m_idx, metric in enumerate(ACTIVE_METRICS):
                        pred_latent[metric].extend(predictions[:seq_len, m_idx])
                
                # --- MAINTAIN LATENT / PROCESSED SPACE ---
                processed_actuals = {}
                processed_preds = {}
                
                for metric in ACTIVE_METRICS:
                    act_lat = latent_inputs[metric]
                    pred_lat = np.array(pred_latent[metric])
                    trans_type, offset, scale = METRIC_TRANSFORMS[metric]
                    
                    if trans_type == 'log10_shift':
                        # Parameterized mathematical inverse: 10^(x * scale - offset)
                        processed_actuals[metric] = 10.0 ** (act_lat * scale - offset)
                        processed_preds[metric] = 10.0 ** (pred_lat * scale - offset)
                    elif trans_type == 'log1p':
                        processed_actuals[metric] = np.expm1(act_lat) 
                        processed_preds[metric] = np.expm1(np.clip(pred_lat, 0, None)) 
                    else:
                        processed_actuals[metric] = act_lat 
                        processed_preds[metric] = pred_lat 
                        
                results[label] = {
                    'dates': np.array(pixel_dates),
                    'actuals': processed_actuals,
                    'preds': processed_preds,
                    'color': loc['color']
                }

    # ==========================================
    # 5. VISUALIZATION & METRICS
    # ==========================================
    print("\nGenerating Time Series Trajectory Plots...")
    num_locs = len(results)
    num_cols = len(ACTIVE_METRICS)
    
    if num_locs == 0:
        print("No valid data found for locations. Exiting.")
        return
        
    fig, axes = plt.subplots(num_locs, num_cols, figsize=(8 * num_cols, 3 * num_locs), sharex=True, squeeze=False)
    fig.canvas.manager.set_window_title(f"Reconstructed Trajectories [{run_name.upper()}]")
    plt.subplots_adjust(hspace=0.3, wspace=0.25 if num_cols > 1 else 0.1)
    
    split_date = datetime(INFERENCE_START_YEAR, 1, 1, tzinfo=timezone.utc)
    title_map = {'evi_map': 'EVI', 'sliding_volume_map': 'Spectral Complexity', 'msd_map': 'MSD', 'sliding_volume_z_score': 'SC Z-Score', 'sliding_volume_local_z_score': 'SC Local Z-Score'}
    
    for row_idx, (label, data) in enumerate(results.items()):
        sort_idx = np.argsort(data['dates'])
        dates = data['dates'][sort_idx]
        train_mask = np.array([d.year < INFERENCE_START_YEAR for d in dates])
        
        for col_idx, metric in enumerate(ACTIVE_METRICS):
            ax = axes[row_idx, col_idx]
            
            act_vals = data['actuals'][metric][sort_idx]
            pred_vals = data['preds'][metric][sort_idx]
            trans_type, _, _ = METRIC_TRANSFORMS[metric]
            
            act_train = act_vals[train_mask]
            pred_train = pred_vals[train_mask]
            
            if len(act_train) > 0:
                rmse = np.sqrt(np.mean((act_train - pred_train)**2))
                ss_res = np.sum((act_train - pred_train) ** 2)
                ss_tot = np.sum((act_train - np.mean(act_train)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                fmt = ".2e" if trans_type in ['log1p', 'log10_shift'] else ".3f"
                fit_text = f"Baseline Fit (≤{INFERENCE_START_YEAR-1}):\nRMSE: {rmse:{fmt}} | $R^2$: {r2:.3f}"
            else:
                fit_text = "No Training Data in Range"
            
            ax.plot(dates, act_vals, marker='o', markersize=4, linestyle='-', linewidth=1.5, 
                    color=data['color'], label=f'Actual {metric.upper()}')
            ax.plot(dates, pred_vals, marker='', linestyle='--', linewidth=2, 
                    color='black', alpha=0.7, label='Model Prediction')
            
            ax.axvline(split_date, color='red', linestyle=':', linewidth=1.5, alpha=0.8, 
                       label='Training End' if (row_idx == 0 and col_idx == 0) else "")
            
            ax.text(0.02, 0.05, fit_text, transform=ax.transAxes, fontsize=9, family='monospace',
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            ax.set_title(f"{label} ({title_map.get(metric, metric.upper())})", fontsize=11, fontweight='bold')
            
            y_label = metric.upper()
            if trans_type == 'log10_shift':
                y_label = "Volume (Log10 scale)"
                ax.set_yscale('log')
            elif trans_type == 'log1p':
                y_label = f"log1p({metric.upper()} * {METRIC_SCALARS[metric]:.0e})"
            elif METRIC_SCALARS[metric] != 1.0:
                y_label = f"{metric.upper()} (* {METRIC_SCALARS[metric]:.0e})"
            ax.set_ylabel(y_label)
            
            ax.grid(True, alpha=0.3)
            if row_idx == 0: ax.legend(loc='upper right', fontsize=9)

    for col_idx in range(num_cols):
        ax_bottom = axes[-1, col_idx]
        ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax_bottom.xaxis.set_major_locator(mdates.YearLocator())
        for tick in ax_bottom.get_xticklabels():
            tick.set_rotation(45)
            
    fig.suptitle(f"Temporal Autoencoder Ablation: [{run_name.upper()}]\n"
                 "(Evaluated in Aligned Space, Rendered in Native Scale)", 
                 fontsize=14, y=0.95)
                 
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"Saved Trajectory Plots to: {OUTPUT_PLOT}")
    plt.show()

if __name__ == "__main__":
    main()