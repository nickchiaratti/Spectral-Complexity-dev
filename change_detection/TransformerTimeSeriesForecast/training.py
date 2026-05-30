import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from datetime import datetime, timezone
import math
from tqdm import tqdm
from scipy import ndimage

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

TRAIN_END_YEAR = 2023   
MASK_PROB = 0.20
BATCH_SIZE = 1024       
EPOCHS = 30
LEARNING_RATE = 1e-4

SUN_ELEVATION_THRESHOLD = 30
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
# 2. DATASET DEFINITION
# ==========================================
class PixelTimeSeriesDataset(Dataset):
    def __init__(self, h5_path, active_metrics, metric_scalars, metric_transforms, max_seq_len=300, pos_encodings=None):
        self.max_seq_len = max_seq_len
        self.active_metrics = active_metrics
        self.metric_scalars = metric_scalars
        self.metric_transforms = metric_transforms
        self.num_metrics = len(self.active_metrics)
        self.pos_encodings = pos_encodings if pos_encodings is not None else []
        self.num_context = len(self.pos_encodings)
        
            
        print(f"Loading HDF5 data into memory (Training data <= {TRAIN_END_YEAR})...")
        with h5py.File(h5_path, 'r') as f:
            data_grp = f['/HDFEOS/GRIDS/LANDSAT/Data Fields']
            sr_ds = data_grp['surface_reflectance']
            
            acq_times = sr_ds.attrs.get('acquisition_time')
            valid_train_indices = []
            self.julian_days = []
            
            for i, dt_ts in enumerate(acq_times):
                dt_obj = datetime.fromtimestamp(float(dt_ts), tz=timezone.utc)
                if dt_obj.year <= TRAIN_END_YEAR:
                    valid_train_indices.append(i)
                    self.julian_days.append(dt_obj.timetuple().tm_yday)
            
            num_frames = len(valid_train_indices)
            self.julian_days = np.array(self.julian_days)

            self.raw_data = {}
            for metric in ['evi_map', 'sliding_volume_map', 'msd_map', 'sliding_volume_z_score', 'sliding_volume_local_z_score']:
                if metric in self.active_metrics:
                    self.raw_data[metric] = data_grp[metric][valid_train_indices, ...]

            height, width = sr_ds.shape[2], sr_ds.shape[3]
            self.height, self.width = height, width
            
            valid_mask = self._get_pixel_mask(data_grp, valid_train_indices, height, width)
            
        print("Restructuring 3D spatiotemporal array into 1D pixel trajectories...")
        self.sequences = []
        stride = 5 
        
        for y in tqdm(range(0, height, stride), desc=f"Extracting valid trajectories for {self.active_metrics}"):
            for x in range(0, width, stride):
                pixel_valid_frames = valid_mask[:, y, x]
                
                if np.sum(pixel_valid_frames) < 15:
                    continue
                
                # Bipolar Geographic Coordinate Injection [-1.0 to 1.0]
                x_ratio = 2.0 * (x / max(1, self.width - 1)) - 1.0
                y_ratio = 2.0 * (y / max(1, self.height - 1)) - 1.0
                
                seq_dict = {
                    'days': self.julian_days[pixel_valid_frames][-self.max_seq_len:],
                    'x_ratio': x_ratio,
                    'y_ratio': y_ratio,
                    'true_len': len(self.julian_days[pixel_valid_frames][-self.max_seq_len:])
                }
                
                for metric in self.active_metrics:
                    pixel_vals = self.raw_data[metric][pixel_valid_frames, y, x]
                    scaled_vals = pixel_vals * self.metric_scalars[metric]
                    
                    trans_type, offset, scale = self.metric_transforms[metric]
                    
                    if trans_type == 'log10_shift':
                        scaled_vals = (np.log10(np.clip(scaled_vals, 1e-12, None)) + offset) / scale
                    elif trans_type == 'log1p':
                        scaled_vals = np.log1p(np.clip(scaled_vals, 0, None))
                    elif trans_type == 'z_score_dynamic':
                        mean_val = np.nanmean(scaled_vals)
                        std_val = np.nanstd(scaled_vals)
                        if std_val > 1e-6:
                            scaled_vals = (scaled_vals - mean_val) / std_val
                        else:
                            scaled_vals = scaled_vals - mean_val
                            
                    seq_dict[metric] = scaled_vals[-self.max_seq_len:]
                    
                self.sequences.append(seq_dict)
                
        print(f"Constructed {len(self.sequences)} valid pixel time-series for training.")

    def _get_pixel_mask(self, data_grp, valid_indices, height, width):
        num_frames = len(valid_indices)
        valid_mask = np.ones((num_frames, height, width), dtype=bool)
        sun_elev_arr = data_grp['surface_reflectance'].attrs.get('sun_elevation')
        kernel = np.ones((3, 3), dtype=bool)

        # Pre-load aerosol data for the masking loop
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

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = seq['true_len']
        
        # Dynamically size the tensor based on the requested encodings
        features = np.zeros((self.max_seq_len, self.num_metrics + self.num_context), dtype=np.float32)
        DOY_ratio = (seq['days'] / 365.25)
        
        for m_idx, metric in enumerate(self.active_metrics):
            features[:seq_len, m_idx] = seq[metric]
            
        # Dynamically inject the requested contextual features
        for p_idx, enc_type in enumerate(self.pos_encodings):
            feature_idx = self.num_metrics + p_idx
            
            if enc_type == 'DOY_ratio':
                features[:seq_len, feature_idx] = 2.0 * DOY_ratio - 1.0 
            elif enc_type == 'DOY_sin':
                features[:seq_len, feature_idx] = np.sin(2.0 * math.pi * DOY_ratio)
            elif enc_type == 'DOY_cos':
                features[:seq_len, feature_idx] = np.cos(2.0 * math.pi * DOY_ratio)
            elif enc_type == 'xPos_ratio':
                features[:seq_len, feature_idx] = seq['x_ratio']
            elif enc_type == 'yPos_ratio':
                features[:seq_len, feature_idx] = seq['y_ratio']
            else:
                raise ValueError(f"Unknown POSITIONAL_ENCODING configuration: {enc_type}")
        
        padding_mask = np.ones(self.max_seq_len, dtype=bool)
        padding_mask[:seq_len] = False
        return {'features': torch.from_numpy(features), 'padding_mask': torch.from_numpy(padding_mask)}

# ==========================================
# 3. MODEL ARCHITECTURE (Transformer)
# ==========================================
class TemporalAttentionAutoencoder(nn.Module):
    def __init__(self, input_channels, output_channels, d_model=128, n_heads=8, num_layers=6, dim_feedforward=512):
        super().__init__()
        
        self.num_metrics = output_channels
        self.num_context = input_channels - output_channels 
        
        # LayerNorm applied to inputs before addition to balance feature variance
        self.phys_proj = nn.Sequential(nn.Linear(self.num_metrics, d_model), nn.LayerNorm(d_model))
        self.context_proj = nn.Sequential(nn.Linear(self.num_context, d_model), nn.LayerNorm(d_model))
        
        # The learnable [MASK] token representation
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, output_channels)
        
    def forward(self, x, padding_mask, rand_mask=None):
        phys_x = x[:, :, :self.num_metrics]
        context_x = x[:, :, self.num_metrics:]
        
        phys_emb = self.phys_proj(phys_x)
        
        if rand_mask is not None:
            phys_emb = torch.where(rand_mask.unsqueeze(-1), self.mask_token, phys_emb)
            
        time_emb = self.context_proj(context_x)
        
        x_emb = phys_emb + time_emb
        latent = self.transformer(x_emb, src_key_padding_mask=padding_mask)
        return self.output_head(latent)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Temporal Autoencoder on {device}...")
    
    # Pass POSITIONAL_ENCODING configuration down to the dataset generator
    dataset = PixelTimeSeriesDataset(
        H5_PATH, TRAINED_METRICS, METRIC_SCALARS, METRIC_TRANSFORMS, 
        max_seq_len=MAX_SEQ_LEN, pos_encodings=POSITIONAL_ENCODING
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Dynamically scale input channels based on the user's config
    in_channels = dataset.num_metrics + dataset.num_context
    out_channels = dataset.num_metrics
    
    model = TemporalAttentionAutoencoder(
        input_channels=in_channels, output_channels=out_channels, 
        d_model=D_MODEL, n_heads=N_HEADS, num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD
    ).to(device)
    
    print("Initializing Exponential Moving Average (EMA) shadow weights...")
    ema_avg_fn = get_ema_multi_avg_fn(0.999)
    ema_model = AveragedModel(model, multi_avg_fn=ema_avg_fn)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    print(f"\n--- Starting Masked Time-Series Training for {TRAINED_METRICS} ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            features = batch['features'].to(device)        
            padding_mask = batch['padding_mask'].to(device)
            batch_size, seq_len, _ = features.shape
            
            # Generate the boolean mask tracking which tokens to hide
            rand_mask = torch.rand((batch_size, seq_len), device=device) < MASK_PROB
            rand_mask = rand_mask & (~padding_mask)
            
            optimizer.zero_grad()
            
            predictions = model(features, padding_mask, rand_mask=rand_mask) 
            true_targets = features[:, :, :dataset.num_metrics]
            
            valid_tokens = ~padding_mask
            
            loss = criterion(predictions[valid_tokens], true_targets[valid_tokens])
            
            loss.backward()
            optimizer.step()
            
            # Update EMA shadow weights
            ema_model.update_parameters(model)
            
            running_loss += loss.item()
            pbar.set_postfix({"MSE Loss": f"{loss.item():.5f}"})
            
        epoch_loss = running_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Completed | Validation MSE: {epoch_loss:.5f} | Current LR: {current_lr:.2e}")
        
        scheduler.step(epoch_loss)
        if current_lr < 1e-8:
            print(f"Learning rate reduced to less than {1e-8:.2e}, stopping training.")
            break

    torch.save(ema_model.module.state_dict(), MODEL_WEIGHTS)
    print(f"\nTraining complete. EMA smoothed model saved to {MODEL_WEIGHTS}")

if __name__ == "__main__":
    main()