import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timezone
import numpy as np
import math
import os
from data_loader import load_merged_datacube, landsat_path, tanager_path

EPOCHS = 15
SEQUENCE_LENGTH = 20
BATCH_SIZE = 10
LEARNING_RATE = 1e-4
anomaly_threshold = 2.5 
# ==========================================
# ARCHITECTURE (Mask-Aware & Time2Vec)
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
        
        # DOCTORAL UPGRADE: Inductive Physical Bias
        # Since time is represented as fractional years, w = 2*pi represents exactly 1 Earth year.
        # We explicitly initialize the first few dimensions to target core ecological phenology harmonics
        # to prevent the network from having to guess the orbital period of the planet.
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
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True
        )
        self.reconstruction = SpatialReconstruction(d_model, h, w)

    def forward(self, src_safe, tgt_safe, src_time, tgt_time):
        # Spatially embed the physical data, then add the exact continuous temporal vector
        src_emb = self.embedding(src_safe) + self.time_encoder(src_time)
        tgt_emb = self.embedding(tgt_safe) + self.time_encoder(tgt_time)
        
        output = self.transformer(src_emb, tgt_emb)
        return self.reconstruction(output)

# ==========================================
# MASKED DATASET & LOSS
# ==========================================

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, tgt_safe, tgt_mask):
        if tgt_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        sq_err = (pred - tgt_safe) ** 2
        masked_err = sq_err * tgt_mask.float()
        return masked_err.sum() / tgt_mask.float().sum()

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
        
        # Yield continuous time sequences
        src_time = self.tensor_times[idx : idx + self.seq_length]
        tgt_time = self.tensor_times[idx + 1 : idx + self.seq_length + 1]
        
        return src_safe, tgt_safe, tgt_mask, src_time, tgt_time

# ==========================================
# PIPELINE EXECUTION
# ==========================================

def execute_pipeline():
    print("Ingesting merged multi-sensor datacube...")
    cube, h5_l, h5_t = load_merged_datacube(landsat_path, tanager_path)
    
    # Apply spatial QA mask post-retrieval
    print("Applying spatial QA masks to spectral volumes...")
    for frame in cube:
        frame['sliding_volume_z_score_masked'][~frame['qa_mask']] = np.nan
        
    split_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    
    train_frames = [f for f in cube if f['datetime_utc'] < split_date]
    eval_frames  = [f for f in cube if f['datetime_utc'] >= split_date]
    
    print(f"Data Split -> Training Frames (< 2024): {len(train_frames)} | Evaluation Frames (>= 2024): {len(eval_frames)}")
    
    if len(train_frames) == 0 or len(eval_frames) == 0:
        raise RuntimeError("Insufficient data in temporal split to proceed.")

    # A sequence length of 10 combined with Time2Vec should now mathematically recognize
    # that multiple months/seasons have passed, even if data points are dropped.
    
    train_dataset = SpatioTemporalDataset(train_frames, seq_length=SEQUENCE_LENGTH)
    eval_dataset  = SpatioTemporalDataset(eval_frames, seq_length=SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader  = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    _, H, W = train_dataset[0][0].shape 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Initializing Time2Vec Masked Transformer Grid: [{H}x{W}] on {device}")
    model = SpatioTemporalTransformer(h=H, w=W, d_model=128, nhead=8).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = MaskedMSELoss() 

    # --- Training Phase ---
    model.train()
    print("\n--- Commencing Training ---")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        valid_batches = 0
        for batch_idx, (src_safe, tgt_safe, tgt_mask, src_time, tgt_time) in enumerate(train_loader):
            # 1. Move all vectors to device
            src_safe = src_safe.to(device)
            tgt_safe = tgt_safe.to(device)
            tgt_mask = tgt_mask.to(device)
            src_time = src_time.to(device)
            tgt_time = tgt_time.to(device)
            
            # 2. Forward pass (now requires time)
            optimizer.zero_grad()
            output = model(src_safe, tgt_safe, src_time, tgt_time)
            
            # 3. Calculate strictly bounded physical loss
            loss = criterion(output, tgt_safe, tgt_mask)
            
            # 4. Backpropagation
            if loss.item() > 0:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                valid_batches += 1
                
        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{EPOCHS} | Avg Masked MSE Loss: {avg_loss:.6f}")

    # --- Model Checkpointing ---
    save_dir = r"C:\satelliteImagery\transformerModel"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "spatiotemporal_transformer_mask_aware.pth")
    print(f"\nSaving trained model state to: {save_path}")
    
    torch.save({
        'epoch': EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'spatial_dims': (H, W)
    }, save_path)

    # --- Evaluation Phase ---
    model.eval()
    print("\n--- Commencing Post-2024 Anomaly Evaluation ---")
    
    
    with torch.no_grad():
        for batch_idx, (src_safe, tgt_safe, tgt_mask, src_time, tgt_time) in enumerate(eval_loader):
            src_safe = src_safe.to(device)
            tgt_safe = tgt_safe.to(device)
            tgt_mask = tgt_mask.to(device)
            src_time = src_time.to(device)
            tgt_time = tgt_time.to(device)
            
            predicted_tgt = model(src_safe, tgt_safe, src_time, tgt_time)
            
            pred_frame = predicted_tgt[0, -1, :, :].cpu().numpy()
            true_frame = tgt_safe[0, -1, :, :].cpu().numpy()
            mask_frame = tgt_mask[0, -1, :, :].cpu().numpy()
            
            residual_map = np.full((H, W), np.nan)
            residual_map[mask_frame] = np.abs(pred_frame[mask_frame] - true_frame[mask_frame])
            
            with np.errstate(invalid='ignore'):
                anomaly_mask = residual_map > anomaly_threshold
                
            anomaly_count = np.sum(anomaly_mask)
            max_res = np.nanmax(residual_map) if np.any(mask_frame) else 0.0
            
            original_frame = eval_frames[batch_idx + SEQUENCE_LENGTH]
            timestamp_str = original_frame['datetime_utc'].strftime('%Y-%m-%d')
            sensor = original_frame['spacecraft']
            
            print(f"Eval Frame {batch_idx+1} ({timestamp_str} - {sensor}): "
                  f"Max Valid Residual = {max_res:.4f} | Anomalous Pixels = {anomaly_count}")

    h5_l.close()
    h5_t.close()

if __name__ == "__main__":
    execute_pipeline()