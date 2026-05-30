"""
Transformer-based Continuous Change Detection Architecture
Designed for irregularly sampled, monochromatic remote sensing time series.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from typing import List, Dict, Tuple

# ==========================================
# 1. CONTINUOUS TIME EMBEDDING (Time2Vec)
# ==========================================
class Time2Vec(nn.Module):
    """
    Implements Time2Vec to handle irregularly sampled data without assuming 
    fixed-interval positional encoding. 
    Ref: Kazemi et al., 2019.
    """
    def __init__(self, out_features: int):
        super(Time2Vec, self).__init__()
        self.out_features = out_features
        
        # Linear component for non-periodic drift
        self.w0 = nn.parameter.Parameter(torch.randn(1, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1, 1))
        
        # Periodic components for phenology/seasonality
        self.W = nn.parameter.Parameter(torch.randn(1, out_features - 1))
        self.B = nn.parameter.Parameter(torch.randn(1, out_features - 1))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [Seq_Len, 1] tensor of continuous timestamps (e.g., fractional years)
        Returns: [Seq_Len, out_features]
        """
        linear = self.w0 * t + self.b0
        periodic = torch.sin(torch.matmul(t, self.W) + self.B)
        return torch.cat([linear, periodic], dim=-1)

# ==========================================
# 2. TRANSFORMER SEQUENCE MODEL
# ==========================================
class PhenologyTransformer(nn.Module):
    def __init__(self, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super(PhenologyTransformer, self).__init__()
        
        self.d_model = d_model
        # 1-dim scalar input + Time2Vec embedding
        self.time_embed = Time2Vec(out_features=d_model - 1)
        
        # The encoder models the temporal dynamics
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output maps the continuous hidden state to the expected baseline scalar
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Seq_Len, 1] - Monochromatic values (e.g., z-scores)
        t: [Batch, Seq_Len, 1] - Continuous time 
        """
        t_emb = self.time_embed(t)
        # Combine input signal with time embedding
        combined = torch.cat([x, t_emb], dim=-1)
        
        encoded = self.transformer(combined)
        pred = self.decoder(encoded)
        return pred

# ==========================================
# 3. DATA EXTRACTION & FILTERING
# ==========================================
def extract_valid_pixel_history(datacube: List[Dict], row: int, col: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts purely valid observations for a single pixel. 
    Explicitly refuses to pad or fill masked values.
    """
    valid_times = []
    valid_values = []
    
    start_time = datacube[0]['datetime_utc']
    
    for frame in datacube:
        mask_val = frame['qa_mask'][row, col]
        if mask_val:  # True = valid, cloud-free
            raw_val = frame['sliding_volume_z_score_masked'][row, col]
            
            # Fail fast if QA mask allowed a NaN through
            if np.isnan(raw_val):
                raise ValueError(f"NaN encountered in supposed valid pixel at {frame['datetime_utc']}. Check upstream mask generation.")
                
            # Convert datetime to continuous fractional years relative to series start
            delta_years = (frame['datetime_utc'] - start_time).total_seconds() / (365.25 * 24 * 3600)
            
            valid_times.append(delta_years)
            valid_values.append(raw_val)
            
    if len(valid_times) < 5:
        raise ValueError(f"Insufficient valid data ({len(valid_times)} observations). Cannot establish temporal baseline.")
        
    return np.array(valid_times), np.array(valid_values)

# ==========================================
# 4. SLIDING WINDOW INFERENCE (CCDC EMULATOR)
# ==========================================
def run_transformer_ccdc(
    times: np.ndarray, 
    values: np.ndarray, 
    window_years: float = 2.0, 
    k_consecutive: int = 5,
    lambda_thresh: float = 3.0,
    save_model_dir: str = None,
    pixel_id: str = "default_pixel"
) -> List[dict]:
    """
    Progresses through the time series, training a baseline on the historical 
    window and evaluating future frames for structural change.
    """
    changes_detected = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhenologyTransformer().to(device)
    
    # Robust loss to resist unmasked clouds in the training window
    criterion = nn.HuberLoss(delta=1.0) 
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    
    current_idx = 0
    total_obs = len(times)
    
    while current_idx < total_obs:
        window_start_time = times[current_idx]
        window_end_time = window_start_time + window_years
        
        # Identify valid indices strictly within the temporal window
        train_indices = np.where((times >= window_start_time) & (times < window_end_time))[0]
        
        if len(train_indices) < 10:
            # Advance index to find a denser temporal block
            current_idx += 1 
            continue
            
        t_train = torch.tensor(times[train_indices], dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
        x_train = torch.tensor(values[train_indices], dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
        
        # Fast local training of the baseline
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            # Predict t_i based on historical sequence (auto-encoding temporal structure)
            pred = model(x_train, t_train)
            loss = criterion(pred, x_train)
            loss.backward()
            optimizer.step()
            
        # Optional: Save the trained model state for configuration reviews
        if save_model_dir:
            os.makedirs(save_model_dir, exist_ok=True)
            model_filename = f"phenology_transformer_{pixel_id}_window_{window_start_time:.3f}.pt"
            save_path = os.path.join(save_model_dir, model_filename)
            torch.save(model.state_dict(), save_path)
            
        # Calculate baseline variance for dynamic thresholding
        model.eval()
        with torch.no_grad():
            train_preds = model(x_train, t_train)
            residuals = torch.abs(train_preds - x_train).cpu().numpy()
            sigma_train = np.std(residuals)
        
        # Evaluate future frames (The 'Next 5 frames' constraint)
        future_idx_start = train_indices[-1] + 1
        if future_idx_start + k_consecutive > total_obs:
            break # Reached end of time series
            
        future_indices = np.arange(future_idx_start, future_idx_start + k_consecutive)
        t_future = torch.tensor(times[future_indices], dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device)
        x_future = values[future_indices]
        
        with torch.no_grad():
            # In an autoregressive setup, we query the temporal embedding of the future
            # using the final learned state (represented here by zero-padding the future signal query)
            x_query = torch.zeros_like(t_future)
            preds = model(x_query, t_future).cpu().numpy().flatten()
            
        # Evaluate anomalies
        future_residuals = np.abs(preds - x_future)
        threshold = lambda_thresh * sigma_train
        
        if np.all(future_residuals > threshold):
            # Prolonged deviation confirmed. Record change.
            change_time = times[future_indices[0]]
            changes_detected.append({
                'time_fractional_year': change_time,
                'prior_sigma': sigma_train,
                'residuals': future_residuals.tolist()
            })
            
            # CCDC Logic: Reset the window to start *after* the confirmed change
            current_idx = future_indices[0]
            # Re-initialize model to prevent historical contamination
            model = PhenologyTransformer().to(device)
            optimizer = optim.AdamW(model.parameters(), lr=0.01)
        else:
            # No change. Advance window by 1 observation.
            current_idx += 1
            
    return changes_detected

# ==========================================
# Example Execution
# ==========================================
if __name__ == "__main__":
    # Assuming 'cube' is defined from your data_loader.py
    # cube, _, _ = load_merged_datacube(...)
    
    # Mocking execution for a single pixel at Row 100, Col 100
    row, col = 100, 100
    # times_ary, values_ary = extract_valid_pixel_history(cube, row, col)
    
    # Mock data to allow script to compile independently
    times_ary = np.sort(np.random.uniform(0, 10, 150))
    values_ary = np.sin(times_ary * 2 * np.pi) + np.random.normal(0, 0.1, 150)
    
    print(f"Executing Transformer-CCDC logic on pixel ({row}, {col})...")
    try:
        changes = run_transformer_ccdc(
            times_ary, 
            values_ary, 
            window_years=2.0, 
            k_consecutive=5,
            save_model_dir="./saved_models",
            pixel_id=f"r{row}_c{col}"
        )
        print(f"Algorithm complete. {len(changes)} structural changes detected.")
        for c in changes:
            print(f" - Change detected at year index: {c['time_fractional_year']:.2f}")
    except ValueError as e:
        print(f"Execution aborted: {e}")