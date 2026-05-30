import os
# Resolves OpenMP DLL conflict between PyTorch and Matplotlib/Intel MKL in Anaconda
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

import h5py
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
from tqdm import tqdm
import SpecComplex as sc 
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Rochester"
Frame_Reg = "WRS16"
landsat_path = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025_{Frame_Reg}_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
OUTPUT_H5 = f"C:/satelliteImagery/LANDSAT/{Location}/CCD_ODE_RNN_{Location}.h5"

# RECONFIGURABLE FEATURE SET
TARGET_METRICS = ['sliding_volume_z_score_masked']
INPUT_DIM = len(TARGET_METRICS)

# ODE-RNN ARCHITECTURE
HIDDEN_DIM = 8
NUM_LAYERS = 2  
SOLVER = 'rk4'
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 1024 

# CCDC THRESHOLDS
TRAIN_END_YEAR = 2023
CONSECUTIVE_ANOMALIES = 3
MIN_TRAIN_OBS = 20 

# ==========================================
# 2. PYTORCH NEURAL ODE MODULES
# ==========================================
class ODEFunc(nn.Module):
    """ Defines continuous derivative: dh(t)/dt = f(h(t), t, theta) """
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim * 2)),
            nn.Tanh(),
            torch.nn.utils.spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim * 2)),
            nn.Tanh(),
            torch.nn.utils.spectral_norm(nn.Linear(hidden_dim * 2, hidden_dim)),
            # DOCTORAL MANDATE: Absolute Derivative Bound.
            # Guarantees dh/dt is strictly in [-1, 1], making exponential divergence 
            # to infinity mathematically impossible during RK4 integration.
            nn.Tanh() 
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

class ODERNN(nn.Module):
    """ Combines continuous ODE integration with discrete RNN jumps governed by a QA Mask. """
    def __init__(self, input_dim, hidden_dim):
        super(ODERNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.ode_func = ODEFunc(hidden_dim)
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        
        self.decoder = nn.Linear(hidden_dim, input_dim) 
        # Initialize decoder weights small to prevent massive initial MSE
        nn.init.normal_(self.decoder.weight, mean=0, std=0.01)
        nn.init.constant_(self.decoder.bias, val=0)

    def forward(self, x, t, mask):
        T, B, _ = x.shape
        device = x.device
        
        h = torch.zeros(B, self.hidden_dim).to(device)
        predictions = []
        hidden_states = []
        
        for i in range(T):
            if i == 0:
                h_minus = h 
            else:
                t_span = torch.stack([t[i-1], t[i]])
                # Added internal sub-stepping options={'step_size': 0.05} (~18 days).
                # Forces RK4 to safely integrate across massive multi-month cloudy gaps 
                # without overshooting the vector field curvature.
                h_trajectory = odeint(
                    self.ode_func, h, t_span, 
                    method=SOLVER, 
                    options={'step_size': 0.05}
                )
                h_minus = h_trajectory[-1] 
                
            x_pred = self.decoder(h_minus)
            predictions.append(x_pred)
            hidden_states.append(h_minus)
            
            h_jump = self.gru_cell(x[i], h_minus)
            m = mask[i].unsqueeze(1) 
            h = torch.where(m, h_jump, h_minus)
            
        return torch.stack(predictions), torch.stack(hidden_states)

# ==========================================
# 3. SPATIAL BATCHING DATASET
# ==========================================
class SpatialPixelDataset(Dataset):
    def __init__(self, multivariate_stack, valid_mask, y_coords, x_coords):
        self.X = torch.from_numpy(multivariate_stack[:, :, y_coords, x_coords]).float() 
        self.X = self.X.permute(0, 2, 1) 
        
        # Explicitly handling posinf/neginf to prevent 3.4e38 cast.
        # This is strictly a dummy tensor for the PyTorch Autograd unused branch; 
        # valid_mask entirely governs actual mathematical evaluation.
        self.X = torch.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.M = torch.from_numpy(valid_mask[:, y_coords, x_coords]).bool() 
        self.y_coords = y_coords
        self.x_coords = x_coords

    def __len__(self):
        return self.X.shape[1] 

    def __getitem__(self, idx):
        return {
            'x': self.X[:, idx, :],       
            'mask': self.M[:, idx],       
            'y_coord': self.y_coords[idx],
            'x_coord': self.x_coords[idx]
        }

def extract_fractional_years(acq_times):
    frac_years = []
    for dt in acq_times:
        dt_obj = datetime.datetime.fromtimestamp(float(dt), tz=datetime.timezone.utc)
        year = dt_obj.year
        start_of_year = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
        start_of_next = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
        year_duration = (start_of_next - start_of_year).total_seconds()
        elapsed = (dt_obj - start_of_year).total_seconds()
        frac_years.append(year + (elapsed / year_duration))
    return np.array(frac_years)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    print(f"Loading HDF5 data into memory...\nFeatures: {TARGET_METRICS}")
    with h5py.File(landsat_path, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/LANDSAT/Data Fields']
        sr_ds = data_grp['surface_reflectance']
        height, width = sr_ds.shape[2], sr_ds.shape[3]
        
        acq_times = sr_ds.attrs.get('acquisition_time')
        frac_years = extract_fractional_years(acq_times)
        
        feature_data = []
        for metric in TARGET_METRICS:
            if metric not in data_grp:
                raise KeyError(f"CRITICAL ERROR: Requested feature '{metric}' not found in HDF5.")
            feature_data.append(data_grp[metric][...])
        
        multivariate_stack = np.stack(feature_data, axis=1) 
        num_frames = len(frac_years)
        valid_mask = np.zeros((num_frames, height, width), dtype=bool)
        
        print("Generating rigorous QA Masks via SpecComplex...")
        for f_idx in tqdm(range(num_frames), desc="Masking frames", leave=False):
            valid_mask[f_idx] = sc.get_landsat_mask(
                data_grp, f_idx, (height, width), 
                sun_elevation_threshold=30, 
                cloud_dilation=1
            )
        
        # STRICT DATA EXCLUSION:
        # Invalidating NaN AND Infinite values strictly excludes them from the valid_mask, 
        # ensuring the MSE Loss never touches anomalous optical data without requiring imputation.
        for f_idx in range(INPUT_DIM):
            valid_mask &= ~np.isnan(multivariate_stack[:, f_idx, ...])
            valid_mask &= ~np.isinf(multivariate_stack[:, f_idx, ...])
            
    t_global = frac_years
    
    if not np.all(np.diff(t_global) > 0):
        raise ValueError("CRITICAL ERROR: Timestamps in the HDF5 are not strictly monotonically increasing. Please correct data pipeline.")
    
    train_time_idx = frac_years <= (TRAIN_END_YEAR + 1.0)
    test_time_idx = frac_years > (TRAIN_END_YEAR + 1.0)
    
    t_train = t_global[train_time_idx]
    
    t_anchor = t_train[0] 
    t_train_scaled = t_train - t_anchor
    t_global_scaled = t_global - t_anchor

    train_valid_counts = np.sum(valid_mask[train_time_idx, :, :], axis=0)
    valid_spatial_mask = train_valid_counts >= MIN_TRAIN_OBS
    
    valid_y_coords, valid_x_coords = np.where(valid_spatial_mask)
    num_valid_pixels = len(valid_y_coords)
    print(f"\nExtracted {num_valid_pixels} spatially valid pixels for vectorized processing.")

    if num_valid_pixels == 0:
        raise ValueError("CRITICAL ERROR: No pixels meet the minimum observation threshold.")

    dataset = SpatialPixelDataset(multivariate_stack, valid_mask, valid_y_coords, valid_x_coords)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Vectorized ODE-RNN on {device}. Batch Size: {BATCH_SIZE}")
    
    model = ODERNN(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    change_detected_map = np.zeros((height, width), dtype=np.uint8)
    change_date_map = np.zeros((height, width), dtype=np.float32)
    out_preds = np.zeros((num_frames, INPUT_DIM, height, width), dtype=np.float32)
    out_hidden = np.zeros((num_frames, HIDDEN_DIM, height, width), dtype=np.float32)

    model.train()
    epoch_losses = []
    epoch_grad_norms = []
    t_train_tensor = torch.tensor(t_train_scaled, dtype=torch.float32).to(device)
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        running_grad_norm = 0.0
        valid_batch_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            x_batch = batch['x'][:, train_time_idx, :].permute(1, 0, 2).to(device)
            m_batch = batch['mask'][:, train_time_idx].permute(1, 0).to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(x_batch, t_train_tensor, m_batch) 
            
            preds_shifted = predictions[:-1]
            targets_shifted = x_batch[1:]
            masks_shifted = m_batch[1:]
            
            valid_preds = preds_shifted[masks_shifted]
            valid_targets = targets_shifted[masks_shifted]
            
            if len(valid_preds) == 0: continue
            
            loss = torch.mean((valid_preds - valid_targets)**2)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            batch_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    batch_grad_norm += p.grad.data.norm(2).item() ** 2
            batch_grad_norm = batch_grad_norm ** 0.5
            running_grad_norm += batch_grad_norm
            
            optimizer.step()
            
            running_loss += loss.item()
            valid_batch_count += 1
            pbar.set_postfix({"MSE Loss": f"{loss.item():.5f}", "Grad Norm": f"{batch_grad_norm:.4f}"})
            
        if valid_batch_count > 0:
            epoch_losses.append(running_loss / valid_batch_count)
            epoch_grad_norms.append(running_grad_norm / valid_batch_count)

    print("\nGenerating training metric visualizations...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', color='#1f77b4', label='MSE Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(epoch_grad_norms) + 1), epoch_grad_norms, marker='s', color='#d62728', label='L2 Gradient Norm')
    plt.title('Average Gradient Norm per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plot_path = OUTPUT_H5.replace('.h5', '_Training_Metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(2) 

    model.eval()
    t_global_tensor = torch.tensor(t_global_scaled, dtype=torch.float32).to(device)
    
    print("\nExecuting Vectorized Spatial Anomaly Detection...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scanning Batches"):
            x_batch = batch['x'].permute(1, 0, 2).to(device)
            m_batch = batch['mask'].permute(1, 0).to(device)
            batch_y = batch['y_coord'].numpy()
            batch_x = batch['x_coord'].numpy()
            
            full_preds, full_hidden = model(x_batch, t_global_tensor, m_batch)
            
            out_preds[:, :, batch_y, batch_x] = full_preds.permute(0, 2, 1).cpu().numpy()
            out_hidden[:, :, batch_y, batch_x] = full_hidden.permute(0, 2, 1).cpu().numpy()
            
            train_preds = full_preds[train_time_idx]
            train_targets = x_batch[train_time_idx]
            train_masks = m_batch[train_time_idx]
            
            squared_errors = torch.mean((train_preds - train_targets)**2, dim=-1) 
            
            valid_counts = train_masks.sum(dim=0, keepdim=True).clamp(min=1)
            mu_pixel = (squared_errors * train_masks).sum(dim=0, keepdim=True) / valid_counts
            
            variance = (((squared_errors - mu_pixel)**2) * train_masks).sum(dim=0, keepdim=True) / valid_counts
            std_pixel = torch.sqrt(variance).clamp(min=1e-5)
            
            thresholds = mu_pixel + (3.0 * std_pixel) 
            
            test_preds = full_preds[test_time_idx]
            test_targets = x_batch[test_time_idx]
            test_masks = m_batch[test_time_idx]
            test_t = t_global[test_time_idx]
            
            test_errors = torch.mean((test_preds - test_targets)**2, dim=-1) 
            is_anomalous = (test_errors > thresholds) & test_masks 
            
            streak_counters = torch.zeros(x_batch.shape[1], dtype=torch.int32, device=device)
            first_break_dates = torch.zeros(x_batch.shape[1], dtype=torch.float32, device=device)
            has_broken = torch.zeros(x_batch.shape[1], dtype=torch.bool, device=device)
            
            for t_idx in range(len(test_t)):
                current_anomalies = is_anomalous[t_idx]
                current_valid = test_masks[t_idx]
                
                streak_counters = torch.where(current_anomalies, streak_counters + 1, 
                                    torch.where(current_valid, torch.zeros_like(streak_counters), streak_counters))
                
                just_broken = (streak_counters >= CONSECUTIVE_ANOMALIES) & (~has_broken)
                first_break_dates = torch.where(just_broken, torch.tensor(test_t[t_idx - CONSECUTIVE_ANOMALIES + 1], device=device), first_break_dates)
                has_broken = has_broken | just_broken
                
            has_broken_cpu = has_broken.cpu().numpy()
            dates_cpu = first_break_dates.cpu().numpy()
            
            broken_indices = np.where(has_broken_cpu)[0]
            for b_idx in broken_indices:
                change_detected_map[batch_y[b_idx], batch_x[b_idx]] = 1
                change_date_map[batch_y[b_idx], batch_x[b_idx]] = dates_cpu[b_idx]

    print(f"\nTotal ODE-RNN Structural Breaks Detected: {np.sum(change_detected_map)}")
    
    print(f"Saving Maps and Temporal Dynamics to {OUTPUT_H5}...")
    with h5py.File(OUTPUT_H5, 'w') as out_f:
        out_f.create_dataset('change_detected_map', data=change_detected_map, compression='gzip')
        out_f.create_dataset('change_date_map', data=change_date_map, compression='gzip')
        out_f.create_dataset('features', data=multivariate_stack, compression='gzip')
        out_f.create_dataset('masks', data=valid_mask, compression='gzip')
        out_f.create_dataset('time_steps', data=t_global, compression='gzip')
        out_f.create_dataset('ode_predictions', data=out_preds, compression='gzip')
        out_f.create_dataset('hidden_states', data=out_hidden, compression='gzip')
        
    print("Vectorized ODE-RNN Training Complete!")

if __name__ == "__main__":
    main()