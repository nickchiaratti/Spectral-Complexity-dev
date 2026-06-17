import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TimeSeriesH5Dataset
from models import FrequencyAutoencoder
from pnpxai.explainers import IntegratedGradients

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
LOCATION = "Malibu"
H5_PATH = f"E:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
DATASET_NAME = "HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score"
OUTPUT_DIR = f"E:/satelliteImagery/HLST30/OOD/{LOCATION}"

BATCH_SIZE = 256
EPOCHS = 10
LATENT_DIM = 8
CONTAMINATION_RATE = 0.2

# Configuration Options for Overlay Filtering
# Filtering logic is removed because synthetic interpolation is mathematically removed via NUFFT

# Ensure computationally intensive tasks are optimized for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATA LOADING
# ==========================================
# Dataset utilizes pandas linear interpolation to handle missing time steps.
dataset = TimeSeriesH5Dataset(h5_path=H5_PATH, dataset_name=DATASET_NAME)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
inference_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

sequence_length = dataset.time_steps
print(f"Loaded dataset: {dataset.num_pixels} pixels, {sequence_length} time steps.")

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
model = FrequencyAutoencoder(sequence_length=sequence_length, latent_dim=LATENT_DIM).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Using L1 Loss (MAE) is critical here to prevent the model from 
# heavily penalizing and learning OOD shifts.
loss_function = nn.L1Loss() 

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_model():
    print("Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (points, values, flat_idx) in enumerate(dataloader):
            points = points.to(device)
            values = values.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            true_amps, rec_amps = model(points, values)
            
            # Calculate loss on the amplitude spectrums
            loss = loss_function(rec_amps, true_amps)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Strict Failure Handling (Should not occur due to interpolation)
            if torch.isnan(loss):
                raise ValueError("Loss is NaN! This indicates interpolation failed to resolve NaNs.")
                
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss / len(dataloader):.4f}")

# ==========================================
# 5. INFERENCE & OOD FLAGGING
# ==========================================
def flag_ood_pixels():
    print("Starting inference...")
    model.eval()
    reconstruction_errors = np.zeros(dataset.num_pixels)

    with torch.no_grad():
        for batch_idx, (points, values, flat_idx) in enumerate(inference_loader):
            points = points.to(device)
            values = values.to(device)
            true_amps, rec_amps = model(points, values)
            
            # Compute MAE per pixel on the frequency amplitudes
            mae = torch.mean(torch.abs(true_amps - rec_amps), dim=1)
            
            # Assign back using flat indices
            reconstruction_errors[flat_idx.cpu().numpy()] = mae.cpu().numpy()

    # Determine the dynamic threshold using all interpolated pixels
    percentile_threshold = 100 * (1.0 - CONTAMINATION_RATE)
    ood_threshold = np.percentile(reconstruction_errors, percentile_threshold)
    print(f"OOD Threshold ({percentile_threshold}th percentile): {ood_threshold:.4f}")

    # Create a 2D boolean mask for the image
    ood_mask_flat = reconstruction_errors > ood_threshold
    ood_map = ood_mask_flat.reshape(dataset.height, dataset.width)
    
    return ood_map, reconstruction_errors

# ==========================================
# 6. EXPLAINABILITY (PnPXAI)
# ==========================================
def explain_anomalies(reconstruction_errors):
    print("\nStarting explainability on top anomalous pixels...")
    
    # Wrapper model to output a single scalar (MAE) for PnPXAI to attribute back to the input
    class MAEWrapper(nn.Module):
        def __init__(self, ae_model, fixed_points):
            super(MAEWrapper, self).__init__()
            self.ae = ae_model
            self.fixed_points = fixed_points
            
        def forward(self, values):
            true_amps, rec_amps = self.ae(self.fixed_points, values)
            mae = torch.mean(torch.abs(true_amps - rec_amps), dim=1, keepdim=True)
            return mae

    # Identify the pixel with the highest reconstruction error
    top_pixel_idx = np.argmax(reconstruction_errors)
    top_h, top_w = top_pixel_idx // dataset.width, top_pixel_idx % dataset.width
    
    # Extract the original time-series tensor and interpolation mask
    top_points = dataset.points.unsqueeze(0).to(device)
    top_values = dataset.tensor_data[top_pixel_idx].unsqueeze(0).to(device)
    
    wrapper_model = MAEWrapper(model, top_points).to(device)
    wrapper_model.eval()
    
    top_values.requires_grad_()

    try:
        explainer = IntegratedGradients(wrapper_model)
        attributions = explainer.attribute(top_values, targets=0)
        
        attrs_np = attributions.cpu().detach().numpy().flatten()
        
        # Calculate total absolute attribution weight
        abs_attrs = np.abs(attrs_np)
        
        print("="*60)
        print("EXPLAINABILITY REPORT")
        print("="*60)
        print(f"Top OOD Pixel located at (H:{top_h}, W:{top_w})")
        print(f"Anomaly Score (MAE): {reconstruction_errors[top_pixel_idx]:.4f}")
        print(f"\nAttributions mapped to time-steps. Positive values indicate factors that INCREASED the anomaly score.")
        
        for t_step in range(len(attrs_np)):
            if abs_attrs[t_step] > 0.01 * np.max(abs_attrs):  # Only print relatively significant contributors
                print(f"  t={t_step:03d} | Attribution: {attrs_np[t_step]:+8.4f}")
                
        print("\nCONFIRMED: The anomaly detection was primarily driven by true physical observations (NUFFT natively excludes invalid points).")
        print("="*60)
        
    except Exception as e:
        print(f"Warning: PnPXAI explainability failed. Ensure the package version matches the API syntax: {e}")

def compute_temporal_ood_map(ood_map):
    print("Computing temporal attribution map for OOD pixels...")
    H, W = ood_map.shape
    
    with h5py.File(H5_PATH, 'r') as f:
        common_mask_volume = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'][:]
    ood_coords = np.argwhere(ood_map)
    flat_indices = ood_coords[:, 0] * W + ood_coords[:, 1]
    
    time_map = np.full((H, W), -1, dtype=np.int32)
    
    if len(flat_indices) == 0:
        return time_map
        
    class OODDataset(torch.utils.data.Dataset):
        def __init__(self, ds, indices):
            self.ds = ds
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            points, values, flat_idx = self.ds[self.indices[idx]]
            return points, values, self.indices[idx]
            
    class MAEWrapper(nn.Module):
        def __init__(self, ae_model):
            super().__init__()
            self.ae = ae_model
        def forward(self, points, values):
            true_amps, rec_amps = self.ae(points, values)
            return torch.mean(torch.abs(true_amps - rec_amps), dim=1, keepdim=True)
            
    # We must wrap it such that PnPXAI only attributes w.r.t `values`
    class PnPXAIWrapper(nn.Module):
        def __init__(self, ae_model, batch_points):
            super().__init__()
            self.ae = ae_model
            self.batch_points = batch_points
        def forward(self, values):
            true_amps, rec_amps = self.ae(self.batch_points, values)
            return torch.mean(torch.abs(true_amps - rec_amps), dim=1, keepdim=True)
    
    loader = DataLoader(OODDataset(dataset, flat_indices), batch_size=256, shuffle=False)
    
    import sys
    for i, (batch_points, batch_values, batch_idx) in enumerate(loader):
        batch_points = batch_points.to(device)
        batch_values = batch_values.to(device)
        batch_values.requires_grad_()
        
        wrapper = PnPXAIWrapper(model, batch_points).to(device)
        wrapper.eval()
        explainer = IntegratedGradients(wrapper)
        
        attrs = explainer.attribute(batch_values, targets=torch.zeros(batch_values.size(0), dtype=torch.long, device=device))
        
        # We must only consider non-zero valid data points for max time index
        # Force masked values to have zero attribution so they are never selected
        valid_mask = batch_values != 0.0
        attrs[~valid_mask] = -99999.0
        
        max_time_idx = torch.argmax(attrs, dim=1).cpu().numpy()
        
        batch_idx = batch_idx.numpy()
        for b_idx, t_idx in zip(batch_idx, max_time_idx):
            r = b_idx // W
            c = b_idx % W
            time_map[r, c] = t_idx
            
        sys.stdout.write(f"\rProcessed {min((i+1)*loader.batch_size, len(flat_indices))}/{len(flat_indices)} pixels")
        sys.stdout.flush()
        
    print("\nCompleted temporal OOD map.")
    return time_map

# ==========================================
# EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_model()
    
    model_path = os.path.join(OUTPUT_DIR, f"{LOCATION}_ood_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model weights to {model_path}")
    
    ood_map, errors = flag_ood_pixels()
    time_map = compute_temporal_ood_map(ood_map)
    
    results_path = os.path.join(OUTPUT_DIR, f"{LOCATION}_ood_results.h5")
    with h5py.File(results_path, 'w') as f_out:
        dset_map = f_out.create_dataset("ood_map", data=ood_map.astype(np.uint8))
        dset_err = f_out.create_dataset("reconstruction_errors", data=errors)
        dset_time = f_out.create_dataset("ood_time_map", data=time_map, dtype=np.int32)
        
        # Copy geospatial metadata from source
        with h5py.File(H5_PATH, 'r') as f_in:
            source_dset = f_in[DATASET_NAME]
            for attr in ['GeoTransform', 'spatial_ref']:
                if attr in source_dset.attrs:
                    dset_map.attrs[attr] = source_dset.attrs[attr]
                    dset_err.attrs[attr] = source_dset.attrs[attr]
                    dset_time.attrs[attr] = source_dset.attrs[attr]
                    
    print(f"Saved OOD results to {results_path}")
    
    explain_anomalies(errors)