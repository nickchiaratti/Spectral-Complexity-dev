import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import rasterio
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.models import ResNet50_Weights
import torchgeo.models
from tqdm import tqdm
from scipy import ndimage

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# Must point to an HDF5 file containing 'sliding_volume_map' if INCLUDE_SPECTRAL_COMPLEXITY = True
H5_TRAIN_PATH = "C:/satelliteImagery/LANDSAT/Rochester/LANDSAT_Stack_Rochester_HDFEOS_SC_EM-7_Gram-datasetMean_Norm-bandCount_Aerosol-low_QA-AllFrames_sunElMin-40.h5"

# Multi-Year Ground Truth paths - Expects the outputs from the multi-year align-ground-truth.py
CDL_PATHS = {
    2023: "C:/satelliteImagery/LANDSAT/Rochester/CDL2023_Aligned_Rochester.tif",
    2024: "C:/satelliteImagery/LANDSAT/Rochester/CDL2024_Aligned_Rochester.tif",
    2025: "C:/satelliteImagery/LANDSAT/Rochester/CDL2025_Aligned_Rochester.tif",
}

# --- MODEL INPUT CONFIGURATION ---
INCLUDE_SPECTRAL_COMPLEXITY = False  # Set to False to train the standard 7-band model baseline

# Output Weights
out_name = "8band_surgery" if INCLUDE_SPECTRAL_COMPLEXITY else "7band_baseline"
OUTPUT_WEIGHTS = f"C:/satelliteImagery/LANDSAT/Rochester/rochester_cdl_{out_name}.pth"

# Training Hyperparameters
PATCH_SIZE = 256
BATCH_SIZE = 16       
EPOCHS = 20
LEARNING_RATE = 1e-4

# --- MULTI-TEMPORAL CONFIGURATION ---
TIME_INDICES = 'all'  

# Relaxed QA masking for deep learning
SUN_ELEVATION_THRESHOLD = 25
CLOUD_DILATION = 0
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_VALUES = 'all'

# ==========================================
# 2. DATA UTILITY
# ==========================================
def _get_landsat_mask(data_grp, num_frames, height, width):
    valid_mask = np.ones((num_frames, height, width), dtype=bool)
    sun_elev_arr = data_grp['surface_reflectance'].attrs.get('sun_elevation')
    kernel = np.ones((3, 3), dtype=bool)

    dropped_by_sun = 0

    for f_idx in range(num_frames):
        if sun_elev_arr is not None and f_idx < len(sun_elev_arr):
            if sun_elev_arr[f_idx] < SUN_ELEVATION_THRESHOLD:
                valid_mask[f_idx] = False
                dropped_by_sun += 1
                continue

        if 'QUALITY_L1_PIXEL' in data_grp:
            qa_pixel = data_grp['QUALITY_L1_PIXEL'][f_idx, ...]
            bad_qa_mask = (qa_pixel & QA_REJECT_MASK) != 0
            if CLOUD_DILATION > 0:
                bad_qa_mask = ndimage.binary_dilation(bad_qa_mask, structure=kernel, iterations=CLOUD_DILATION)
            valid_mask[f_idx] &= ~bad_qa_mask

        if 'RADIOMETRIC_SATURATION' in data_grp:
            bad_radsat = data_grp['RADIOMETRIC_SATURATION'][f_idx, ...] != RADSAT_ACCEPT_VALUE
            if CLOUD_DILATION > 0:
                bad_radsat = ndimage.binary_dilation(bad_radsat, structure=kernel, iterations=CLOUD_DILATION)
            valid_mask[f_idx] &= ~bad_radsat

        if 'QUALITY_L2_AEROSOL' in data_grp and AEROSOL_ACCEPT_VALUES != 'all':
            aerosol = data_grp['QUALITY_L2_AEROSOL'][f_idx, ...]
            invalid_aerosol = ~np.isin(aerosol, AEROSOL_ACCEPT_VALUES)
            if CLOUD_DILATION > 0:
                invalid_aerosol = ndimage.binary_dilation(invalid_aerosol, structure=kernel, iterations=CLOUD_DILATION)
            valid_mask[f_idx] &= ~invalid_aerosol
            
    print(f"Mask Diagnostics: {dropped_by_sun} / {num_frames} frames dropped by Sun Elevation (<{SUN_ELEVATION_THRESHOLD}°).")
    return valid_mask

# ==========================================
# 3. PYTORCH DATASET DEFINITION
# ==========================================
class LandsatCDLDataset(Dataset):
    def __init__(self, h5_path, cdl_paths, patch_size=256, time_indices='all', include_sc=True):
        self.patch_size = patch_size
        self.include_sc = include_sc
        
        print(f"Loading HDF5 Surface Reflectance {'and Spectral Complexity ' if include_sc else ''}into memory...")
        with h5py.File(h5_path, 'r') as f:
            data_grp = f['/HDFEOS/GRIDS/LANDSAT/Data Fields']
            sr_ds = data_grp['surface_reflectance']
            
            # Extract and Parse Acquisition Times (Handles Unix Timestamps properly!)
            acq_times = sr_ds.attrs.get('acquisition_time')
            if acq_times is None:
                raise ValueError("Missing 'acquisition_time' attribute in HDF5 dataset.")
            
            self.frame_years = []
            for dt in acq_times:
                try:
                    # Parse as float (Unix timestamp in seconds)
                    year = datetime.datetime.fromtimestamp(float(dt)).year
                except ValueError:
                    # Fallback string parsing if format changes
                    dt_str = dt.decode('utf-8') if isinstance(dt, bytes) else str(dt)
                    year = int(dt_str[:4])
                self.frame_years.append(year)
            
            total_frames = sr_ds.shape[0]
            height, width = sr_ds.shape[2], sr_ds.shape[3]
            if time_indices == 'all':
                self.t_idxs = list(range(total_frames))
            else:
                self.t_idxs = time_indices
                
            print("Generating spatial valid masks...")
            full_mask = _get_landsat_mask(data_grp, total_frames, height, width)
            self.valid_mask = full_mask[self.t_idxs, ...]
                
            # Load surface reflectance (Shape: [frames, 7, H, W])
            sr_images = np.nan_to_num(sr_ds[self.t_idxs, ...], nan=0.0)
            
            if self.include_sc:
                sc_ds = data_grp['sliding_volume_map']
                sc_images = np.nan_to_num(sc_ds[self.t_idxs, ...], nan=0.0)
                sc_images = np.expand_dims(sc_images, axis=1) # Shape: [frames, 1, H, W]
                # Concatenate to create an 8-channel input stack
                self.images = np.concatenate((sr_images, sc_images), axis=1)
            else:
                self.images = sr_images
            
            # Zero out the masked pixels across all bands using np.where
            self.images = np.where(self.valid_mask[:, None, :, :], self.images, 0.0)
            
        self.num_loaded_frames, self.channels, self.height, self.width = self.images.shape
        print(f"Loaded {self.num_loaded_frames} temporal frames with {self.channels} channels.")
        
        print("Loading Aligned CDL Masks into memory...")
        self.masks = {}
        required_years = set([self.frame_years[t] for t in self.t_idxs])
        
        for year in required_years:
            if year not in cdl_paths:
                print(f"WARNING: No CDL path configured for year {year}. Frames from this year will be skipped.")
                continue
                
            cdl_path = cdl_paths[year]
            if not os.path.exists(cdl_path):
                print(f"WARNING: Aligned CDL for {year} not found at {cdl_path}. Skipping.")
                continue
                
            with rasterio.open(cdl_path) as src:
                # The arrays are ALREADY mapped 1 to N by the alignment script!
                self.masks[year] = src.read(1).astype(np.int64)
            print(f"Successfully loaded CDL data for {year}.")

        # Dynamically determine the total number of classes across all loaded masks
        max_class = 0
        for mask in self.masks.values():
            max_class = max(max_class, int(np.max(mask)))
        self.num_classes = max_class + 1
        print(f"Dynamically detected {self.num_classes} total global classes (including background).")

        print("Calculating sliding window patches across all time indices...")
        self.patches = []
        stride = patch_size // 2  
        
        for t in range(self.num_loaded_frames):
            original_t_idx = self.t_idxs[t]
            year = self.frame_years[original_t_idx]
            
            if year not in self.masks:
                continue
                
            current_mask = self.masks[year]
            
            for y in range(0, self.height - patch_size + 1, stride):
                for x in range(0, self.width - patch_size + 1, stride):
                    
                    mask_patch = current_mask[y:y+self.patch_size, x:x+self.patch_size].copy()
                    valid_patch = self.valid_mask[t, y:y+self.patch_size, x:x+self.patch_size]
                    mask_patch[~valid_patch] = 0
                    
                    valid_pixel_count = np.sum(mask_patch > 0)
                    total_pixels = self.patch_size * self.patch_size
                    
                    # Require 1% valid pixels to qualify as a training patch
                    if (valid_pixel_count / total_pixels) >= 0.01:
                        self.patches.append((t, y, x))
                
        print(f"Dataset initialized with {len(self.patches)} valid training patches.")
        if len(self.patches) == 0:
            raise ValueError("CRITICAL: 0 valid training patches found! Your QA Masks filtered out 100% of the dataset.")
                
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        t, y, x = self.patches[idx]
        img_patch = self.images[t, :, y:y+self.patch_size, x:x+self.patch_size]
        
        original_t_idx = self.t_idxs[t]
        year = self.frame_years[original_t_idx]
        
        mask_patch = self.masks[year][y:y+self.patch_size, x:x+self.patch_size].copy()
        valid_patch = self.valid_mask[t, y:y+self.patch_size, x:x+self.patch_size]
        mask_patch[~valid_patch] = 0
        
        return {
            'image': torch.from_numpy(img_patch).float(),
            'mask': torch.from_numpy(mask_patch).long()
        }

# ==========================================
# 4. MAIN TRAINING LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")
    
    dataset = LandsatCDLDataset(H5_TRAIN_PATH, CDL_PATHS, PATCH_SIZE, TIME_INDICES, include_sc=INCLUDE_SPECTRAL_COMPLEXITY)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    landsat_weight = ResNet50_Weights.LANDSAT_OLI_SR_MOCO 
    print(f"Building U-Net model with pre-trained weights: {landsat_weight}")

    task = SemanticSegmentationTask(
        model="unet", backbone="resnet50", weights=landsat_weight, 
        in_channels=7, num_classes=dataset.num_classes,
    )
    model = task.model.to(device)

    # Perform weight surgery if the user toggled 8+ channels on
    if dataset.channels > 7:
        print(f"Performing weight surgery: Adapting 7-channel weights for {dataset.channels} inputs...")
        
        old_conv1 = model.encoder.conv1
        new_conv1 = nn.Conv2d(
            in_channels=dataset.channels, out_channels=old_conv1.out_channels, 
            kernel_size=old_conv1.kernel_size, stride=old_conv1.stride, 
            padding=old_conv1.padding, bias=False
        ).to(device)
        
        with torch.no_grad():
            new_conv1.weight[:, :7, :, :] = old_conv1.weight.clone()
            new_conv1.weight[:, 7:, :, :] = old_conv1.weight.mean(dim=1, keepdim=True).repeat(1, dataset.channels - 7, 1, 1)
            
        model.encoder.conv1 = new_conv1

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print("\n--- Starting Fine-Tuning ---")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        valid_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            inputs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, masks)
            if torch.isnan(loss):
                continue
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            valid_batches += 1
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        epoch_loss = running_loss / max(1, valid_batches)
        print(f"Epoch {epoch+1} Completed | Average Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), OUTPUT_WEIGHTS)
    print(f"\nFine-tuning complete. Model saved to {OUTPUT_WEIGHTS}")

if __name__ == "__main__":
    main()