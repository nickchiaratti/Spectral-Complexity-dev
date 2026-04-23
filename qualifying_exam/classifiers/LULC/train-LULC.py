import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, Resampling
from torchgeo.trainers import SemanticSegmentationTask
from torchgeo.models import ResNet50_Weights
import torchgeo.models
from tqdm import tqdm
from scipy import ndimage

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
H5_TRAIN_PATH = "C:/satelliteImagery/LANDSAT/Rochester/LANDSAT_Stack_Rochester_HDFEOS.h5"

# Ground Truth paths - Map the years to their respective aligned masks
CDL_PATHS = {
    2023: "C:/satelliteImagery/LANDSAT/Rochester/CDL2023_Aligned_Rochester.tif",
    2024: "C:/satelliteImagery/LANDSAT/Rochester/CDL2024_Aligned_Rochester.tif",
    2025: "C:/satelliteImagery/LANDSAT/Rochester/CDL2025_Aligned_Rochester.tif"
}

# Output Weights
OUTPUT_WEIGHTS = "C:/satelliteImagery/LANDSAT/Rochester/rochester_cdl_resnet50_unet.pth"

# Training Hyperparameters
PATCH_SIZE = 256
BATCH_SIZE = 16       
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 12      

# --- MULTI-TEMPORAL CONFIGURATION ---
TIME_INDICES = 'all'  

# --- Pixel Masking Configuration ---
SUN_ELEVATION_THRESHOLD = 25
CLOUD_DILATION = 0
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_VALUES = [2, 4, 32, 66, 68, 96, 100, 130, 132, 160, 164] # 'medium' level

# ==========================================
# 2. DATA ALIGNMENT UTILITY
# ==========================================
def align_ground_truth_to_hdf5(raw_mask_path, aligned_mask_path, h5_path):
    print(f"Aligning {raw_mask_path} to match HDF5 grid from {h5_path}...")
    
    with h5py.File(h5_path, 'r') as f:
        ds = f['/HDFEOS/GRIDS/LANDSAT/Data Fields/surface_reflectance']
        h_30, w_30 = ds.shape[2], ds.shape[3]

        # Ensure spatial reference is safely decoded if stored as bytes
        spatial_ref = ds.attrs['spatial_ref']
        if isinstance(spatial_ref, bytes):
            spatial_ref = spatial_ref.decode('utf-8')
        dst_crs = rasterio.crs.CRS.from_user_input(spatial_ref)
        
        # Directly unpack the Rasterio-ordered attributes from the HDF5
        dst_transform = rasterio.Affine(*ds.attrs['GeoTransform'])
        
    with rasterio.open(raw_mask_path) as src:
        aligned_mask = np.zeros((h_30, w_30), dtype='uint8')
        source_crs = rasterio.crs.CRS.from_epsg(5070)
        with WarpedVRT(src, src_crs=source_crs, crs=dst_crs, transform=dst_transform, width=w_30, height=h_30, resampling=Resampling.nearest) as vrt:
            aligned_mask = vrt.read(1)
            
        print("Reprojection Complete")
        profile = src.profile.copy()
        profile.update({
            'crs': dst_crs, 'transform': dst_transform,
            'width': w_30, 'height': h_30, 'dtype': 'uint8'
        })
        
        with rasterio.open(aligned_mask_path, 'w', **profile) as dst:
            dst.write(aligned_mask, 1)
            
    print(f"Aligned ground truth saved to {aligned_mask_path}")

def _get_landsat_mask(data_grp, num_frames, height, width):
    """Generates a boolean mask for LANDSAT data based on active filters."""
    valid_mask = np.ones((num_frames, height, width), dtype=bool)
    sun_elev_arr = data_grp['surface_reflectance'].attrs.get('sun_elevation')
    kernel = np.ones((3, 3), dtype=bool)
    
    dropped_by_sun = 0
    pixels_dropped_qa = 0
    pixels_dropped_radsat = 0
    pixels_dropped_aerosol = 0

    for f_idx in range(num_frames):
        if sun_elev_arr is not None and f_idx < len(sun_elev_arr):
            if sun_elev_arr[f_idx] < SUN_ELEVATION_THRESHOLD:
                valid_mask[f_idx] = False
                dropped_by_sun += 1
                continue

        # QA Reject Mask
        if 'QUALITY_L1_PIXEL' in data_grp:
            qa_pixel = data_grp['QUALITY_L1_PIXEL'][f_idx, ...]
            bad_qa_mask = (qa_pixel & QA_REJECT_MASK) != 0
            if CLOUD_DILATION > 0:
                bad_qa_mask = ndimage.binary_dilation(bad_qa_mask, structure=kernel, iterations=CLOUD_DILATION)
            pixels_dropped_qa += np.sum(bad_qa_mask)
            valid_mask[f_idx] &= ~bad_qa_mask

        # RADSAT Accept Value
        if 'RADIOMETRIC_SATURATION' in data_grp:
            bad_radsat = data_grp['RADIOMETRIC_SATURATION'][f_idx, ...] != RADSAT_ACCEPT_VALUE
            if CLOUD_DILATION > 0:
                bad_radsat = ndimage.binary_dilation(bad_radsat, structure=kernel, iterations=CLOUD_DILATION)
            pixels_dropped_radsat += np.sum(bad_radsat)
            valid_mask[f_idx] &= ~bad_radsat

        # Aerosol Accept Values
        if 'QUALITY_L2_AEROSOL' in data_grp and AEROSOL_ACCEPT_VALUES != 'all':
            aerosol = data_grp['QUALITY_L2_AEROSOL'][f_idx, ...]
            invalid_aerosol = ~np.isin(aerosol, AEROSOL_ACCEPT_VALUES)
            if CLOUD_DILATION > 0:
                invalid_aerosol = ndimage.binary_dilation(invalid_aerosol, structure=kernel, iterations=CLOUD_DILATION)
            pixels_dropped_aerosol += np.sum(invalid_aerosol)
            valid_mask[f_idx] &= ~invalid_aerosol

    total_pixels = num_frames * height * width
    print(f"\n--- Mask Diagnostics ---")
    print(f"Frames dropped by Sun Elevation (<{SUN_ELEVATION_THRESHOLD}°): {dropped_by_sun} / {num_frames}")
    print(f"Pixels dropped by QA Mask (Clouds/Shadows): {pixels_dropped_qa / max(1, total_pixels):.1%}")
    print(f"Pixels dropped by RadSat: {pixels_dropped_radsat / max(1, total_pixels):.1%}")
    if AEROSOL_ACCEPT_VALUES != 'all':
        print(f"Pixels dropped by Aerosol: {pixels_dropped_aerosol / max(1, total_pixels):.1%}")
    else:
        print(f"Aerosol Masking: Bypassed ('all')")
    print("------------------------\n")
        
    return valid_mask

# ==========================================
# 3. PYTORCH DATASET DEFINITION
# ==========================================
class LandsatCDLDataset(Dataset):
    def __init__(self, h5_path, cdl_paths, patch_size=256, time_indices='all'):
        self.patch_size = patch_size
        
        print("Loading HDF5 Surface Reflectance into memory...")
        with h5py.File(h5_path, 'r') as f:
            data_grp = f['/HDFEOS/GRIDS/LANDSAT/Data Fields']
            sr_ds = data_grp['surface_reflectance']
            
            # --- NEW: Extract and Parse Acquisition Times ---
            acq_times = sr_ds.attrs.get('acquisition_time')
            if acq_times is None:
                raise ValueError("Missing 'acquisition_time' attribute in HDF5 dataset.")
            
            self.frame_years = []
            for dt in acq_times:
                dt_str = dt.decode('utf-8') if isinstance(dt, bytes) else str(dt)
                # Parse out the 4-digit year from string (e.g. "2024-06-15...")
                self.frame_years.append(int(dt_str[:4]))
            
            # Determine which frames to load
            total_frames = sr_ds.shape[0]
            height, width = sr_ds.shape[2], sr_ds.shape[3]
            if time_indices == 'all':
                self.t_idxs = list(range(total_frames))
            else:
                self.t_idxs = time_indices
                
            print("Generating spatial valid masks...")
            full_mask = _get_landsat_mask(data_grp, total_frames, height, width)
            self.valid_mask = full_mask[self.t_idxs, ...]
                
            # Load only the requested frames into memory
            self.images = np.nan_to_num(sr_ds[self.t_idxs, ...], nan=0.0)
            self.images = np.where(self.valid_mask[:, None, :, :], self.images, 0.0)
            
        self.num_loaded_frames, self.channels, self.height, self.width = self.images.shape
        print(f"Loaded {self.num_loaded_frames} temporal frames.")
        
        self.cdl_mapping = {
            1: 1, 5: 2, 24: 3, 36: 4, 68: 5, 69: 6, 111: 7, 
            121: 8, 122: 8, 123: 8, 124: 8, 141: 9, 142: 9, 
            143: 9, 176: 10, 190: 11, 195: 11
        }
        
        # --- NEW: Load specific CDL masks into memory by year ---
        print("Loading Aligned CDL Masks into memory...")
        self.masks = {}
        required_years = set([self.frame_years[t] for t in self.t_idxs])
        
        for year in required_years:
            if year not in cdl_paths:
                print(f"WARNING: No CDL path configured for year {year}. Frames from this year will be skipped.")
                continue
                
            cdl_path = cdl_paths[year]
            if not os.path.exists(cdl_path):
                print(f"WARNING: Aligned CDL for {year} not found at {cdl_path}. Frames from this year will be skipped.")
                continue
                
            with rasterio.open(cdl_path) as src:
                raw_mask = src.read(1)
                
            valid_cdl_pixels = np.sum(raw_mask > 0)
            if valid_cdl_pixels == 0:
                print(f"WARNING: CDL for {year} contains NO valid data (all zeroes).")
                
            # Remap the specific mask immediately upon loading
            mapped_mask = np.zeros_like(raw_mask, dtype=np.int64)
            for cdl_val, contiguous_val in self.cdl_mapping.items():
                mapped_mask[raw_mask == cdl_val] = contiguous_val
                
            self.masks[year] = mapped_mask
            print(f"Successfully loaded and mapped CDL data for {year}.")

        print("Calculating sliding window patches across all time indices...")
        self.patches = []
        stride = patch_size // 2  
        
        # Iterate over time indices AS WELL AS spatial coordinates
        for t in range(self.num_loaded_frames):
            original_t_idx = self.t_idxs[t]
            year = self.frame_years[original_t_idx]
            
            # Skip frames if we don't have a matching CDL mask for that year
            if year not in self.masks:
                continue
                
            current_mask = self.masks[year]
            
            for y in range(0, self.height - patch_size + 1, stride):
                for x in range(0, self.width - patch_size + 1, stride):
                    
                    # Temporarily check the mask for this specific patch & year
                    mask_patch = current_mask[y:y+self.patch_size, x:x+self.patch_size].copy()
                    valid_patch = self.valid_mask[t, y:y+self.patch_size, x:x+self.patch_size]
                    mask_patch[~valid_patch] = 0 # Apply cloud mask
                    
                    valid_pixel_count = np.sum(mask_patch > 0)
                    total_pixels = self.patch_size * self.patch_size
                    
                    if (valid_pixel_count / total_pixels) >= 0.01:
                        self.patches.append((t, y, x))
                
        print(f"Dataset initialized with {len(self.patches)} valid training patches (cloudy patches removed).")
        
        if len(self.patches) == 0:
            raise ValueError("CRITICAL: 0 valid training patches found! Your QA Masks filtered out 100% of the dataset, or your CDL masks are blank/missing.")
                
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        # Unpack the time index alongside spatial coordinates
        t, y, x = self.patches[idx]
        
        img_patch = self.images[t, :, y:y+self.patch_size, x:x+self.patch_size]
        
        # Grab the correct mask mapping for the year this frame was acquired
        original_t_idx = self.t_idxs[t]
        year = self.frame_years[original_t_idx]
        
        # We copy the mask slice so we don't accidentally mutate the master 2D mask
        mask_patch = self.masks[year][y:y+self.patch_size, x:x+self.patch_size].copy()
        
        # Enforce ignore_index (0) on clouded/invalid pixels for this specific timeframe
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
    
    dataset = LandsatCDLDataset(H5_TRAIN_PATH, CDL_PATHS, PATCH_SIZE, TIME_INDICES)
    
    # Set num_workers=0 to prevent Windows IPC pipe truncation with massive in-memory arrays
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    landsat_weight = ResNet50_Weights.LANDSAT_OLI_SR_MOCO 
    print(f"Building U-Net model with pre-trained weights: {landsat_weight}")

    task = SemanticSegmentationTask(
        model="unet", backbone="resnet50", weights=landsat_weight, 
        in_channels=dataset.channels, num_classes=NUM_CLASSES,
    )
    model = task.model.to(device)

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
            
            # Safety Catch for NaN loss
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