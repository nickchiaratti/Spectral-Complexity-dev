import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import h5py
import rasterio
from torchgeo.trainers import SemanticSegmentationTask
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# Source HDF5 containing both surface_reflectance and sliding_volume_map
H5_SOURCE_PATH = "C:/satelliteImagery/LANDSAT/Rochester/LANDSAT_Stack_Rochester_HDFEOS_SC_EM-7_Gram-datasetMean_Norm-bandCount_Aerosol-low_QA-AllFrames_sunElMin-40.h5"

# The new HDF5 file where the outputs will be saved
H5_OUTPUT_PATH = "C:/satelliteImagery/LANDSAT/Rochester/LANDSAT_Stack_Rochester_EVALUATED.h5"

# Ground Truth path to dynamically determine the number of classes
ALIGNED_CDL_PATH = "C:/satelliteImagery/LANDSAT/Rochester/CDL2024_Aligned_Rochester.tif"

# Dictionary of models to evaluate. Keys become the dataset names in the HDF5 file.
MODELS_TO_EVALUATE = {
    "predicted_cdl_7band": "C:/satelliteImagery/LANDSAT/Rochester/rochester_cdl_7band_baseline.pth",
    "predicted_cdl_8band": "C:/satelliteImagery/LANDSAT/Rochester/rochester_cdl_8band_surgery.pth"
}

# Inference Hyperparameters
PATCH_SIZE = 256

# ==========================================
# 2. MODEL LOADING & ARCHITECTURE
# ==========================================
def load_model(device, model_path, num_classes):
    """Reconstructs the TorchGeo U-Net and dynamically adapts to the saved weights."""
    print(f"\nLoading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    # Automatically determine input channels from the saved weights
    input_channels = state_dict['encoder.conv1.weight'].shape[1]
    print(f"Detected {input_channels} input channels in saved weights.")

    task = SemanticSegmentationTask(
        model="unet", backbone="resnet50", weights=None,
        in_channels=7, num_classes=num_classes
    )
    model = task.model
    
    # Replicate the weight surgery if evaluating an 8+ band model
    if input_channels > 7:
        old_conv1 = model.encoder.conv1
        new_conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=old_conv1.out_channels, 
            kernel_size=old_conv1.kernel_size, stride=old_conv1.stride, 
            padding=old_conv1.padding, bias=False
        )
        model.encoder.conv1 = new_conv1

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, input_channels

# ==========================================
# 3. INFERENCE LOOP
# ==========================================
def process_model_inference(h5_path, dataset_name, model, device, input_channels):
    """Runs inference across all frames and saves directly to the HDF5 file."""
    
    with h5py.File(h5_path, 'r+') as f:
        data_grp = f['/HDFEOS/GRIDS/LANDSAT/Data Fields']
        sr_ds = data_grp['surface_reflectance']
        
        num_frames, _, height, width = sr_ds.shape
        
        # Create or overwrite the output dataset
        if dataset_name in data_grp:
            print(f"Dataset '{dataset_name}' already exists. Overwriting...")
            del data_grp[dataset_name]
            
        out_ds = data_grp.create_dataset(
            dataset_name, 
            shape=(num_frames, height, width), 
            dtype='uint8', 
            compression='gzip'
        )
        
        out_ds.attrs['description'] = f"Model Predictions ({input_channels}-channel input)"
        out_ds.attrs['num_classes'] = model.segmentation_head[0].out_channels
        
        # Optional: Copy spatial metadata from surface_reflectance for GIS compatibility
        if 'spatial_ref' in sr_ds.attrs:
            out_ds.attrs['spatial_ref'] = sr_ds.attrs['spatial_ref']
        if 'GeoTransform' in sr_ds.attrs:
            out_ds.attrs['GeoTransform'] = sr_ds.attrs['GeoTransform']

        print(f"Running inference for {num_frames} frames...")
        
        with torch.no_grad():
            for t in tqdm(range(num_frames), desc=f"Evaluating {dataset_name}"):
                
                # 1. Load frame data
                sr_data = sr_ds[t, ...]
                sr_data = np.nan_to_num(sr_data, nan=0.0)
                
                if input_channels > 7:
                    sc_data = data_grp['sliding_volume_map'][t, ...]
                    sc_data = np.nan_to_num(sc_data, nan=0.0)
                    sc_data = np.expand_dims(sc_data, axis=0) # Shape: [1, H, W]
                    full_image = np.concatenate((sr_data, sc_data), axis=0) # Shape: [8, H, W]
                else:
                    full_image = sr_data
                    
                # 2. Sliding window prediction
                predicted_map = np.zeros((height, width), dtype=np.uint8)
                stride = PATCH_SIZE
                
                for y in range(0, height, stride):
                    for x in range(0, width, stride):
                        y_end = min(y + PATCH_SIZE, height)
                        x_end = min(x + PATCH_SIZE, width)
                        
                        patch = full_image[:, y:y_end, x:x_end]
                        
                        # Pad edges if necessary
                        pad_y = PATCH_SIZE - patch.shape[1]
                        pad_x = PATCH_SIZE - patch.shape[2]
                        if pad_y > 0 or pad_x > 0:
                            patch = np.pad(patch, ((0,0), (0, pad_y), (0, pad_x)), mode='constant')
                        
                        tensor_patch = torch.from_numpy(patch).float().unsqueeze(0).to(device)
                        logits = model(tensor_patch)
                        pred_classes = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
                        
                        predicted_map[y:y_end, x:x_end] = pred_classes[:PATCH_SIZE-pad_y, :PATCH_SIZE-pad_x]
                        
                # 3. Save map to the HDF5 array
                out_ds[t, ...] = predicted_map

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")
    
    # 1. Detect number of classes dynamically
    if not os.path.exists(ALIGNED_CDL_PATH):
        raise FileNotFoundError(f"Missing Aligned CDL path: {ALIGNED_CDL_PATH}. Needed to detect class counts.")
        
    print(f"Loading Ground Truth from {ALIGNED_CDL_PATH} to detect class parameters...")
    with rasterio.open(ALIGNED_CDL_PATH) as src:
        true_mask = src.read(1)
    num_classes = int(np.max(true_mask)) + 1
    print(f"Dynamically detected {num_classes} total classes (including background).")

    # 2. Make a safe copy of the HDF5 file
    if not os.path.exists(H5_OUTPUT_PATH):
        print(f"\nDuplicating HDF5 source to {H5_OUTPUT_PATH} (This may take a moment)...")
        shutil.copy2(H5_SOURCE_PATH, H5_OUTPUT_PATH)
    else:
        print(f"\nTarget HDF5 file {H5_OUTPUT_PATH} already exists. Appending to it.")

    # 3. Evaluate models
    for dataset_name, model_path in MODELS_TO_EVALUATE.items():
        if not os.path.exists(model_path):
            print(f"\n[SKIPPING] Model weights not found at {model_path}")
            continue
            
        # Load and configure the model for this specific path
        model, input_channels = load_model(device, model_path, num_classes)
        
        # Run predictions and inject directly into HDF5
        process_model_inference(H5_OUTPUT_PATH, dataset_name, model, device, input_channels)
        
        # Free up GPU memory between models
        del model
        torch.cuda.empty_cache()

    print("\nEvaluation Complete! All predictions have been saved to the evaluated HDF5 stack.")

if __name__ == "__main__":
    main()