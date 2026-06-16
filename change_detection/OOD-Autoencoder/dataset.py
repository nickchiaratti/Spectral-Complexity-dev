import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesH5Dataset(Dataset):
    def __init__(self, h5_path, dataset_name):
        """
        Loads the 3D HDF5 dataset into memory and provides 1D time-series 
        for each spatial pixel. Uses linear interpolation to fill NaNs 
        for FFT compatibility, and tracks which values were synthetic.
        """
        self.h5_path = h5_path
        self.dataset_name = dataset_name
        
        # Load entirely into memory. 
        # (573, 117, 147) in float32 is ~39MB, easily held in RAM, avoiding HDF5 chunking overhead.
        with h5py.File(self.h5_path, 'r') as f:
            self.data = f[self.dataset_name][:]
            
        self.time_steps, self.height, self.width = self.data.shape
        self.num_pixels = self.height * self.width
        
        # Reshape to (num_pixels, time_steps)
        # We transpose because original is (T, H, W) -> reshape to (T, H*W) -> transpose to (H*W, T)
        self.flattened_data = self.data.reshape(self.time_steps, -1).T
        
        # 1. Generate Interpolation Provenance Mask (True where data was NaN)
        self.interpolation_mask = np.isnan(self.flattened_data)
        
        # 2. Linear Interpolation using Pandas
        # axis=1 interpolates across the time steps for each pixel
        # limit_direction='both' ensures extrapolation (forward/backward fill) for bounding NaNs
        print(f"Interpolating missing data for {self.num_pixels} pixels to satisfy FFT constraints...")
        df = pd.DataFrame(self.flattened_data)
        interpolated_data = df.interpolate(axis=1, limit_direction='both').values
        
        # Ensure we use PyTorch float32
        self.tensor_data = torch.tensor(interpolated_data, dtype=torch.float32)

    def __len__(self):
        return self.num_pixels

    def __getitem__(self, idx):
        """
        Returns:
            pixel_ts: 1D Tensor of shape (time_steps,)
            h: Original row index
            w: Original col index
        """
        h = idx // self.width
        w = idx % self.width
        return self.tensor_data[idx], h, w
