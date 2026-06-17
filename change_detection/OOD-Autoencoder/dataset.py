import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesH5Dataset(Dataset):
    def __init__(self, h5_path, dataset_name):
        """
        Loads the 3D HDF5 dataset into memory and provides 1D time-series 
        for each spatial pixel. Replaces NaNs and QA-masked values with 0.0
        and maps acquisition times to [-pi, pi] for NUFFT compatibility.
        """
        self.h5_path = h5_path
        self.dataset_name = dataset_name
        
        # Load entirely into memory. 
        with h5py.File(self.h5_path, 'r') as f:
            self.data = f[self.dataset_name][:]
            self.common_mask = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask'][:]
            acq_time = f[self.dataset_name].attrs['acquisition_time'][:]
            
        self.time_steps, self.height, self.width = self.data.shape
        self.num_pixels = self.height * self.width
        
        # Reshape to (num_pixels, time_steps)
        self.flattened_data = self.data.reshape(self.time_steps, -1).T
        
        # 1. Generate Master Validation Mask
        flat_cmask = self.common_mask.reshape(self.time_steps, -1).T
        self.invalid_mask = np.isnan(self.flattened_data) | (flat_cmask > 0)
        
        # 2. Zero-out invalid data for NUFFT
        # In Type 1 NUFFT, a coefficient of 0.0 perfectly ignores the point.
        self.flattened_data[self.invalid_mask] = 0.0
        
        self.tensor_data = torch.tensor(self.flattened_data, dtype=torch.float32)
        
        # 3. Map Acq Times to [-pi, pi]
        # NUFFT requires points in [-pi, pi]
        t_min, t_max = np.min(acq_time), np.max(acq_time)
        scaled_time = -np.pi + 2 * np.pi * (acq_time - t_min) / (t_max - t_min)
        self.points = torch.tensor(scaled_time, dtype=torch.float32)

    def __len__(self):
        return self.num_pixels

    def __getitem__(self, idx):
        """
        Returns:
            points: 1D Tensor of scaled timestamps (time_steps,)
            values: 1D Tensor of valid data (0.0 where invalid) (time_steps,)
            idx: Global flat index for batching
        """
        return self.points, self.tensor_data[idx], idx
