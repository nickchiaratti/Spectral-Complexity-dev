import h5py
import numpy as np
from datetime import datetime, timezone
import torch
from torch.utils.data import Dataset
import math

class SITSDataset(Dataset):
    def __init__(self, h5_path, mode='calibration', train_end_date="2024-01-01"):
        """
        mode: 'calibration' (pre-train_end_date) or 'monitoring' (post-train_end_date) or 'all' (for inference context)
        """
        self.h5_path = h5_path
        self.mode = mode
        self.train_end_date = train_end_date
        
        self.L_freqs = 10 # 20 features for X (10 sin, 10 cos), 20 for Y
        
        self.samples = None # Will be a PyTorch Shared Memory Tensor
        
        self._init_samples()

    def _fourier_features(self, val, freqs=10):
        # val is normalized to [-1, 1]
        features = []
        for i in range(freqs):
            features.append(math.sin((2**i) * math.pi * val))
            features.append(math.cos((2**i) * math.pi * val))
        return features

    def _init_samples(self):
        print(f"[{self.mode}] Loading HDF5 into Shared Memory (Main Process)...")
        with h5py.File(self.h5_path, 'r') as f:
            harm_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
            
            z_score = harm_grp['sliding_volume_z_score'][:]
            np.clip(z_score, -5.0, 5.0, out=z_score)
            self.num_frames, self.h, self.w = z_score.shape
            
            # Read and transpose arrays to (h, w, frames) for CPU cache-friendly contiguous memory
            z_score = np.ascontiguousarray(np.transpose(z_score, (1, 2, 0)))
            unified_masks = harm_grp['common_mask'][:]
            valid_mask = ~unified_masks.astype(bool)
            valid_mask = np.ascontiguousarray(np.transpose(valid_mask, (1, 2, 0)))
            valid_mask &= ~np.isnan(z_score)
            
            acq_time = harm_grp['sliding_volume_z_score'].attrs['acquisition_time'][:]
            
            doy = np.array([datetime.fromtimestamp(ts, timezone.utc).timetuple().tm_yday for ts in acq_time])
            doy_sin = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
            doy_cos = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)
            
            tod_hours = np.array([
                (datetime.fromtimestamp(ts, timezone.utc).hour +
                 datetime.fromtimestamp(ts, timezone.utc).minute / 60.0 +
                 datetime.fromtimestamp(ts, timezone.utc).second / 3600.0)
                for ts in acq_time
            ], dtype=np.float32)
            tod_sin = np.sin(2 * np.pi * tod_hours / 24.0).astype(np.float32)
            tod_cos = np.cos(2 * np.pi * tod_hours / 24.0).astype(np.float32)
            
            # Place arrays into PyTorch shared memory to prevent workers from duplicating 1.5GB arrays
            self.z_score = torch.from_numpy(z_score).share_memory_()
            self.valid_mask = torch.from_numpy(valid_mask).share_memory_()
            self.acq_time = torch.from_numpy(acq_time).share_memory_()
            self.doy_sin = torch.from_numpy(doy_sin).share_memory_()
            self.doy_cos = torch.from_numpy(doy_cos).share_memory_()
            self.tod_sin = torch.from_numpy(tod_sin).share_memory_()
            self.tod_cos = torch.from_numpy(tod_cos).share_memory_()
            
            dt = datetime.strptime(self.train_end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            split_time = dt.timestamp()
            
            # Pre-allocate array for extreme speed and to bypass python object memory limits
            max_samples = self.h * self.w * self.num_frames
            samples_buf = np.zeros((max_samples, 3), dtype=np.int32)
            sample_count = 0
            
            # Identify valid windows
            for y in range(self.h):
                for x in range(self.w):
                    valid_idx = np.where(valid_mask[y, x, :])[0]
                    valid_initial_count = len(valid_idx)
                    
                    if valid_initial_count < 23:
                        continue # Insufficient valid observations
                    
                    valid_acq_time = acq_time[valid_idx]
                    
                    # Construct sliding window indices of 23 valid frames
                    for i in range(valid_initial_count - 23 + 1):
                        ts21 = valid_acq_time[i+20]
                        ts23 = valid_acq_time[i+22]
                        
                        if self.mode == 'calibration' and ts23 >= split_time:
                            continue
                        if self.mode == 'monitoring' and ts23 < split_time:
                            continue
                            
                        samples_buf[sample_count, 0] = y
                        samples_buf[sample_count, 1] = x
                        samples_buf[sample_count, 2] = i
                        sample_count += 1
                        
            print(f"[{self.mode}] Converting {sample_count} samples to Shared Memory...")
            self.samples = torch.from_numpy(samples_buf[:sample_count]).share_memory_()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        y, x, start_i = self.samples[idx]
        
        valid_idx = torch.where(self.valid_mask[y, x, :])[0]
        window_idx = valid_idx[start_i : start_i + 23]
        
        pixel_z = self.z_score[y, x, window_idx]
        pixel_doy_sin = self.doy_sin[window_idx]
        pixel_doy_cos = self.doy_cos[window_idx]
        pixel_tod_sin = self.tod_sin[window_idx]
        pixel_tod_cos = self.tod_cos[window_idx]
        valid_acq_time = self.acq_time[window_idx]
        
        ts21 = valid_acq_time[20].item()
        ts23 = valid_acq_time[22].item()
        
        # Time delta: elapsed time between current target forecast obs (ts21) 
        # and each historical frame (in days)
        delta_t = (ts21 - valid_acq_time[:20]) / 86400.0
        dt_log = torch.log(1 + delta_t).to(torch.float32)
        
        # Combine all 6 features: [doy_sin, doy_cos, tod_sin, tod_cos, dt_log, z_score]
        history = torch.stack([
            pixel_doy_sin[:20], 
            pixel_doy_cos[:20], 
            pixel_tod_sin[:20],
            pixel_tod_cos[:20],
            dt_log,
            pixel_z[:20]
        ], dim=-1)
        
        targets = pixel_z[20:23]
        
        # Spatial features
        norm_x = (x / (self.w - 1)) * 2 - 1 if self.w > 1 else 0
        norm_y = (y / (self.h - 1)) * 2 - 1 if self.h > 1 else 0
        sf_x = self._fourier_features(norm_x, self.L_freqs)
        sf_y = self._fourier_features(norm_y, self.L_freqs)
        spatial_feats = torch.tensor(sf_x + sf_y, dtype=torch.float32)
        
        return {
            'X_seq': history,
            'X_spatial': spatial_feats,
            'Y_target': targets,
            # metadata as tuple: (y, x, ts21, ts23, act1, act2, act3)
            'metadata': (y, x, ts21, ts23, targets[0].item(), targets[1].item(), targets[2].item())
        }
