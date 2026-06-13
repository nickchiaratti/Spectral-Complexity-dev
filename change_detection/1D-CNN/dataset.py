import h5py
import numpy as np
from datetime import datetime, timezone
import torch
from torch.utils.data import Dataset
import math

class SITSDataset(Dataset):
    def __init__(self, h5_path, mode='calibration', train_end_date="2024-01-01", 
                 consecutive_anomalies=3, time_window_years=3.0, 
                 enable_elastic_window=True, max_elastic_window_years=5.0, 
                 min_samples=38):
        """
        mode: 'calibration' (pre-train_end_date) or 'monitoring' (post-train_end_date) or 'all' (for inference context)
        """
        self.h5_path = h5_path
        self.mode = mode
        self.train_end_date = train_end_date
        self.consecutive_anomalies = consecutive_anomalies
        self.time_window_years = time_window_years
        self.enable_elastic_window = enable_elastic_window
        self.max_elastic_window_years = max_elastic_window_years
        self.min_samples = min_samples
        
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
                    
                    if valid_initial_count < self.min_samples + self.consecutive_anomalies:
                        continue # Insufficient valid observations
                    
                    valid_acq_time = acq_time[valid_idx]
                    
                    # Target indices start after MIN_SAMPLES history
                    for t_idx in range(self.min_samples, valid_initial_count - self.consecutive_anomalies + 1):
                        ts21 = valid_acq_time[t_idx]
                        ts_last = valid_acq_time[t_idx + self.consecutive_anomalies - 1]
                        
                        if self.mode == 'calibration' and ts_last >= split_time:
                            continue
                        if self.mode == 'monitoring' and ts_last < split_time:
                            continue
                            
                        # Check window validity
                        past_times = valid_acq_time[:t_idx]
                        window_start = ts21 - (self.time_window_years * 365.25 * 86400.0)
                        in_window_count = np.sum(past_times >= window_start)
                        
                        if in_window_count < self.min_samples:
                            if not self.enable_elastic_window:
                                continue
                            
                            elastic_start = ts21 - (self.max_elastic_window_years * 365.25 * 86400.0)
                            in_elastic_count = np.sum(past_times >= elastic_start)
                            if in_elastic_count < self.min_samples:
                                continue
                                
                        samples_buf[sample_count, 0] = y
                        samples_buf[sample_count, 1] = x
                        samples_buf[sample_count, 2] = t_idx
                        sample_count += 1
                        
            print(f"[{self.mode}] Converting {sample_count} samples to Shared Memory...")
            self.samples = torch.from_numpy(samples_buf[:sample_count]).share_memory_()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        y, x, t_idx = self.samples[idx]
        
        valid_idx = torch.where(self.valid_mask[y, x, :])[0]
        valid_acq_time = self.acq_time[valid_idx]
        
        # Targets
        target_idx = valid_idx[t_idx : t_idx + self.consecutive_anomalies]
        targets = self.z_score[y, x, target_idx]
        ts21 = valid_acq_time[t_idx].item()
        ts_last = valid_acq_time[t_idx + self.consecutive_anomalies - 1].item()
        
        # History (Temporal Subset + Elastic Fallback)
        TIME_WINDOW_SEC = self.time_window_years * 365.25 * 86400.0
        MAX_SEQ_LEN = 350
        
        past_idx = valid_idx[:t_idx]
        past_times = valid_acq_time[:t_idx]
        
        window_start_time = ts21 - TIME_WINDOW_SEC
        in_window_mask = past_times >= window_start_time
        
        if torch.sum(in_window_mask) < self.min_samples and self.enable_elastic_window:
            elastic_start_time = ts21 - (self.max_elastic_window_years * 365.25 * 86400.0)
            in_elastic_mask = past_times >= elastic_start_time
            history_idx = past_idx[in_elastic_mask][-self.min_samples:]
            history_times = past_times[in_elastic_mask][-self.min_samples:]
        else:
            history_idx = past_idx[in_window_mask]
            history_times = past_times[in_window_mask]
            
        if len(history_idx) > MAX_SEQ_LEN:
            history_idx = history_idx[-MAX_SEQ_LEN:]
            history_times = history_times[-MAX_SEQ_LEN:]
            
        seq_len = len(history_idx)
        
        pixel_z = self.z_score[y, x, history_idx]
        pixel_doy_sin = self.doy_sin[history_idx]
        pixel_doy_cos = self.doy_cos[history_idx]
        pixel_tod_sin = self.tod_sin[history_idx]
        pixel_tod_cos = self.tod_cos[history_idx]
        
        # Time delta: elapsed time between current target forecast obs (ts21) 
        # and each historical frame (in days)
        delta_t = (ts21 - history_times) / 86400.0
        
        # Harmonic Features (1st, 2nd, 3rd)
        dt_years = delta_t / 365.25
        dt_sin1 = torch.sin(1 * 2 * math.pi * dt_years).to(torch.float32)
        dt_cos1 = torch.cos(1 * 2 * math.pi * dt_years).to(torch.float32)
        dt_sin2 = torch.sin(2 * 2 * math.pi * dt_years).to(torch.float32)
        dt_cos2 = torch.cos(2 * 2 * math.pi * dt_years).to(torch.float32)
        dt_sin3 = torch.sin(3 * 2 * math.pi * dt_years).to(torch.float32)
        dt_cos3 = torch.cos(3 * 2 * math.pi * dt_years).to(torch.float32)
        
        # Combine all 11 features
        history = torch.stack([
            pixel_doy_sin, 
            pixel_doy_cos, 
            pixel_tod_sin,
            pixel_tod_cos,
            dt_sin1,
            dt_cos1,
            dt_sin2,
            dt_cos2,
            dt_sin3,
            dt_cos3,
            pixel_z
        ], dim=-1)
        
        # Pad sequence and create mask
        pad_len = MAX_SEQ_LEN - seq_len
        if pad_len > 0:
            pad_tensor = torch.zeros((pad_len, 11), dtype=torch.float32)
            history = torch.cat([pad_tensor, history], dim=0)
            seq_mask = torch.cat([torch.zeros(pad_len, dtype=torch.bool), torch.ones(seq_len, dtype=torch.bool)], dim=0)
        else:
            seq_mask = torch.ones(MAX_SEQ_LEN, dtype=torch.bool)
        
        # Spatial features
        norm_x = (x / (self.w - 1)) * 2 - 1 if self.w > 1 else 0
        norm_y = (y / (self.h - 1)) * 2 - 1 if self.h > 1 else 0
        sf_x = self._fourier_features(norm_x, self.L_freqs)
        sf_y = self._fourier_features(norm_y, self.L_freqs)
        spatial_feats = torch.tensor(sf_x + sf_y, dtype=torch.float32)
        
        metadata = [y, x, ts21, ts_last] + [t.item() for t in targets]
        return {
            'X_seq': history,
            'X_spatial': spatial_feats,
            'seq_mask': seq_mask,
            'Y_target': targets,
            'metadata': metadata
        }
