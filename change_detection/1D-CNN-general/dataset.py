# dataset.py
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
        
        self.h5_path = h5_path
        self.mode = mode
        self.train_end_date = train_end_date
        self.consecutive_anomalies = consecutive_anomalies
        self.time_window_years = time_window_years
        self.enable_elastic_window = enable_elastic_window
        self.max_elastic_window_years = max_elastic_window_years
        self.min_samples = min_samples
        
        # self.L_freqs removed as spatial fourier features are deprecated
        self.samples = None 
        self._init_samples()

    # _fourier_features method removed

    def _init_samples(self):
        # [Unchanged: Shared memory loading and sample identification logic remains identical]
        # ...
        pass

    def __getitem__(self, idx):
        y, x, t_idx = self.samples[idx]
        
        valid_idx = torch.where(self.valid_mask[y, x, :])[0]
        valid_acq_time = self.acq_time[valid_idx]
        
        # Targets
        target_idx = valid_idx[t_idx : t_idx + self.consecutive_anomalies]
        targets = self.z_score[y, x, target_idx]
        ts21 = valid_acq_time[t_idx].item()
        ts_last = valid_acq_time[t_idx + self.consecutive_anomalies - 1].item()
        
        # [Unchanged: Time window extraction, elastic fallback, and harmonic features]
        # ...
        
        # Combine all 11 features
        history = torch.stack([
            pixel_doy_sin, pixel_doy_cos, pixel_tod_sin, pixel_tod_cos,
            dt_sin1, dt_cos1, dt_sin2, dt_cos2, dt_sin3, dt_cos3, pixel_z
        ], dim=-1)
        
        # Pad sequence and create mask
        pad_len = MAX_SEQ_LEN - seq_len
        if pad_len > 0:
            pad_tensor = torch.zeros((pad_len, 11), dtype=torch.float32)
            history = torch.cat([pad_tensor, history], dim=0)
            seq_mask = torch.cat([torch.zeros(pad_len, dtype=torch.bool), torch.ones(seq_len, dtype=torch.bool)], dim=0)
        else:
            seq_mask = torch.ones(MAX_SEQ_LEN, dtype=torch.bool)
        
        # Spatial features calculation removed entirely
        
        metadata = [y, x, ts21, ts_last] + [t.item() for t in targets]
        return {
            'X_seq': history,
            'seq_mask': seq_mask,
            'Y_target': targets,
            'metadata': metadata
        }