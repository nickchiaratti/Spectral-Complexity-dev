import h5py
import numpy as np
import datetime
import torch
from torch.utils.data import Dataset

def extract_fractional_years(acq_times_unix):
    """Converts UNIX timestamps to absolute continuous fractional years."""
    if np.any(acq_times_unix < 0):
        raise ValueError("Corrupted UNIX timestamps detected. Halting execution.")

    frac_years = []
    for dt in acq_times_unix:
        dt_obj = datetime.datetime.fromtimestamp(float(dt), tz=datetime.timezone.utc)
        year = dt_obj.year
        start_of_year = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
        start_of_next = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
        year_duration = (start_of_next - start_of_year).total_seconds()
        elapsed = (dt_obj - start_of_year).total_seconds()
        frac_years.append(year + (elapsed / year_duration))
    return np.array(frac_years, dtype=np.float64)

def load_dataset(h5_path, target_metric='sliding_volume_z_score'):
    """Loads the full HDF5 dataset into memory. Validates data integrity."""
    print(f"Loading data from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        metric_ds = data_grp[target_metric]

        acq_times = metric_ds.attrs['acquisition_time'][:]
        y_data = metric_ds[...]
        common_mask = data_grp['common_mask'][...]

        geo_transform = metric_ds.attrs.get('GeoTransform')
        spatial_ref = metric_ds.attrs.get('spatial_ref')

    # Quality mask: 1 = bad per dataset spec. Invert for True = valid.
    valid_mask = (common_mask == 0)

    # Incorporate dataset NaN values into the validity mask
    # This prevents unflagged NaN values from bypassing the quality mask
    unmasked_nans = np.isnan(y_data) & valid_mask
    if np.any(unmasked_nans):
        valid_mask[unmasked_nans] = False
        print(f"Warning: Incorporated {np.sum(unmasked_nans)} unflagged NaN values into the valid_mask.")

    # Sort chronologically
    sort_idx = np.argsort(acq_times)
    acq_times = acq_times[sort_idx]
    y_data = y_data[sort_idx, ...]
    valid_mask = valid_mask[sort_idx, ...]

    frac_years = extract_fractional_years(acq_times)

    num_frames, height, width = y_data.shape
    print(f"Dataset: {num_frames} frames, {height}x{width} pixels")
    print(f"Temporal range: {frac_years[0]:.2f} - {frac_years[-1]:.2f}")

    return y_data, valid_mask, acq_times, frac_years, geo_transform, spatial_ref

class ALFTSequenceDataset(Dataset):
    """
    Assembles L_MAX-length sequences from pre-computed ALFT features
    for Deep SVDD one-class training.

    Each sample is a (pixel, target_timestep) pair. The sequence consists
    of the L_MAX most recent ALFT feature vectors ending at the target
    timestep, left-padded with NaN if the sequence is shorter than L_MAX.
    """
    def __init__(self, alft_features, alft_valid, frac_years,
                 train_end_frac_year, l_max, alft_dim, stride=1):
        """
        Args:
            alft_features: (num_frames, H, W, ALFT_DIM) numpy float32
            alft_valid: (num_frames, H, W) numpy bool
            frac_years: (num_frames,) numpy float64
            train_end_frac_year: float - temporal cutoff (exclusive)
            l_max: int - maximum sequence length
            alft_dim: int - feature dimension
            stride: int - subsample target timesteps (every Nth)
        """
        self.alft_features = alft_features
        self.alft_valid = alft_valid
        self.frac_years = frac_years.astype(np.float32)
        self.l_max = l_max
        self.alft_dim = alft_dim

        num_frames = alft_valid.shape[0]

        # Training constraint: only pre-change timesteps
        train_mask = frac_years < train_end_frac_year
        train_indices = np.where(train_mask)[0]
        train_indices = train_indices[::stride]

        # Build index: (y, x, t) where target token is valid
        self.samples = []
        for t in train_indices:
            valid_yx = np.argwhere(alft_valid[t])  # (N, 2) of [y, x]
            for yx in valid_yx:
                self.samples.append((yx[0], yx[1], t))

        print(f"Training dataset: {len(self.samples):,} sequences "
              f"(stride={stride}, cutoff={train_end_frac_year:.1f})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        y, x, t = self.samples[idx]

        # Sequence ending at timestep t
        seq_start = max(0, t - self.l_max + 1)
        actual_len = t - seq_start + 1
        pad_len = self.l_max - actual_len

        features = self.alft_features[seq_start:t + 1, y, x, :]  # (actual_len, ALFT_DIM)
        valid = self.alft_valid[seq_start:t + 1, y, x]            # (actual_len,)
        times = self.frac_years[seq_start:t + 1]                   # (actual_len,)

        # Left-pad to L_MAX with NaN features (approved fill value for padding)
        if pad_len > 0:
            features = np.concatenate([
                np.full((pad_len, self.alft_dim), np.nan, dtype=np.float32),
                features
            ], axis=0)
            valid = np.concatenate([
                np.zeros(pad_len, dtype=bool),
                valid
            ], axis=0)
            times = np.concatenate([
                np.zeros(pad_len, dtype=np.float32),
                times
            ], axis=0)

        # Padding mask: True = ignore (padded OR invalid ALFT token)
        padding_mask = ~valid

        return {
            'features': torch.from_numpy(features),             # (L_MAX, ALFT_DIM)
            'times': torch.from_numpy(times).unsqueeze(-1),      # (L_MAX, 1)
            'padding_mask': torch.from_numpy(padding_mask),      # (L_MAX,)
        }
