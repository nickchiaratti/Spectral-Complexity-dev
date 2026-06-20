"""
Latent Neural ODE for Spatiotemporal Regime Shift Detection
============================================================
Detects regime shifts in spectral complexity time series using two
complementary methods with separate flags for explainability:

    A) Reconstruction-error anomaly detection
       Flags individual observations that deviate from the learned
       continuous-time dynamics. Sensitive to abrupt changes.

    B) Latent-space drift detection via CUSUM
       Monitors the Mahalanobis distance of the latent trajectory z(t)
       from its empirical distribution. Sensitive to gradual regime shifts.

The model is trained as a full-sequence autoencoder: encode the observed
time series into a latent initial condition z(t0), solve an autonomous ODE
dz/dt = f(z), and decode back to observations. Regime shifts manifest as
regions where the smooth ODE trajectory cannot explain the observations.

References:
    - Chen et al. (2018). "Neural Ordinary Differential Equations." NeurIPS.
    - Rubanova et al. (2019). "Latent ODEs for Irregularly-Sampled Time Series." NeurIPS.
    - Page (1954). "Continuous Inspection Schemes." Biometrika.  (CUSUM)
    - Belkin & Niyogi (2006). "Manifold Regularization." JMLR.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import logging
import warnings
from tqdm import tqdm
from torchdiffeq import odeint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================
# 1. CONFIGURATION
# ============================================================
LOCATION = "Tait"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
TARGET_METRIC = 'sliding_volume_z_score'
OUTPUT_DIR = "C:/satelliteImagery/HLST30/NeuralODE"

# --- Training ---
PATCH_SIZE = 32
LATENT_DIM = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
SAMPLES_PER_EPOCH = 500
GRAD_CLIP_NORM = 1.0
TRAIN_TIME_SUBSAMPLE = 64   # Timesteps sampled per training step
PATIENCE = 15                # Early stopping patience
LAMBDA_SPATIAL = 0.1         # Spatial Laplacian regularization weight
RANDOM_SEED = 42

# --- Reconstruction Anomaly Detection (Method A) ---
RECON_THRESHOLD_SIGMA = 3.0
CONSECUTIVE_ANOMALIES_RECON = 4

# --- Latent Drift Detection / CUSUM (Method B) ---
CUSUM_ALLOWANCE = 0.5       # k: half the minimum shift to detect (in sigma)
CUSUM_THRESHOLD = 5.0       # h: decision boundary
CONSECUTIVE_ANOMALIES_DRIFT = 4

# --- Output ---
SAVE_LATENT_TRAJECTORY = True  # ~630 MB for (573, 117, 147, 16); disable to save disk


# ============================================================
# 2. DATA LOADING
# ============================================================
class HDF5SpatiotemporalDataset(torch.utils.data.Dataset):
    """
    Loads spatial patches from the harmonized HDF5 spectral complexity
    dataset, preserving data integrity through explicit masking.

    HDF5 Schema (Tait):
        /HDFEOS/GRIDS/HARMONIZED/Data Fields/
            sliding_volume_z_score  (T, H, W) float32
            common_mask             (T, H, W) uint8, 1 = invalid
        Attributes on metric dataset:
            acquisition_time        (T,) float64  UNIX epoch seconds
            GeoTransform            (6,) float64
            spatial_ref             str (WKT)
    """

    def __init__(self, h5_filepath, target_metric, patch_size, samples_per_epoch):
        super().__init__()
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch

        assert os.path.exists(h5_filepath), \
            f"HDF5 file not found: {h5_filepath}"

        with h5py.File(h5_filepath, 'r') as f:
            grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
            metric_ds = grp[target_metric]

            raw_data = metric_ds[:]            # (T, H, W) float32
            common_mask = grp['common_mask'][:]  # (T, H, W) uint8
            acq_times = metric_ds.attrs['acquisition_time'][:]  # (T,) float64

            self.geo_transform = metric_ds.attrs['GeoTransform']
            self.spatial_ref = metric_ds.attrs['spatial_ref']

        # Sort chronologically
        sort_idx = np.argsort(acq_times)
        raw_data = raw_data[sort_idx]
        common_mask = common_mask[sort_idx]
        self.acq_times = acq_times[sort_idx]

        self.time_len, self.height, self.width = raw_data.shape
        assert self.height >= patch_size and self.width >= patch_size, \
            (f"Spatial dimensions ({self.height}, {self.width}) are smaller "
             f"than patch_size ({patch_size})")

        # Construct validity mask: valid = not-NaN AND not-masked
        nan_mask = np.isnan(raw_data)
        self.valid_mask = (~nan_mask) & (common_mask == 0)

        invalid_count = int(np.count_nonzero(~self.valid_mask))
        total_count = int(self.valid_mask.size)
        valid_pct = 100.0 * (1.0 - invalid_count / total_count)
        logger.info(
            f"Validity: {total_count - invalid_count:,}/{total_count:,} "
            f"observations ({valid_pct:.1f}%)"
        )

        # Replace invalid values with 0.0 — neutral for GRU input gating.
        # The mask is already frozen above; this is a tensor preparation step,
        # NOT a fill-value injection. The loss and anomaly detection always
        # reference self.valid_mask to exclude these positions.
        raw_data[~self.valid_mask] = 0.0
        self.data = raw_data

        # Normalize timestamps to [0, 1] for ODE numerical stability
        t_min, t_max = float(self.acq_times.min()), float(self.acq_times.max())
        self.norm_timestamps = (
            (self.acq_times - t_min) / (t_max - t_min)
        ).astype(np.float32)
        logger.info(
            f"Timestamps normalized: [{t_min:.0f}, {t_max:.0f}] UNIX → [0.0, 1.0]"
        )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        y = np.random.randint(0, self.height - self.patch_size + 1)
        x = np.random.randint(0, self.width - self.patch_size + 1)

        patch_data = self.data[:, y:y + self.patch_size, x:x + self.patch_size]
        patch_mask = self.valid_mask[:, y:y + self.patch_size, x:x + self.patch_size]

        # Shape: (T, 1, patch_H, patch_W)
        patch_data = torch.from_numpy(patch_data).unsqueeze(1).float()
        patch_mask = torch.from_numpy(patch_mask).unsqueeze(1).bool()
        timestamps = torch.from_numpy(self.norm_timestamps).float()

        return timestamps, patch_data, patch_mask


# ============================================================
# 3. MODEL ARCHITECTURE
# ============================================================
class ODEFunc(nn.Module):
    """
    Autonomous continuous-time dynamics: dz/dt = f(z).

    Tanh activations are used because:
        1. ODE derivatives must be Lipschitz continuous for existence/uniqueness.
        2. ReLU's non-differentiability at zero violates this requirement,
           causing adaptive ODE solvers to waste steps near the discontinuity.

    Ref: Chen et al. (2018), §3.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, t, z):
        # t is accepted by the odeint interface but unused (autonomous dynamics)
        return self.net(z)


class MaskedEncoderRNN(nn.Module):
    """
    Backward GRU encoder with observation-level masking.

    For irregularly observed time series, standard GRU processing corrupts
    the hidden state at timesteps with no valid data. This encoder holds
    the hidden state constant when the observation mask is entirely False,
    ensuring that only real observations contribute to z(t0).
    """

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.gru_cell = nn.GRUCell(input_dim, latent_dim)
        self.linear_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x_seq, valid_mask_seq):
        """
        Args:
            x_seq:          (T, N, input_dim)  — observation values
            valid_mask_seq: (T, N, input_dim)  — boolean validity mask

        Returns:
            z_t0: (N, latent_dim) — encoded initial latent state
        """
        T, N, _ = x_seq.shape
        device = x_seq.device

        h = torch.zeros(N, self.latent_dim, device=device)

        # Process backward: encode from t=T-1 down to t=0
        for t in range(T - 1, -1, -1):
            x_t = x_seq[t]                         # (N, input_dim)
            m_t = valid_mask_seq[t]                 # (N, input_dim) bool

            # Any valid feature at this pixel → update hidden state
            any_valid = m_t.any(dim=-1)             # (N,) bool

            # Zero out invalid inputs so they contribute no signal
            x_t_masked = x_t * m_t.float()

            h_new = self.gru_cell(x_t_masked, h)

            # Only update hidden state at pixels with valid observations
            h = torch.where(any_valid.unsqueeze(-1), h_new, h)

        z_t0 = self.linear_out(h)
        return z_t0


class SpatiotemporalLatentODE(nn.Module):
    """
    Encode → ODE-solve → Decode pipeline for spatiotemporal data.

    The spatial structure is flattened into the batch dimension for the
    temporal encoder and ODE solver (which are per-pixel operations),
    then restored for the spatial regularization loss.
    """

    def __init__(self, input_dim=1, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MaskedEncoderRNN(input_dim, latent_dim)
        self.ode_func = ODEFunc(latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, timestamps, data_seq, mask_seq,
                return_latent_trajectory=False):
        """
        Args:
            timestamps: (T,)
            data_seq:   (T, B, C, H, W)
            mask_seq:   (T, B, C, H, W)

        Returns:
            pred_x_spatial: (T, B, C, H, W)
            spatial_z0:     (B, latent_dim, H, W)
            latent_traj:    (T, B, H, W, latent_dim) — only if requested
        """
        T, B, C, H, W = data_seq.shape

        # Flatten spatial → batch for per-pixel temporal processing
        flat_data = data_seq.reshape(T, B * H * W, C)
        flat_mask = mask_seq.reshape(T, B * H * W, C)

        # Encode to initial latent state
        z_t0 = self.encoder(flat_data, flat_mask)   # (B*H*W, latent_dim)

        # Spatial z0 for regularization loss
        spatial_z0 = z_t0.view(B, H, W, self.latent_dim).permute(0, 3, 1, 2)

        # Solve ODE forward: adaptive Runge-Kutta 4(5)
        pred_z = odeint(
            self.ode_func, z_t0, timestamps, method='dopri5'
        )  # (T, B*H*W, latent_dim)

        # Decode back to observation space
        pred_x = self.decoder(pred_z)               # (T, B*H*W, C)
        pred_x_spatial = pred_x.reshape(T, B, H, W, C).permute(0, 1, 4, 2, 3)

        if return_latent_trajectory:
            latent_traj = pred_z.reshape(T, B, H, W, self.latent_dim)
            return pred_x_spatial, spatial_z0, latent_traj

        return pred_x_spatial, spatial_z0


# ============================================================
# 4. LOSS FUNCTION
# ============================================================
def compute_spatiotemporal_loss(pred_x, target_x, mask, spatial_z0,
                                lambda_spatial=0.1):
    """
    Masked temporal reconstruction + latent spatial Laplacian regularization.

    The spatial regularizer penalizes differences between each pixel's
    latent state and its 4-connected neighbors, enforcing that spatially
    adjacent pixels share similar dynamics (manifold prior).

    Ref: Belkin & Niyogi (2006), §2.
    """
    assert mask.any(), \
        "CRITICAL: Received batch with zero valid samples. Upstream pipeline failure."

    # --- Temporal Reconstruction Loss (masked Smooth L1) ---
    unreduced = F.smooth_l1_loss(pred_x, target_x, reduction='none')
    masked_loss = unreduced.masked_fill(~mask, 0.0)
    temporal_loss = masked_loss.sum() / mask.sum()

    # --- Spatial Laplacian Regularization (4-connected graph) ---
    # spatial_z0: (B, D, H, W)
    padded = F.pad(spatial_z0, (1, 1, 1, 1), mode='replicate')
    neighbors = [
        padded[..., :-2, 1:-1],    # up
        padded[..., 2:,  1:-1],    # down
        padded[..., 1:-1, :-2],    # left
        padded[..., 1:-1, 2:],     # right
    ]
    spatial_loss = sum(
        F.mse_loss(spatial_z0, n) for n in neighbors
    ) / 4.0

    return temporal_loss + lambda_spatial * spatial_loss


# ============================================================
# 5. TRAINING
# ============================================================
def train_epoch(model, dataloader, optimizer, device, time_subsample):
    """One epoch of training with timestep subsampling for efficiency."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for timestamps, patch_data, patch_mask in dataloader:
        timestamps = timestamps[0].to(device)       # (T,) — shared across batch
        # Permute: (B, T, C, H, W) → (T, B, C, H, W)
        patch_data = patch_data.to(device).permute(1, 0, 2, 3, 4)
        patch_mask = patch_mask.to(device).permute(1, 0, 2, 3, 4)

        T_full = timestamps.shape[0]

        # Subsample timesteps: random sorted subset for ODE integration
        if time_subsample and time_subsample < T_full:
            idx = torch.sort(
                torch.randperm(T_full, device=device)[:time_subsample]
            ).values
            timestamps = timestamps[idx]
            patch_data = patch_data[idx]
            patch_mask = patch_mask[idx]

        # Skip degenerate batches
        if not patch_mask.any():
            logger.warning("Skipping batch: no valid observations in patch")
            continue

        optimizer.zero_grad()

        pred_x, spatial_z0 = model(timestamps, patch_data, patch_mask)
        loss = compute_spatiotemporal_loss(
            pred_x, patch_data, patch_mask, spatial_z0, LAMBDA_SPATIAL
        )

        # Check for training instability
        assert torch.isfinite(loss), \
            f"CRITICAL: Non-finite loss ({loss.item()}) — ODE solver instability"

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# 6. FULL-SCENE INFERENCE
# ============================================================
@torch.no_grad()
def run_full_inference(model, dataset, device, tile_size=32):
    """
    Run the trained model over the full spatial grid in non-overlapping tiles.

    Returns:
        predictions:    (T, H, W) float32
        latent_traj:    (T, H, W, D) float32
        z0:             (H, W, D) float32
    """
    model.eval()

    T = dataset.time_len
    H, W = dataset.height, dataset.width
    D = model.latent_dim

    timestamps = torch.from_numpy(dataset.norm_timestamps).float().to(device)

    # Allocate output arrays on CPU
    all_pred = np.full((T, H, W), np.nan, dtype=np.float32)
    all_latent = np.full((T, H, W, D), np.nan, dtype=np.float32)
    all_z0 = np.full((H, W, D), np.nan, dtype=np.float32)

    # Tile the spatial grid
    y_starts = list(range(0, H, tile_size))
    x_starts = list(range(0, W, tile_size))
    total_tiles = len(y_starts) * len(x_starts)

    logger.info(
        f"Inference: {total_tiles} tiles of ≤{tile_size}×{tile_size} "
        f"over {H}×{W} grid"
    )

    pbar = tqdm(total=total_tiles, desc="Inference")

    for ys in y_starts:
        ye = min(ys + tile_size, H)
        for xs in x_starts:
            xe = min(xs + tile_size, W)

            tile_data = dataset.data[:, ys:ye, xs:xe]       # (T, tH, tW)
            tile_mask = dataset.valid_mask[:, ys:ye, xs:xe]  # (T, tH, tW)

            # Model expects (T, B=1, C=1, tH, tW)
            td = (torch.from_numpy(tile_data).float()
                  .unsqueeze(1).unsqueeze(1).to(device))
            tm = (torch.from_numpy(tile_mask).bool()
                  .unsqueeze(1).unsqueeze(1).to(device))

            pred_x, spatial_z0, latent_traj = model(
                timestamps, td, tm, return_latent_trajectory=True
            )

            # Write to output arrays
            all_pred[:, ys:ye, xs:xe] = pred_x[:, 0, 0].cpu().numpy()
            all_latent[:, ys:ye, xs:xe, :] = latent_traj[:, 0].cpu().numpy()
            all_z0[ys:ye, xs:xe, :] = (
                spatial_z0[0].permute(1, 2, 0).cpu().numpy()
            )

            pbar.update(1)

    pbar.close()

    # Verify integrity
    nan_pred = int(np.isnan(all_pred).sum())
    assert nan_pred == 0, \
        f"CRITICAL: {nan_pred} NaN values in predictions — ODE solver instability"

    return all_pred, all_latent, all_z0


# ============================================================
# 7. ANOMALY DETECTION — METHOD A: RECONSTRUCTION ERROR
# ============================================================
def detect_reconstruction_anomalies(actual, predicted, valid_mask, acq_times,
                                     threshold_sigma, consecutive_required):
    """
    Flag timesteps where |actual − predicted| exceeds a MAD-based robust
    threshold. This detects abrupt deviations from the learned ODE dynamics.

    Args:
        actual:       (T, H, W)  — observed values (0.0 at invalid positions)
        predicted:    (T, H, W)  — ODE-decoded predictions
        valid_mask:   (T, H, W)  — boolean validity mask
        acq_times:    (T,)       — UNIX timestamps for change dating
        threshold_sigma: float   — multiplier on robust sigma
        consecutive_required: int — consecutive anomalies to trigger regime shift

    Returns:
        flags:          (T, H, W) uint8
        change_date:    (H, W)    float64 — UNIX timestamp of first shift
        change_count:   (H, W)    int32   — number of shift triggers
        recon_error:    (T, H, W) float32 — raw reconstruction error (NaN if invalid)
        sigma:          (H, W)    float32 — per-pixel robust sigma
    """
    T, H, W = actual.shape
    valid = valid_mask.astype(bool)

    # Reconstruction error only at valid observations
    recon_error = np.full((T, H, W), np.nan, dtype=np.float32)
    recon_error[valid] = np.abs(actual[valid] - predicted[valid])

    # Per-pixel robust sigma via MAD (Median Absolute Deviation)
    # Pixels with zero valid observations produce all-NaN slices;
    # nanmedian correctly returns NaN for these (no flags will be set).
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice', RuntimeWarning)
        med = np.nanmedian(recon_error, axis=0, keepdims=True) # (1, H, W)
        mad = np.nanmedian(np.abs(recon_error - med), axis=0)  # (H, W)
    sigma = np.clip(1.4826 * mad, a_min=1e-6, a_max=None)     # MAD → σ

    logger.info(
        f"Reconstruction σ: median={np.nanmedian(sigma):.4f}, "
        f"range=[{np.nanmin(sigma):.4f}, {np.nanmax(sigma):.4f}]"
    )

    # Per-timestep anomaly flags
    threshold = threshold_sigma * sigma[None, :, :]  # (1, H, W) broadcast
    is_anomaly = (recon_error > threshold) & valid

    # Consecutive anomaly tracking for regime-shift triggers
    flags = np.zeros((T, H, W), dtype=np.uint8)
    change_date = np.full((H, W), np.nan, dtype=np.float64)
    change_count = np.zeros((H, W), dtype=np.int32)

    consec = np.zeros((H, W), dtype=np.int32)
    streak_start = np.zeros((H, W), dtype=np.int32)

    for t in range(T):
        anom_t = is_anomaly[t]                      # (H, W) bool

        # Track streak start
        new_streak = anom_t & (consec == 0)
        streak_start[new_streak] = t

        consec[anom_t] += 1
        consec[~anom_t] = 0

        flags[t] = anom_t.astype(np.uint8)

        # Trigger regime-shift detection
        triggered = consec >= consecutive_required
        change_count[triggered] += 1

        first_detect = triggered & np.isnan(change_date)
        if np.any(first_detect):
            change_date[first_detect] = acq_times[
                streak_start[first_detect]
            ]

    total_flags = int(np.count_nonzero(flags))
    total_shifts = int(np.count_nonzero(change_count > 0))
    logger.info(
        f"Reconstruction anomalies: {total_flags:,} flagged observations, "
        f"{total_shifts:,} pixels with regime shifts"
    )

    return flags, change_date, change_count, recon_error, sigma


# ============================================================
# 8. ANOMALY DETECTION — METHOD B: LATENT CUSUM DRIFT
# ============================================================
def detect_latent_drift(latent_trajectory, valid_mask, acq_times,
                        cusum_k, cusum_h, consecutive_required):
    """
    Detect regime shifts via CUSUM on Mahalanobis distance in latent space.

    For each pixel:
        1. Compute mean μ and covariance Σ of z(t) over all valid timesteps
        2. Compute Mahalanobis distance d(t) at each timestep
        3. Standardize d(t) and compute data-driven reference level
        4. Run one-sided CUSUM: S(t) = max(0, S(t-1) + d(t) - ref - k)
        5. Flag when S(t) > h for consecutive_required consecutive timesteps

    The data-driven reference level (median standardized Mahalanobis distance)
    ensures robustness to non-Gaussian latent distributions and estimation noise.

    Args:
        latent_trajectory: (T, H, W, D) — latent states at each timestep
        valid_mask:        (T, H, W)    — boolean validity mask
        acq_times:         (T,)         — UNIX timestamps
        cusum_k:           float        — allowance parameter
        cusum_h:           float        — decision threshold
        consecutive_required: int

    Returns:
        drift_flags:      (T, H, W) uint8
        change_date:      (H, W)    float64
        change_count:     (H, W)    int32
        mahal_dist:       (T, H, W) float32  — standardized Mahalanobis distance
        cusum_stat:       (T, H, W) float32  — CUSUM accumulator
    """
    T, H, W, D = latent_trajectory.shape
    N = H * W

    # Flatten spatial dims
    Z = latent_trajectory.reshape(T, N, D)      # (T, N, D)
    M = valid_mask.reshape(T, N)                 # (T, N) bool

    valid_counts = M.sum(axis=0)                 # (N,)

    # Need at least D+2 samples for stable covariance estimation
    min_valid = D + 2
    sufficient = valid_counts >= min_valid

    # Allocate outputs
    mahal_dist = np.full((T, H, W), np.nan, dtype=np.float32)
    cusum_stat = np.zeros((T, H, W), dtype=np.float32)
    drift_flags = np.zeros((T, H, W), dtype=np.uint8)
    change_date = np.full((H, W), np.nan, dtype=np.float64)
    change_count = np.zeros((H, W), dtype=np.int32)

    suff_idx = np.where(sufficient)[0]
    if len(suff_idx) == 0:
        logger.warning(
            "No pixels with sufficient valid observations for drift detection"
        )
        return drift_flags, change_date, change_count, mahal_dist, cusum_stat

    insuff_count = N - len(suff_idx)
    logger.info(
        f"Latent drift: {len(suff_idx):,}/{N:,} pixels eligible "
        f"({insuff_count:,} excluded, need ≥{min_valid} valid obs)"
    )

    # Process in chunks to manage memory for covariance computation
    CHUNK = 4096

    for chunk_start in tqdm(
        range(0, len(suff_idx), CHUNK), desc="Drift detection"
    ):
        chunk_end = min(chunk_start + CHUNK, len(suff_idx))
        cidx = suff_idx[chunk_start:chunk_end]
        P = len(cidx)

        Z_chunk = Z[:, cidx, :]                  # (T, P, D)
        M_chunk = M[:, cidx]                     # (T, P) bool

        # --- Per-pixel mean ---
        N_valid = M_chunk.sum(axis=0)[:, None]   # (P, 1)
        Z_sum = (Z_chunk * M_chunk[:, :, None]).sum(axis=0)   # (P, D)
        Z_mean = Z_sum / N_valid                 # (P, D)

        # --- Per-pixel covariance ---
        Z_centered = Z_chunk - Z_mean[None, :, :]             # (T, P, D)
        Z_centered_masked = Z_centered * M_chunk[:, :, None]  # zero invalid

        # (P, D, D) via einsum: outer product accumulated over time
        cov = np.einsum('tpd,tpe->pde', Z_centered_masked, Z_centered_masked)
        cov = cov / (N_valid[:, :, None] - 1)    # unbiased estimator

        # Tikhonov regularization for numerical stability
        cov += np.eye(D, dtype=np.float32)[None, :, :] * 1e-5

        # --- Covariance inverse ---
        try:
            cov_inv = np.linalg.inv(cov)         # (P, D, D)
        except np.linalg.LinAlgError:
            logger.warning(
                f"Singular covariance at chunk {chunk_start}; "
                f"falling back to pseudo-inverse"
            )
            cov_inv = np.linalg.pinv(cov)

        # --- Mahalanobis distance at each timestep ---
        # d²(t) = (z-μ)ᵀ Σ⁻¹ (z-μ)
        mahal_sq = np.einsum(
            'tpd,pde,tpe->tp', Z_centered, cov_inv, Z_centered
        )  # (T, P)
        mahal = np.sqrt(np.maximum(mahal_sq, 0.0))

        # Standardize by expected chi-distribution mean ≈ √D
        expected_null = np.sqrt(float(D))
        mahal_standardized = mahal / expected_null  # (T, P)

        # Data-driven reference level: median per pixel (robust to shifts)
        mahal_with_nan = np.where(M_chunk, mahal_standardized, np.nan)
        ref_level = np.nanmedian(mahal_with_nan, axis=0)  # (P,)

        # Map flat indices back to spatial coordinates
        pixel_y, pixel_x = np.unravel_index(cidx, (H, W))

        # Store Mahalanobis distances
        mahal_dist[:, pixel_y, pixel_x] = mahal_standardized.astype(
            np.float32
        )

        # --- CUSUM per pixel ---
        S = np.zeros(P, dtype=np.float32)
        consec = np.zeros(P, dtype=np.int32)
        streak_start_chunk = np.zeros(P, dtype=np.int32)
        count = np.zeros(P, dtype=np.int32)
        first_date = np.full(P, np.nan, dtype=np.float64)

        for t in range(T):
            valid_t = M_chunk[t]                  # (P,) bool
            d_t = mahal_standardized[t]           # (P,)

            # CUSUM: S(t) = max(0, S(t-1) + d(t) - reference - k)
            S_new = np.maximum(0.0, S + d_t - ref_level - cusum_k)

            # Only update CUSUM at valid observations
            S = np.where(valid_t, S_new, S)

            cusum_stat[t, pixel_y, pixel_x] = S

            # Threshold crossing
            triggered = (S > cusum_h) & valid_t
            drift_flags[t, pixel_y, pixel_x] = triggered.astype(np.uint8)

            # Consecutive tracking
            new_streak = triggered & (consec == 0)
            streak_start_chunk[new_streak] = t

            consec[triggered] += 1
            consec[~triggered] = 0

            # Regime shift trigger
            regime_shift = consec >= consecutive_required
            count[regime_shift] += 1

            first_detect = regime_shift & np.isnan(first_date)
            if np.any(first_detect):
                first_date[first_detect] = acq_times[
                    streak_start_chunk[first_detect]
                ]

        change_count[pixel_y, pixel_x] = count
        change_date[pixel_y, pixel_x] = first_date

    total_drift_flags = int(np.count_nonzero(drift_flags))
    total_drift_shifts = int(np.count_nonzero(change_count > 0))
    logger.info(
        f"Latent drift: {total_drift_flags:,} flagged observations, "
        f"{total_drift_shifts:,} pixels with regime shifts"
    )

    return drift_flags, change_date, change_count, mahal_dist, cusum_stat


# ============================================================
# 9. MAIN
# ============================================================
def main():
    # --- Reproducibility ---
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ----- Data -----
    logger.info(f"Loading {H5_PATH}")
    dataset = HDF5SpatiotemporalDataset(
        H5_PATH, TARGET_METRIC, PATCH_SIZE, SAMPLES_PER_EPOCH
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    logger.info(
        f"Dataset: {dataset.time_len} frames × "
        f"{dataset.height}×{dataset.width} pixels"
    )

    # ----- Model -----
    model = SpatiotemporalLatentODE(
        input_dim=1, latent_dim=LATENT_DIM
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # ----- Training -----
    logger.info(
        f"Training: up to {NUM_EPOCHS} epochs, "
        f"{SAMPLES_PER_EPOCH} patches/epoch, "
        f"{TRAIN_TIME_SUBSAMPLE} timesteps/batch, "
        f"patience={PATIENCE}"
    )

    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    loss_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = train_epoch(
            model, dataloader, optimizer, device, TRAIN_TIME_SUBSAMPLE
        )
        scheduler.step(epoch_loss)
        loss_history.append(epoch_loss)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS} │ "
            f"Loss: {epoch_loss:.6f} │ LR: {current_lr:.2e}"
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(best loss: {best_loss:.6f})"
            )
            break

    # Restore best model
    assert best_state is not None, \
        "CRITICAL: No valid training epoch completed"
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    logger.info(f"Restored best model (loss: {best_loss:.6f})")

    # ----- Full Inference -----
    logger.info("Running full-scene inference...")
    predictions, latent_traj, z0 = run_full_inference(
        model, dataset, device, tile_size=PATCH_SIZE
    )

    # ----- Anomaly Detection: Reconstruction (Method A) -----
    logger.info("Detecting reconstruction anomalies (Method A)...")
    recon_flags, recon_date, recon_count, recon_error, recon_sigma = \
        detect_reconstruction_anomalies(
            dataset.data, predictions, dataset.valid_mask, dataset.acq_times,
            RECON_THRESHOLD_SIGMA, CONSECUTIVE_ANOMALIES_RECON,
        )

    # ----- Anomaly Detection: Latent Drift (Method B) -----
    logger.info("Detecting latent-space drift (Method B)...")
    drift_flags, drift_date, drift_count, mahal_dist, cusum_stat = \
        detect_latent_drift(
            latent_traj, dataset.valid_mask, dataset.acq_times,
            CUSUM_ALLOWANCE, CUSUM_THRESHOLD, CONSECUTIVE_ANOMALIES_DRIFT,
        )

    # ----- Save Output -----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
        OUTPUT_DIR, f"{LOCATION}_LatentODE_RegimeShifts.h5"
    )
    logger.info(f"Saving results to {output_path}")

    with h5py.File(output_path, 'w') as out:
        # --- Root Metadata (geolocation) ---
        out.attrs['spatial_ref'] = dataset.spatial_ref
        out.attrs['GeoTransform'] = dataset.geo_transform
        out.attrs['acquisition_time'] = dataset.acq_times
        out.attrs['SOURCE_DATA'] = H5_PATH
        out.attrs['TARGET_METRIC'] = TARGET_METRIC
        out.attrs['LOCATION'] = LOCATION

        # --- Hyperparameters ---
        hp = out.create_group('hyperparameters')
        hp.attrs['LATENT_DIM'] = LATENT_DIM
        hp.attrs['PATCH_SIZE'] = PATCH_SIZE
        hp.attrs['LEARNING_RATE'] = LEARNING_RATE
        hp.attrs['BEST_TRAINING_LOSS'] = best_loss
        hp.attrs['EPOCHS_TRAINED'] = len(loss_history)
        hp.attrs['TRAIN_TIME_SUBSAMPLE'] = TRAIN_TIME_SUBSAMPLE
        hp.attrs['LAMBDA_SPATIAL'] = LAMBDA_SPATIAL
        hp.attrs['RECON_THRESHOLD_SIGMA'] = RECON_THRESHOLD_SIGMA
        hp.attrs['CONSECUTIVE_ANOMALIES_RECON'] = CONSECUTIVE_ANOMALIES_RECON
        hp.attrs['CUSUM_ALLOWANCE'] = CUSUM_ALLOWANCE
        hp.attrs['CUSUM_THRESHOLD'] = CUSUM_THRESHOLD
        hp.attrs['CONSECUTIVE_ANOMALIES_DRIFT'] = CONSECUTIVE_ANOMALIES_DRIFT
        hp.create_dataset('loss_history', data=np.array(loss_history))

        # --- Predictions ---
        out.create_dataset(
            'predicted_series', data=predictions, compression='gzip'
        )
        out.create_dataset(
            'reconstruction_error', data=recon_error, compression='gzip'
        )
        out.create_dataset(
            'reconstruction_sigma', data=recon_sigma, compression='gzip'
        )

        # --- Reconstruction Anomalies (Method A) ---
        rg = out.create_group('reconstruction_anomalies')
        rg.create_dataset('flags', data=recon_flags, compression='gzip')
        rg.create_dataset(
            'change_date_timestamp', data=recon_date, compression='gzip'
        )
        rg.create_dataset(
            'change_count', data=recon_count, compression='gzip'
        )

        # --- Latent Drift (Method B) ---
        dg = out.create_group('latent_drift')
        dg.create_dataset('flags', data=drift_flags, compression='gzip')
        dg.create_dataset(
            'change_date_timestamp', data=drift_date, compression='gzip'
        )
        dg.create_dataset(
            'change_count', data=drift_count, compression='gzip'
        )
        dg.create_dataset(
            'mahalanobis_distance', data=mahal_dist, compression='gzip'
        )
        dg.create_dataset(
            'cusum_statistic', data=cusum_stat, compression='gzip'
        )

        # --- Latent Space ---
        lg = out.create_group('latent_space')
        lg.create_dataset('z0', data=z0, compression='gzip')
        if SAVE_LATENT_TRAJECTORY:
            traj_mb = latent_traj.nbytes / (1024 * 1024)
            logger.info(
                f"Saving latent trajectory: "
                f"{latent_traj.shape} ({traj_mb:.0f} MB uncompressed)"
            )
            lg.create_dataset(
                'trajectory', data=latent_traj, compression='gzip'
            )

    # --- Save Model Weights ---
    model_path = os.path.join(
        OUTPUT_DIR, f"{LOCATION}_LatentODE_model.pt"
    )
    torch.save(best_state, model_path)
    logger.info(f"Model weights saved to {model_path}")

    logger.info("Pipeline complete.")


if __name__ == '__main__':
    main()