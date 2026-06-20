"""
Contrastive Transformer for Spatiotemporal Regime Shift Detection
=================================================================
Self-supervised contrastive learning on temporal trajectories for change
detection in spectral complexity time series.

Unlike reconstruction-based methods (Neural ODE, DHR), this approach learns
*representations* of temporal dynamics via spatial InfoNCE: pixels with
similar temporal patterns should have similar embeddings. Regime shifts
manifest as sudden divergence in a pixel's embedding from either its own
temporal baseline (Method A) or its spatial neighborhood (Method B).

Architecture:
    - Time2Vec continuous-time positional encoding (Kazemi et al. 2019)
    - Masked-attention Transformer encoder (invalid observations excluded)
    - CLS-token aggregation → projection head for contrastive training
    - Pre-projection embeddings used for post-training anomaly analysis

References:
    - Kazemi, S. M., et al. (2019). "Time2Vec: Learning a Vector
      Representation of Time."
    - Jean, N., et al. (2019). "Tile2Vec: Unsupervised representation
      learning for spatially distributed data."
    - Chen, T., et al. (2020). "A Simple Framework for Contrastive Learning
      of Visual Representations." (SimCLR — InfoNCE formulation)
    - Page, E. S. (1954). "Continuous Inspection Schemes." Biometrika.
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
OUTPUT_DIR = "C:/satelliteImagery/HLST30/ContrastiveTransformer"

# --- Training ---
PATCH_SIZE = 32
WINDOW_SIZE = 100            # Temporal window per sample
EMBED_DIM = 64               # Transformer embedding dimension
NUM_HEADS = 4
NUM_LAYERS = 3
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
SAMPLES_PER_EPOCH = 500
GRAD_CLIP_NORM = 1.0
PATIENCE = 15
TEMPERATURE = 0.1            # InfoNCE temperature
RANDOM_SEED = 42

# --- Inference ---
INFERENCE_STRIDE = 50        # Sliding window stride for full-scene inference
INFERENCE_TILE_SIZE = 32     # Spatial tile size for inference

# --- Temporal Drift Anomaly Detection (Method A) ---
DRIFT_REFERENCE_WINDOWS = 3  # Number of initial windows for reference embedding
DRIFT_THRESHOLD_SIGMA = 3.0
CONSECUTIVE_ANOMALIES_DRIFT = 3

# --- Spatial Coherence Anomaly Detection (Method B) ---
COHERENCE_THRESHOLD_SIGMA = 3.0
CONSECUTIVE_ANOMALIES_COHERENCE = 3


# ============================================================
# 2. DATA LOADING
# ============================================================
class HDF5TemporalWindowDataset(torch.utils.data.Dataset):
    """
    Loads spatial patches and temporal windows from an HDF5 dataset.

    Invalid data (NaN + common_mask=1) is set to 0.0 for numerical
    neutrality in the Transformer's linear projections. The boolean
    valid_mask is the authoritative record of data integrity and drives
    the Transformer's key_padding_mask to exclude invalid timesteps
    from attention computation.

    HDF5 Schema:
        /HDFEOS/GRIDS/HARMONIZED/Data Fields/
            sliding_volume_z_score  (T, H, W) float32
            common_mask             (T, H, W) uint8, 1 = invalid
        Attributes on metric dataset:
            acquisition_time        (T,) float64  UNIX epoch seconds
            GeoTransform            (6,) float64
            spatial_ref             str (WKT)
    """

    def __init__(self, h5_filepath, target_metric, patch_size, window_size,
                 samples_per_epoch):
        super().__init__()
        self.patch_size = patch_size
        self.window_size = window_size
        self.samples_per_epoch = samples_per_epoch

        assert os.path.exists(h5_filepath), \
            f"HDF5 file not found: {h5_filepath}"

        data_path = f"/HDFEOS/GRIDS/HARMONIZED/Data Fields/{target_metric}"
        mask_path = "/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask"

        with h5py.File(h5_filepath, 'r') as f:
            raw_data = f[data_path][:]
            quality_mask_raw = f[mask_path][:]
            self.acq_times = np.array(
                f[data_path].attrs['acquisition_time'], dtype=np.float64
            )
            self.geo_transform = f[data_path].attrs['GeoTransform']
            self.spatial_ref = f[data_path].attrs['spatial_ref']

        # Data integrity assertions
        assert raw_data.dtype == np.float32, \
            f"CRITICAL: Expected float32 data, got {raw_data.dtype}"
        assert quality_mask_raw.dtype == np.uint8, \
            f"CRITICAL: Expected uint8 mask, got {quality_mask_raw.dtype}"
        assert self.acq_times.shape[0] == raw_data.shape[0], \
            "CRITICAL: Timestamp count does not match temporal depth."

        # Sort chronologically
        sort_idx = np.argsort(self.acq_times)
        raw_data = raw_data[sort_idx]
        quality_mask_raw = quality_mask_raw[sort_idx]
        self.acq_times = self.acq_times[sort_idx]

        self.time_len, self.height, self.width = raw_data.shape
        assert self.height >= patch_size and self.width >= patch_size, \
            (f"Spatial dimensions ({self.height}, {self.width}) are smaller "
             f"than patch_size ({patch_size})")
        assert self.time_len >= window_size, \
            (f"Temporal depth ({self.time_len}) is smaller than "
             f"window_size ({window_size})")

        # Construct validity mask
        nan_mask = np.isnan(raw_data)
        quality_mask = quality_mask_raw == 1
        self.valid_mask = (~nan_mask) & (~quality_mask)

        invalid_count = int(np.count_nonzero(~self.valid_mask))
        total_count = int(self.valid_mask.size)
        valid_pct = 100.0 * (1.0 - invalid_count / total_count)
        logger.info(
            f"Validity: {total_count - invalid_count:,}/{total_count:,} "
            f"observations ({valid_pct:.1f}%)"
        )

        # Set invalid positions to 0.0 — neutral for masked attention.
        # The valid_mask is the authoritative integrity record.
        raw_data[~self.valid_mask] = 0.0
        self.data = raw_data

        # Normalize timestamps to [0, 1] for Time2Vec stability
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
        t_start = np.random.randint(0, self.time_len - self.window_size + 1)
        t_end = t_start + self.window_size

        patch_data = self.data[
            t_start:t_end, y:y + self.patch_size, x:x + self.patch_size
        ]
        patch_mask = self.valid_mask[
            t_start:t_end, y:y + self.patch_size, x:x + self.patch_size
        ]
        window_timestamps = self.norm_timestamps[t_start:t_end]

        # Shape: (T, 1, patch_H, patch_W)
        patch_data = torch.from_numpy(patch_data).unsqueeze(1).float()
        patch_mask = torch.from_numpy(patch_mask).unsqueeze(1).bool()
        timestamps = torch.from_numpy(window_timestamps).float()

        return timestamps, patch_data, patch_mask


# ============================================================
# 3. MODEL ARCHITECTURE
# ============================================================
class Time2Vec(nn.Module):
    """
    Continuous-time positional encoding.

    The first component is a learned linear function of time (trend),
    and the remaining components are periodic (sine) functions that
    capture multi-scale temporal patterns.

    Ref: Kazemi, S. M., et al. (2019). "Time2Vec: Learning a Vector
    Representation of Time."
    """

    def __init__(self, out_features):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(out_features - 1))
        self.b = nn.Parameter(torch.randn(out_features - 1))

    def forward(self, tau):
        # tau: (...) → (..., out_features)
        tau = tau.unsqueeze(-1)
        v1 = self.w0 * tau + self.b0
        v2 = torch.sin(self.w * tau + self.b)
        return torch.cat([v1, v2], dim=-1)


class ContinuousTimeTransformer(nn.Module):
    """
    Transformer with masked attention for irregularly observed time series.

    Uses a CLS token to aggregate temporal information into a single
    per-pixel representation. The key_padding_mask excludes invalid
    observations from attention computation (PyTorch convention: True = ignore).

    For inference, `encode()` returns the pre-projection embedding (richer
    representation), while `forward()` returns the post-projection embedding
    (used during contrastive training).
    """

    def __init__(self, feature_dim=1, embed_dim=64, num_heads=4,
                 num_layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_projection = nn.Linear(feature_dim, embed_dim)
        self.time2vec = Time2Vec(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2)
        )

    def _encode_core(self, timestamps, data_seq, valid_mask_seq):
        """
        Shared encoder logic. Returns CLS-token embedding before projection.

        Args:
            timestamps:      (T,)
            data_seq:        (T, N, C)  — observation values (0.0 at invalid)
            valid_mask_seq:  (T, N, C)  — boolean validity mask

        Returns:
            cls_embedding: (N, embed_dim) — pre-projection CLS representation
        """
        T, N, C = data_seq.shape
        device = data_seq.device

        # Zero out invalid inputs (redundant if data is already 0.0, but
        # ensures correctness regardless of upstream state)
        safe_data = data_seq * valid_mask_seq.float()

        x = self.input_projection(safe_data)        # (T, N, embed_dim)

        # Time2Vec encoding — broadcast timestamps across pixels
        t_expand = timestamps.unsqueeze(1).expand(T, N)
        t_embed = self.time2vec(t_expand)            # (T, N, embed_dim)
        x = x + t_embed

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(1, N, -1)
        x = torch.cat((cls_tokens, x), dim=0)       # (T+1, N, embed_dim)

        # Build padding mask: True = ignore (PyTorch convention)
        padding_mask = ~valid_mask_seq.squeeze(-1).transpose(0, 1)  # (N, T)
        cls_mask = torch.zeros(N, 1, dtype=torch.bool, device=device)
        full_padding_mask = torch.cat(
            (cls_mask, padding_mask), dim=1
        )  # (N, T+1)

        out = self.transformer(x, src_key_padding_mask=full_padding_mask)
        cls_embedding = out[0, :, :]                 # (N, embed_dim)

        return cls_embedding

    def forward(self, timestamps, data_seq, valid_mask_seq):
        """Returns post-projection embedding for contrastive training."""
        cls_embedding = self._encode_core(
            timestamps, data_seq, valid_mask_seq
        )
        z = self.projection_head(cls_embedding)      # (N, embed_dim // 2)
        return z

    def encode(self, timestamps, data_seq, valid_mask_seq):
        """Returns pre-projection embedding for downstream analysis."""
        return self._encode_core(timestamps, data_seq, valid_mask_seq)


# ============================================================
# 4. LOSS FUNCTION — SPATIAL InfoNCE
# ============================================================
def compute_spatial_infonce_loss(latent_grid, temperature=0.1):
    """
    InfoNCE loss treating immediate spatial neighbors as positive pairs.

    Uses the numerically stable log-sum-exp formulation from SimCLR
    (Chen et al. 2020): for each anchor-positive pair, the loss is
    computed as cross-entropy over the concatenated [positive, negatives]
    similarity vector, avoiding manual exp/division that can overflow.

    Ref: Jean, N., et al. (2019). "Tile2Vec: Unsupervised representation
    learning for spatially distributed data."

    Args:
        latent_grid: (B, D, H, W) — L2-normalized not required (done here)
        temperature: float — sharpness of the softmax distribution

    Returns:
        loss: scalar
    """
    B, D, H, W = latent_grid.shape

    # L2 normalize along feature dimension
    z_norm = F.normalize(latent_grid, dim=1)

    # Interior pixels only (border excluded — no valid neighbors)
    anchors = z_norm[:, :, 1:-1, 1:-1]              # (B, D, H-2, W-2)

    # 4-connected positive neighbors
    pos_up = z_norm[:, :, :-2, 1:-1]
    pos_down = z_norm[:, :, 2:, 1:-1]
    pos_left = z_norm[:, :, 1:-1, :-2]
    pos_right = z_norm[:, :, 1:-1, 2:]

    # Flatten spatial dims: (B*(H-2)*(W-2), D)
    M = B * (H - 2) * (W - 2)
    anchors_flat = anchors.permute(0, 2, 3, 1).reshape(M, D)

    positives = [pos_up, pos_down, pos_left, pos_right]
    total_loss = torch.tensor(0.0, device=latent_grid.device)

    for pos in positives:
        pos_flat = pos.permute(0, 2, 3, 1).reshape(M, D)

        # Positive similarity: (M, 1)
        sim_pos = (anchors_flat * pos_flat).sum(dim=-1, keepdim=True)

        # Negative similarities: all other anchors in the batch
        # (M, M) — diagonal entries are self-similarity, masked out
        sim_neg = torch.matmul(anchors_flat, anchors_flat.T)

        # Mask self-similarity with large negative value
        sim_neg.fill_diagonal_(-1e9)

        # Logits: [positive, negatives] / temperature
        # Shape: (M, 1 + M)
        logits = torch.cat([sim_pos, sim_neg], dim=1) / temperature

        # Target: index 0 is the positive
        labels = torch.zeros(M, dtype=torch.long, device=logits.device)

        total_loss = total_loss + F.cross_entropy(logits, labels)

    return total_loss / 4.0


# ============================================================
# 5. TRAINING
# ============================================================
def train_epoch(model, dataloader, optimizer, device):
    """One epoch of contrastive training."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for timestamps, patch_data, patch_mask in dataloader:
        timestamps = timestamps[0].to(device)       # (T,) — shared across batch
        B = patch_data.shape[0]
        T = timestamps.shape[0]
        H = patch_data.shape[3]
        W = patch_data.shape[4]

        # Reshape: (B, T, C, H, W) → (T, B*H*W, C)
        data_seq = (patch_data.to(device)
                    .permute(1, 0, 3, 4, 2)
                    .reshape(T, B * H * W, 1))
        mask_seq = (patch_mask.to(device)
                    .permute(1, 0, 3, 4, 2)
                    .reshape(T, B * H * W, 1))

        # Skip degenerate batches
        if not mask_seq.any():
            logger.warning("Skipping batch: no valid observations")
            continue

        optimizer.zero_grad()
        z = model(timestamps, data_seq, mask_seq)    # (B*H*W, D_proj)

        # Reshape to spatial grid for InfoNCE
        D_proj = z.shape[-1]
        latent_grid = z.view(B, H, W, D_proj).permute(0, 3, 1, 2)

        loss = compute_spatial_infonce_loss(latent_grid, TEMPERATURE)

        assert torch.isfinite(loss), \
            f"CRITICAL: Non-finite loss ({loss.item()}) — training instability"

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
def run_full_inference(model, dataset, device, tile_size, stride):
    """
    Slide the temporal window across the full sequence and tile the
    spatial grid to produce a per-pixel embedding trajectory.

    Args:
        model:     Trained ContinuousTimeTransformer
        dataset:   HDF5TemporalWindowDataset
        device:    torch.device
        tile_size: Spatial tile dimension
        stride:    Temporal window stride

    Returns:
        embeddings:    (num_windows, H, W, embed_dim) float32
        window_times:  (num_windows,) float64 — center timestamp per window
    """
    model.eval()

    T = dataset.time_len
    H, W = dataset.height, dataset.width
    D = model.embed_dim
    ws = dataset.window_size

    # Window start positions
    starts = list(range(0, T - ws + 1, stride))
    num_windows = len(starts)

    # Center timestamps (UNIX) for each window
    window_times = np.array([
        dataset.acq_times[s + ws // 2] for s in starts
    ], dtype=np.float64)

    # Allocate output on CPU
    all_embeddings = np.full(
        (num_windows, H, W, D), np.nan, dtype=np.float32
    )

    # Spatial tiling
    y_starts = list(range(0, H, tile_size))
    x_starts = list(range(0, W, tile_size))
    total_tiles = num_windows * len(y_starts) * len(x_starts)

    logger.info(
        f"Inference: {num_windows} windows × "
        f"{len(y_starts)*len(x_starts)} spatial tiles = "
        f"{total_tiles} total forward passes"
    )

    pbar = tqdm(total=total_tiles, desc="Inference")

    for wi, t_start in enumerate(starts):
        t_end = t_start + ws
        timestamps = torch.from_numpy(
            dataset.norm_timestamps[t_start:t_end]
        ).float().to(device)

        for ys in y_starts:
            ye = min(ys + tile_size, H)
            for xs in x_starts:
                xe = min(xs + tile_size, W)
                tH, tW = ye - ys, xe - xs

                tile_data = dataset.data[t_start:t_end, ys:ye, xs:xe]
                tile_mask = dataset.valid_mask[t_start:t_end, ys:ye, xs:xe]

                # (T_win, tH*tW, 1)
                td = (torch.from_numpy(tile_data).float()
                      .reshape(ws, tH * tW, 1).to(device))
                tm = (torch.from_numpy(tile_mask).bool()
                      .reshape(ws, tH * tW, 1).to(device))

                # Pre-projection embedding: (tH*tW, embed_dim)
                emb = model.encode(timestamps, td, tm)
                emb_spatial = emb.cpu().numpy().reshape(tH, tW, D)

                all_embeddings[wi, ys:ye, xs:xe, :] = emb_spatial
                pbar.update(1)

    pbar.close()

    # Verify integrity
    nan_count = int(np.isnan(all_embeddings).sum())
    assert nan_count == 0, \
        f"CRITICAL: {nan_count} NaN values in embeddings"

    return all_embeddings, window_times


# ============================================================
# 7. ANOMALY DETECTION — METHOD A: TEMPORAL COSINE DRIFT
# ============================================================
def detect_temporal_drift(embeddings, window_times, ref_window_count,
                          threshold_sigma, consecutive_required):
    """
    Detect regime shifts by tracking cosine distance between each pixel's
    embedding and a reference embedding derived from the first N windows.

    Args:
        embeddings:       (num_windows, H, W, D) — pre-projection embeddings
        window_times:     (num_windows,) — UNIX center timestamps
        ref_window_count: int — number of initial windows for reference
        threshold_sigma:  float — MAD-based threshold multiplier
        consecutive_required: int

    Returns:
        flags:            (num_windows, H, W) uint8
        change_date:      (H, W) float64
        change_count:     (H, W) int32
        cosine_distance:  (num_windows, H, W) float32
    """
    num_w, H, W, D = embeddings.shape

    assert ref_window_count <= num_w, \
        (f"ref_window_count ({ref_window_count}) must be less than or equal to "
         f"num_windows ({num_w})")

    # Reference embedding: L2-normalized mean of first N windows
    ref = embeddings[:ref_window_count].mean(axis=0)  # (H, W, D)
    ref_norm = ref / (np.linalg.norm(ref, axis=-1, keepdims=True) + 1e-8)

    # Cosine distance at each window
    emb_norm = embeddings / (
        np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8
    )
    cosine_sim = np.sum(emb_norm * ref_norm[None, :, :, :], axis=-1)
    cosine_distance = (1.0 - cosine_sim).astype(np.float32)

    # Per-pixel robust threshold via MAD
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice', RuntimeWarning)
        med = np.median(cosine_distance, axis=0, keepdims=True)
        mad = np.median(np.abs(cosine_distance - med), axis=0)
    sigma = np.clip(1.4826 * mad, a_min=1e-6, a_max=None)

    logger.info(
        f"Temporal drift σ: median={np.median(sigma):.4f}, "
        f"range=[{np.min(sigma):.4f}, {np.max(sigma):.4f}]"
    )

    threshold = threshold_sigma * sigma[None, :, :]
    is_anomaly = cosine_distance > threshold

    # Consecutive tracking
    flags = np.zeros((num_w, H, W), dtype=np.uint8)
    change_date = np.full((H, W), np.nan, dtype=np.float64)
    change_count = np.zeros((H, W), dtype=np.int32)

    consec = np.zeros((H, W), dtype=np.int32)
    streak_start = np.zeros((H, W), dtype=np.int32)

    for w in range(num_w):
        anom_w = is_anomaly[w]

        new_streak = anom_w & (consec == 0)
        streak_start[new_streak] = w

        consec[anom_w] += 1
        consec[~anom_w] = 0

        flags[w] = anom_w.astype(np.uint8)

        triggered = consec >= consecutive_required
        change_count[triggered] += 1

        first_detect = triggered & np.isnan(change_date)
        if np.any(first_detect):
            change_date[first_detect] = window_times[
                streak_start[first_detect]
            ]

    total_flags = int(np.count_nonzero(flags))
    total_shifts = int(np.count_nonzero(change_count > 0))
    logger.info(
        f"Temporal drift: {total_flags:,} flagged windows, "
        f"{total_shifts:,} pixels with regime shifts"
    )

    return flags, change_date, change_count, cosine_distance


# ============================================================
# 8. ANOMALY DETECTION — METHOD B: SPATIAL COHERENCE
# ============================================================
def detect_spatial_coherence_anomaly(embeddings, window_times,
                                     threshold_sigma,
                                     consecutive_required):
    """
    Detect regime shifts by monitoring cosine similarity between each
    pixel's embedding and its 4-connected spatial neighbors. Exploits
    the same spatial prior used during contrastive training.

    Args:
        embeddings:       (num_windows, H, W, D)
        window_times:     (num_windows,)
        threshold_sigma:  float
        consecutive_required: int

    Returns:
        flags:              (num_windows, H, W) uint8
        change_date:        (H, W) float64
        change_count:       (H, W) int32
        neighbor_similarity: (num_windows, H, W) float32
    """
    num_w, H, W, D = embeddings.shape

    emb_norm = embeddings / (
        np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8
    )

    # Mean cosine similarity to 4-connected neighbors
    neighbor_sim = np.zeros((num_w, H, W), dtype=np.float32)
    neighbor_count = np.zeros((H, W), dtype=np.float32)

    # Accumulate neighbor similarities
    # Up neighbor
    neighbor_sim[:, 1:, :] += np.sum(
        emb_norm[:, 1:, :, :] * emb_norm[:, :-1, :, :], axis=-1
    )
    neighbor_count[1:, :] += 1

    # Down neighbor
    neighbor_sim[:, :-1, :] += np.sum(
        emb_norm[:, :-1, :, :] * emb_norm[:, 1:, :, :], axis=-1
    )
    neighbor_count[:-1, :] += 1

    # Left neighbor
    neighbor_sim[:, :, 1:] += np.sum(
        emb_norm[:, :, 1:, :] * emb_norm[:, :, :-1, :], axis=-1
    )
    neighbor_count[:, 1:] += 1

    # Right neighbor
    neighbor_sim[:, :, :-1] += np.sum(
        emb_norm[:, :, :-1, :] * emb_norm[:, :, 1:, :], axis=-1
    )
    neighbor_count[:, :-1] += 1

    # Average over available neighbors (corners=2, edges=3, interior=4)
    neighbor_count = np.maximum(neighbor_count, 1.0)
    neighbor_sim = neighbor_sim / neighbor_count[None, :, :]

    # Low similarity = divergent → anomalous
    # Invert so higher values = more anomalous (consistent with drift method)
    divergence = (1.0 - neighbor_sim).astype(np.float32)

    # Per-pixel robust threshold via MAD
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice', RuntimeWarning)
        med = np.median(divergence, axis=0, keepdims=True)
        mad = np.median(np.abs(divergence - med), axis=0)
    sigma = np.clip(1.4826 * mad, a_min=1e-6, a_max=None)

    logger.info(
        f"Spatial coherence σ: median={np.median(sigma):.4f}, "
        f"range=[{np.min(sigma):.4f}, {np.max(sigma):.4f}]"
    )

    threshold = threshold_sigma * sigma[None, :, :]
    is_anomaly = divergence > threshold

    # Consecutive tracking
    flags = np.zeros((num_w, H, W), dtype=np.uint8)
    change_date = np.full((H, W), np.nan, dtype=np.float64)
    change_count = np.zeros((H, W), dtype=np.int32)

    consec = np.zeros((H, W), dtype=np.int32)
    streak_start = np.zeros((H, W), dtype=np.int32)

    for w in range(num_w):
        anom_w = is_anomaly[w]

        new_streak = anom_w & (consec == 0)
        streak_start[new_streak] = w

        consec[anom_w] += 1
        consec[~anom_w] = 0

        flags[w] = anom_w.astype(np.uint8)

        triggered = consec >= consecutive_required
        change_count[triggered] += 1

        first_detect = triggered & np.isnan(change_date)
        if np.any(first_detect):
            change_date[first_detect] = window_times[
                streak_start[first_detect]
            ]

    total_flags = int(np.count_nonzero(flags))
    total_shifts = int(np.count_nonzero(change_count > 0))
    logger.info(
        f"Spatial coherence: {total_flags:,} flagged windows, "
        f"{total_shifts:,} pixels with regime shifts"
    )

    return flags, change_date, change_count, neighbor_sim


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
    dataset = HDF5TemporalWindowDataset(
        H5_PATH, TARGET_METRIC, PATCH_SIZE, WINDOW_SIZE, SAMPLES_PER_EPOCH
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    logger.info(
        f"Dataset: {dataset.time_len} frames × "
        f"{dataset.height}×{dataset.width} pixels, "
        f"window_size={WINDOW_SIZE}"
    )

    # ----- Model -----
    model = ContinuousTimeTransformer(
        feature_dim=1,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
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
        f"patience={PATIENCE}"
    )

    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    loss_history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = train_epoch(model, dataloader, optimizer, device)
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
    logger.info(
        f"Running full-scene inference (stride={INFERENCE_STRIDE})..."
    )
    embeddings, window_times = run_full_inference(
        model, dataset, device,
        tile_size=INFERENCE_TILE_SIZE,
        stride=INFERENCE_STRIDE,
    )
    logger.info(f"Embedding trajectory shape: {embeddings.shape}")

    # ----- Anomaly Detection: Temporal Drift (Method A) -----
    logger.info("Detecting temporal cosine drift (Method A)...")
    drift_flags, drift_date, drift_count, cosine_dist = \
        detect_temporal_drift(
            embeddings, window_times, DRIFT_REFERENCE_WINDOWS,
            DRIFT_THRESHOLD_SIGMA, CONSECUTIVE_ANOMALIES_DRIFT,
        )

    # ----- Anomaly Detection: Spatial Coherence (Method B) -----
    logger.info("Detecting spatial coherence anomalies (Method B)...")
    coher_flags, coher_date, coher_count, neighbor_sim = \
        detect_spatial_coherence_anomaly(
            embeddings, window_times,
            COHERENCE_THRESHOLD_SIGMA, CONSECUTIVE_ANOMALIES_COHERENCE,
        )

    # ----- Save Output -----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{LOCATION}_ContrastiveTransformer_RegimeShifts.h5"
    )
    logger.info(f"Saving results to {output_path}")

    with h5py.File(output_path, 'w') as out:
        # --- Root Metadata (geolocation) ---
        out.attrs['spatial_ref'] = dataset.spatial_ref
        out.attrs['GeoTransform'] = dataset.geo_transform
        out.attrs['acquisition_time'] = dataset.acq_times
        out.attrs['window_center_times'] = window_times
        out.attrs['SOURCE_DATA'] = H5_PATH
        out.attrs['TARGET_METRIC'] = TARGET_METRIC
        out.attrs['LOCATION'] = LOCATION

        # --- Hyperparameters ---
        hp = out.create_group('hyperparameters')
        hp.attrs['EMBED_DIM'] = EMBED_DIM
        hp.attrs['NUM_HEADS'] = NUM_HEADS
        hp.attrs['NUM_LAYERS'] = NUM_LAYERS
        hp.attrs['PATCH_SIZE'] = PATCH_SIZE
        hp.attrs['WINDOW_SIZE'] = WINDOW_SIZE
        hp.attrs['INFERENCE_STRIDE'] = INFERENCE_STRIDE
        hp.attrs['TEMPERATURE'] = TEMPERATURE
        hp.attrs['LEARNING_RATE'] = LEARNING_RATE
        hp.attrs['BEST_TRAINING_LOSS'] = best_loss
        hp.attrs['EPOCHS_TRAINED'] = len(loss_history)
        hp.attrs['DRIFT_REFERENCE_WINDOWS'] = DRIFT_REFERENCE_WINDOWS
        hp.attrs['DRIFT_THRESHOLD_SIGMA'] = DRIFT_THRESHOLD_SIGMA
        hp.attrs['CONSECUTIVE_ANOMALIES_DRIFT'] = CONSECUTIVE_ANOMALIES_DRIFT
        hp.attrs['COHERENCE_THRESHOLD_SIGMA'] = COHERENCE_THRESHOLD_SIGMA
        hp.attrs['CONSECUTIVE_ANOMALIES_COHERENCE'] = \
            CONSECUTIVE_ANOMALIES_COHERENCE
        hp.create_dataset(
            'loss_history', data=np.array(loss_history)
        )

        # --- Embeddings ---
        eg = out.create_group('embeddings')
        eg.create_dataset(
            'trajectory', data=embeddings, compression='gzip'
        )
        eg.create_dataset(
            'window_center_times', data=window_times
        )

        # --- Temporal Drift (Method A) ---
        dg = out.create_group('temporal_drift')
        dg.create_dataset('flags', data=drift_flags, compression='gzip')
        dg.create_dataset(
            'change_date_timestamp', data=drift_date, compression='gzip'
        )
        dg.create_dataset(
            'change_count', data=drift_count, compression='gzip'
        )
        dg.create_dataset(
            'cosine_distance', data=cosine_dist, compression='gzip'
        )

        # --- Spatial Coherence (Method B) ---
        cg = out.create_group('spatial_coherence')
        cg.create_dataset('flags', data=coher_flags, compression='gzip')
        cg.create_dataset(
            'change_date_timestamp', data=coher_date, compression='gzip'
        )
        cg.create_dataset(
            'change_count', data=coher_count, compression='gzip'
        )
        cg.create_dataset(
            'neighbor_similarity', data=neighbor_sim, compression='gzip'
        )

    # --- Save Model Weights ---
    model_path = os.path.join(
        OUTPUT_DIR, f"{LOCATION}_ContrastiveTransformer_model.pt"
    )
    torch.save(best_state, model_path)
    logger.info(f"Model weights saved to {model_path}")

    logger.info("Pipeline complete.")


if __name__ == '__main__':
    main()