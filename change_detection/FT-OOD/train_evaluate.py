import os
import torch
import torch.nn as nn
import math
import numpy as np
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import ALFTSequenceDataset
from model import BatchedStreamingDriftDetector

from pnpxai.explainers import IntegratedGradients
from pnpxai.evaluator.metrics import Sensitivity, MuFidelity, Complexity


class PnPXAIWrapper(nn.Module):
    """Wraps OOD_Anomaly_Detector for PnPXAI single-input interface.

    PnPXAI v0.1.4 does not support additional_forward_args. This wrapper
    freezes T_seq and padding_mask as closure state and exposes only X_alft
    as the differentiable input. The scalar SVDD distance is reshaped to
    (B, 1) so PnPXAI can attribute against target index 0.

    The wrapper also handles internal batch expansion that PnPXAI metrics
    (e.g., Sensitivity) may perform during evaluation.
    """
    def __init__(self, base_model, T_seq, padding_mask):
        super().__init__()
        self.base_model = base_model
        self.T_seq = T_seq
        self.padding_mask = padding_mask

    def forward(self, x):
        orig_shape = x.shape
        # x could be (B, L, D) or (B, n_subsets, L, D) during MuFidelity
        x_flat = x.view(-1, orig_shape[-2], orig_shape[-1])
        flat_batch = x_flat.size(0)

        # Chunk large batches internally to prevent OOM
        # (PnPXAI's MuFidelity often passes thousands of subsets in a single batch
        # because it chunks by dim=0, which is just 1 for individual pixels)
        MAX_CHUNK = 64
        all_scores = []
        
        for i in range(0, flat_batch, MAX_CHUNK):
            chunk = x_flat[i:i + MAX_CHUNK]
            chunk_size = chunk.size(0)
            
            # Expand T_seq and padding_mask to match the chunk size
            batch_ratio = chunk_size // self.T_seq.size(0)
            if batch_ratio > 1:
                t = self.T_seq.repeat_interleave(batch_ratio, dim=0)
                m = self.padding_mask.repeat_interleave(batch_ratio, dim=0)
            else:
                t = self.T_seq[:chunk_size]
                m = self.padding_mask[:chunk_size]
                
            chunk_scores = self.base_model(chunk, t, m)  # (chunk_size,)
            all_scores.append(chunk_scores)
            
        scores = torch.cat(all_scores, dim=0)

        # PnPXAI expects (B, n_subsets, 1) for the target score indexing
        out_shape = orig_shape[:-2] + (1,)
        return scores.view(out_shape)

def extract_alft_window_batched(Y_chunk, M_chunk, frac_years_gpu, t_idx,
                                window_years, k_freqs, Omega, min_samples, device):
    """
    Batched ALFT feature extraction for one temporal window, one target timestep,
    across all P pixels in a spatial chunk.
    """
    P = Y_chunk.shape[1]
    fpw = 2 * k_freqs + 2
    out_features = torch.full((P, fpw), float('nan'), device=device)
    out_freqs = torch.full((P, k_freqs), float('nan'), device=device)
    out_valid = torch.zeros(P, dtype=torch.bool, device=device)

    target_time = frac_years_gpu[t_idx]
    window_start = target_time - window_years

    # Identify timesteps within the causal lookback window [window_start, target_time)
    in_window = (frac_years_gpu >= window_start) & (frac_years_gpu < target_time)
    W_indices = torch.where(in_window)[0]
    W = len(W_indices)

    if W < min_samples:
        return out_features, out_freqs, out_valid

    Y_win = Y_chunk[W_indices, :]     # (W, P)
    M_win = M_chunk[W_indices, :]     # (W, P)
    T_win = frac_years_gpu[W_indices]  # (W,)

    # Per-pixel valid sample count
    N_valid = M_win.sum(dim=0)  # (P,)
    has_enough = N_valid >= min_samples
    active_indices = torch.where(has_enough)[0]

    if len(active_indices) == 0:
        return out_features, out_freqs, out_valid

    P_active = len(active_indices)
    Y_active = Y_win[:, active_indices]  # (W, P_active)
    M_active = M_win[:, active_indices]  # (W, P_active)

    # Center data for spectral search (remove DC for cleaner OMP)
    Y_active_sum = (Y_active * M_active).sum(dim=0)
    M_active_sum = M_active.sum(dim=0)
    Y_active_mean = Y_active_sum / M_active_sum
    Y_active_centered = (Y_active - Y_active_mean.unsqueeze(0)) * M_active

    # NDFT matrix: E[k, w] = exp(-j * Omega[k] * T_win[w])
    E = torch.exp(-1j * Omega.unsqueeze(1) * T_win.unsqueeze(0))  # (K_grid, W)

    # ── Iterative OMP Frequency Extraction ──
    Y_residual = Y_active_centered.clone()
    Omega_active_list = []

    for k in range(k_freqs):
        # Spectrum: magnitude of NDFT of current residual
        Spectrum = torch.abs(
            torch.matmul(E, Y_residual.to(torch.complex64))
        )  # (K_grid, P_active)
        top1_indices = torch.argmax(Spectrum, dim=0)  # (P_active,)
        Omega_k = Omega[top1_indices]  # (P_active,)
        Omega_active_list.append(Omega_k)

        # Subtract fit of frequencies found so far (prepare residual for next OMP iter)
        if k < k_freqs - 1:
            Omega_so_far = torch.stack(Omega_active_list, dim=0)  # (k+1, P_active)
            angles_sf = (
                T_win.unsqueeze(1).unsqueeze(2) * Omega_so_far.unsqueeze(0)
            )  # (W, k+1, P_active)
            X_cos_sf = torch.cos(angles_sf)
            X_sin_sf = torch.sin(angles_sf)
            X_const_sf = torch.ones(W, 1, P_active, device=device)
            X_sf = torch.cat([X_const_sf, X_cos_sf, X_sin_sf], dim=1)  # (W, F_sf, P_active)
            X_sf = X_sf.permute(2, 0, 1)  # (P_active, W, F_sf)

            M_exp_sf = M_active.transpose(0, 1).unsqueeze(-1)  # (P_active, W, 1)
            X_masked_sf = X_sf * M_exp_sf

            F_sf = 2 * (k + 1) + 1
            XtX_sf = torch.bmm(X_masked_sf.transpose(1, 2), X_masked_sf)
            XtX_sf += torch.eye(F_sf, device=device) * 1e-5  # Tikhonov regularization

            Y_centered_exp = Y_active_centered.transpose(0, 1).unsqueeze(-1)
            Xty_sf = torch.bmm(
                X_masked_sf.transpose(1, 2), Y_centered_exp * M_exp_sf
            )

            beta_sf = torch.linalg.solve(XtX_sf, Xty_sf)
            Y_pred_sf = torch.bmm(X_sf, beta_sf).squeeze(-1).transpose(0, 1)
            Y_residual = (Y_active_centered - Y_pred_sf) * M_active

    # ── Final Design Matrix (all K frequencies, no trend term) ──
    Omega_active = torch.stack(Omega_active_list, dim=0)  # (K, P_active)
    angles = (
        T_win.unsqueeze(1).unsqueeze(2) * Omega_active.unsqueeze(0)
    )  # (W, K, P_active)
    X_cos = torch.cos(angles)
    X_sin = torch.sin(angles)
    X_const = torch.ones(W, 1, P_active, device=device)
    # Design matrix: [1, cos_1, cos_2, sin_1, sin_2]  (no trend column)
    X_active = torch.cat([X_const, X_cos, X_sin], dim=1)  # (W, F, P_active)
    X_active = X_active.permute(2, 0, 1)  # (P_active, W, F)

    M_active_exp = M_active.transpose(0, 1).unsqueeze(-1)  # (P_active, W, 1)
    X_masked = X_active * M_active_exp

    F = 2 * k_freqs + 1
    XtX = torch.bmm(X_masked.transpose(1, 2), X_masked)  # (P_active, F, F)
    XtX += torch.eye(F, device=device) * 1e-5

    # Use non-centered Y for final OLS (X_const absorbs the mean)
    Y_active_exp = Y_active.transpose(0, 1).unsqueeze(-1)  # (P_active, W, 1)
    Xty = torch.bmm(
        X_masked.transpose(1, 2), Y_active_exp * M_active_exp
    )  # (P_active, F, 1)

    beta = torch.linalg.solve(XtX, Xty)  # (P_active, F, 1)

    # ── MAD-based Robust RMSE (identical to dhr_main_pytorch.py lines 226-246) ──
    Y_pred = torch.bmm(X_active, beta)  # (P_active, W, 1)
    e = Y_active_exp - Y_pred
    e_valid = torch.where(
        M_active_exp.bool(), e,
        torch.tensor(float('nan'), device=device)
    )
    med_e = torch.nanmedian(e_valid, dim=1, keepdim=True).values
    mad_e = torch.nanmedian(
        torch.abs(e_valid - med_e), dim=1, keepdim=True
    ).values
    # 1.4826 assumes asymptotic normality of inliers (consistent estimator)
    sigma = torch.clamp(1.4826 * mad_e, min=1e-5).view(-1)  # (P_active,)

    # ── Assemble Feature Vector ──
    beta_0 = beta[:, 0, 0]                              # (P_active,)
    beta_cos = beta[:, 1:k_freqs + 1, 0]                # (P_active, K)
    beta_sin = beta[:, k_freqs + 1:2 * k_freqs + 1, 0]  # (P_active, K)

    features = torch.cat([
        beta_0.unsqueeze(1),   # (P_active, 1)
        beta_cos,              # (P_active, K)
        beta_sin,              # (P_active, K)
        sigma.unsqueeze(1)     # (P_active, 1)
    ], dim=1)  # (P_active, 2K+2)

    out_features[active_indices] = features
    out_freqs[active_indices] = Omega_active.transpose(0, 1)
    out_valid[active_indices] = True

    return out_features, out_freqs, out_valid


def precompute_alft_features(y_data, valid_mask, frac_years, alft_dim, windows, 
                             k_frequencies, f_grid_min, f_grid_max, f_grid_n, 
                             min_samples, chunk_size, device):
    """
    Pre-computes ALFT multi-scale features for the entire dataset.
    """
    num_frames, height, width = y_data.shape
    fpw = 2 * k_frequencies + 2

    alft_features = np.full(
        (num_frames, height, width, alft_dim), np.nan, dtype=np.float32
    )
    alft_freqs = np.full(
        (num_frames, height, width, len(windows), k_frequencies), np.nan, dtype=np.float32
    )
    alft_valid = np.zeros((num_frames, height, width), dtype=bool)

    # Prepare GPU tensors
    y_torch = torch.from_numpy(y_data).float()
    y_torch = torch.nan_to_num(y_torch, nan=0.0)  # NaN→0 under mask control
    m_torch = torch.from_numpy(valid_mask).bool()
    frac_years_gpu = torch.from_numpy(frac_years).float().to(device)

    f_grid = torch.linspace(f_grid_min, f_grid_max, f_grid_n, device=device)
    Omega = 2.0 * math.pi * f_grid

    y_chunks = list(range(0, height, chunk_size))
    x_chunks = list(range(0, width, chunk_size))
    total_chunks = len(y_chunks) * len(x_chunks)

    print(f"\nPhase 1: ALFT Feature Extraction ({total_chunks} spatial chunks)...")
    pbar = tqdm(total=total_chunks, desc="ALFT Extraction")

    for y_start in y_chunks:
        y_end = min(y_start + chunk_size, height)
        for x_start in x_chunks:
            x_end = min(x_start + chunk_size, width)

            chunk_h = y_end - y_start
            chunk_w = x_end - x_start
            P = chunk_h * chunk_w

            Y_chunk = y_torch[:, y_start:y_end, x_start:x_end].reshape(
                num_frames, P
            ).to(device)
            M_chunk = m_torch[:, y_start:y_end, x_start:x_end].reshape(
                num_frames, P
            ).to(device)

            # Per-timestep, per-window extraction
            chunk_features = torch.full(
                (num_frames, P, alft_dim), float('nan'), device=device
            )
            chunk_freqs = torch.full(
                (num_frames, P, len(windows), k_frequencies), float('nan'), device=device
            )
            chunk_valid = torch.zeros(
                (num_frames, P), dtype=torch.bool, device=device
            )

            for t in range(num_frames):
                all_windows_valid = torch.ones(P, dtype=torch.bool, device=device)

                for w_idx, w_len in enumerate(windows):
                    feat, freqs, valid = extract_alft_window_batched(
                        Y_chunk, M_chunk, frac_years_gpu, t,
                        w_len, k_frequencies, Omega, min_samples, device
                    )
                    offset = w_idx * fpw
                    chunk_features[t, :, offset:offset + fpw] = feat
                    chunk_freqs[t, :, w_idx, :] = freqs
                    # If ANY window fails, mark entire multi-scale token invalid
                    all_windows_valid &= valid

                chunk_valid[t, :] = all_windows_valid

            # Write back to CPU output arrays
            cf_cpu = chunk_features.cpu().numpy().reshape(
                num_frames, chunk_h, chunk_w, alft_dim
            )
            cfr_cpu = chunk_freqs.cpu().numpy().reshape(
                num_frames, chunk_h, chunk_w, len(windows), k_frequencies
            )
            cv_cpu = chunk_valid.cpu().numpy().reshape(
                num_frames, chunk_h, chunk_w
            )
            alft_features[:, y_start:y_end, x_start:x_end, :] = cf_cpu
            alft_freqs[:, y_start:y_end, x_start:x_end, :, :] = cfr_cpu
            alft_valid[:, y_start:y_end, x_start:x_end] = cv_cpu

            pbar.update(1)

    pbar.close()

    valid_count = alft_valid.sum()
    total_tokens = num_frames * height * width
    print(f"Valid ALFT tokens: {valid_count:,} / {total_tokens:,} "
          f"({100.0 * valid_count / total_tokens:.1f}%)")

    return alft_features, alft_freqs, alft_valid

def train_svdd(model, alft_features, alft_valid, frac_years,
               train_end_frac_year, l_max, alft_dim, stride, 
               batch_size, center_init_samples, epochs, lr, weight_decay,
               weights_path, device):
    """
    Trains the Deep SVDD model on pre-change (normal) sequences.
    """
    dataset = ALFTSequenceDataset(
        alft_features, alft_valid, frac_years,
        train_end_frac_year, l_max, alft_dim, stride=stride
    )

    if len(dataset) == 0:
        raise ValueError(
            "No valid training sequences found. "
            "Check TRAIN_END_DATE and data quality."
        )

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )

    # ── Step 1: Initialize Hypersphere Center ──
    print("\nPhase 2a: Initializing hypersphere center...")
    model.eval()

    center_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    all_embeddings = []
    n_collected = 0
    with torch.no_grad():
        for batch in center_loader:
            feat = batch['features'].to(device)
            times = batch['times'].to(device)
            mask = batch['padding_mask'].to(device)
            z = model.get_embeddings(feat, times, mask)
            all_embeddings.append(z)
            n_collected += z.size(0)
            if n_collected >= center_init_samples:
                break

    all_embeddings = torch.cat(all_embeddings, dim=0)
    center = all_embeddings.mean(dim=0)
    model.hypersphere_center.copy_(center)
    print(f"Center initialized from {len(all_embeddings):,} embeddings")

    # ── Step 2: Train ──
    print(f"\nPhase 2b: Training Deep SVDD ({epochs} epochs)...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_samples = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}",
                          leave=False):
            feat = batch['features'].to(device, non_blocking=True)
            times = batch['times'].to(device, non_blocking=True)
            mask = batch['padding_mask'].to(device, non_blocking=True)

            optimizer.zero_grad()
            scores = model(feat, times, mask)
            loss = scores.mean()  # Deep SVDD: minimize mean distance to center
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * feat.size(0)
            n_samples += feat.size(0)

        avg_loss = epoch_loss / n_samples
        print(f"Epoch {epoch + 1}/{epochs}, SVDD Loss: {avg_loss:.6f}")

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")


def run_inference(model, alft_features, alft_valid, frac_years, acq_times, 
                  l_max, alft_dim, chunk_size, inference_batch, 
                  warning_sigma, drift_sigma, consecutive_anomalies, 
                  ema_alpha, warmup_period, device,
                  enable_xai=False, windows=None, features_per_window=None):
    """
    Full retrospective inference with streaming drift detection.

    When enable_xai=True, computes PnPXAI IntegratedGradients attributions
    for all WARNING and DRIFT timesteps. Attributions are aggregated per
    ALFT window to quantify each temporal scale's contribution to the
    anomaly score.

    Args:
        windows: list of window lengths (e.g. [0.5, 1.0, 3.0]). Required
                 when enable_xai=True. Length determines the number of
                 attribution buckets.
        features_per_window: int, features per window (2*K+2). Required
                             when enable_xai=True.
    """
    num_frames, height, width = alft_valid.shape
    frac_years_f32 = frac_years.astype(np.float32)
    num_windows = len(windows) if windows is not None else 0

    # Output arrays (spatial map format)
    score_map = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    status_map = np.zeros((num_frames, height, width), dtype=np.uint8)
    first_drift_ts = np.full((height, width), np.nan, dtype=np.float64)
    drift_count_map = np.zeros((height, width), dtype=np.int32)

    # XAI output arrays (only allocated when enabled)
    if enable_xai:
        window_attr_map = np.full(
            (num_frames, height, width, num_windows), np.nan, dtype=np.float32
        )
        xai_sensitivity_map = np.full(
            (num_frames, height, width), np.nan, dtype=np.float32
        )
        xai_mu_fidelity_map = np.full(
            (num_frames, height, width), np.nan, dtype=np.float32
        )
        xai_complexity_map = np.full(
            (num_frames, height, width), np.nan, dtype=np.float32
        )

    model.eval()

    y_chunks = list(range(0, height, chunk_size))
    x_chunks = list(range(0, width, chunk_size))
    total_chunks = len(y_chunks) * len(x_chunks)

    print(f"\nPhase 3: Retrospective Inference ({total_chunks} spatial chunks)...")
    if enable_xai:
        print(f"  XAI enabled: IntegratedGradients + [Sensitivity, MuFidelity, Complexity]")
        print(f"  Attribution buckets: {num_windows} windows ({windows})")
    pbar = tqdm(total=total_chunks, desc="Inference")

    with torch.no_grad():
        for y_start in y_chunks:
            y_end = min(y_start + chunk_size, height)
            for x_start in x_chunks:
                x_end = min(x_start + chunk_size, width)

                ch = y_end - y_start
                cw = x_end - x_start
                P = ch * cw

                # Flatten spatial chunk
                c_feat = alft_features[
                    :, y_start:y_end, x_start:x_end, :
                ].reshape(num_frames, P, alft_dim)
                c_valid = alft_valid[
                    :, y_start:y_end, x_start:x_end
                ].reshape(num_frames, P)

                # Initialize drift detector for this chunk
                det = BatchedStreamingDriftDetector(
                    P, warning_sigma=warning_sigma, drift_sigma=drift_sigma, 
                    consecutive_anomalies=consecutive_anomalies, alpha=ema_alpha, 
                    warmup_period=warmup_period, device=device
                )

                # Local output buffers
                l_scores = np.full((num_frames, P), np.nan, dtype=np.float32)
                l_status = np.zeros((num_frames, P), dtype=np.uint8)
                l_first_drift = np.full(P, np.nan, dtype=np.float64)
                l_drift_count = np.zeros(P, dtype=np.int32)

                if enable_xai:
                    l_window_attr = np.full(
                        (num_frames, P, num_windows), np.nan, dtype=np.float32
                    )
                    l_xai_sens = np.full((num_frames, P), np.nan, dtype=np.float32)
                    l_xai_mufid = np.full((num_frames, P), np.nan, dtype=np.float32)
                    l_xai_comp = np.full((num_frames, P), np.nan, dtype=np.float32)
                    # Collect (timestep, pixel_indices, assembled tensors)
                    # for deferred XAI pass after the no_grad scoring loop
                    xai_targets = []

                for t in range(num_frames):
                    # ── Assemble L_MAX-length sequences ──
                    seq_start = max(0, t - l_max + 1)
                    actual_len = t - seq_start + 1
                    pad_len = l_max - actual_len

                    # Slices from pre-computed arrays
                    feat_slice = c_feat[seq_start:t + 1, :, :]   # (actual_len, P, ALFT_DIM)
                    valid_slice = c_valid[seq_start:t + 1, :]     # (actual_len, P)

                    # Shared time vector (same for all pixels at this timestep)
                    time_vec = np.zeros(l_max, dtype=np.float32)
                    time_vec[pad_len:] = frac_years_f32[seq_start:t + 1]
                    T_seq = torch.from_numpy(time_vec).float().view(
                        1, l_max, 1
                    ).to(device)

                    # Identify pixels with valid target tokens
                    target_valid_np = c_valid[t, :]
                    valid_pix = np.where(target_valid_np)[0]

                    raw_scores = torch.full((P,), float('nan'), device=device)

                    if len(valid_pix) > 0:
                        # Sub-batched forward pass for GPU memory management
                        for b_start in range(0, len(valid_pix), inference_batch):
                            b_end = min(
                                b_start + inference_batch, len(valid_pix)
                            )
                            pix = valid_pix[b_start:b_end]
                            bs = len(pix)

                            # Build padded sequences for sub-batch
                            feat_sub = feat_slice[:, pix, :]  # (actual_len, bs, ALFT_DIM)
                            valid_sub = valid_slice[:, pix]    # (actual_len, bs)

                            if pad_len > 0:
                                feat_padded = np.full(
                                    (l_max, bs, alft_dim), np.nan,
                                    dtype=np.float32
                                )
                                feat_padded[pad_len:, :, :] = feat_sub
                                valid_padded = np.zeros(
                                    (l_max, bs), dtype=bool
                                )
                                valid_padded[pad_len:, :] = valid_sub
                            else:
                                feat_padded = feat_sub
                                valid_padded = valid_sub

                            # → (bs, L_MAX, ALFT_DIM)
                            f_b = torch.tensor(
                                np.ascontiguousarray(
                                    feat_padded.transpose(1, 0, 2)
                                ),
                                dtype=torch.float32, device=device
                            )
                            # Padding mask: True = ignore
                            m_b = torch.tensor(
                                np.ascontiguousarray(~valid_padded.T),
                                dtype=torch.bool, device=device
                            )
                            t_b = T_seq.expand(bs, -1, -1)

                            raw_scores[pix] = model(f_b, t_b, m_b)

                    # ── Update Drift Detector ──
                    tv_gpu = torch.from_numpy(target_valid_np).to(device)
                    # Replace NaN with 0 for detector (masked by tv_gpu)
                    scores_clean = torch.nan_to_num(raw_scores, nan=0.0)
                    status = det.update(scores_clean, tv_gpu)

                    # Handle confirmed drifts
                    drifts = (
                        status == BatchedStreamingDriftDetector.STATUS_DRIFT
                    )
                    if drifts.any():
                        det.reset_model_adaptation(drifts)

                    # Store results
                    l_scores[t, :] = raw_scores.cpu().numpy()
                    l_status[t, :] = status.cpu().numpy().astype(np.uint8)

                    drift_cpu = drifts.cpu().numpy()
                    new_drift = drift_cpu & np.isnan(l_first_drift)
                    l_first_drift[new_drift] = acq_times[t]
                    l_drift_count[drift_cpu] += 1

                    # ── Collect XAI targets (WARNING or DRIFT pixels) ──
                    if enable_xai:
                        status_np = l_status[t, :]
                        anom_pix = np.where(
                            (status_np == BatchedStreamingDriftDetector.STATUS_WARNING) |
                            (status_np == BatchedStreamingDriftDetector.STATUS_DRIFT)
                        )[0]
                        if len(anom_pix) > 0:
                            # Store the sequence assembly info for deferred XAI
                            xai_targets.append((
                                t, anom_pix, seq_start, actual_len, pad_len
                            ))

                # ── Deferred XAI Attribution Pass ──
                if enable_xai and len(xai_targets) > 0:
                    torch.set_grad_enabled(True)
                    total_xai = sum(len(entry[1]) for entry in xai_targets)
                    xai_pbar = tqdm(
                        total=total_xai,
                        desc=f"  XAI chunk ({y_start},{x_start})",
                        leave=False
                    )

                    for t, anom_pix, seq_start, actual_len, pad_len in xai_targets:
                        # Re-assemble sequences for anomalous pixels
                        feat_slice = c_feat[seq_start:seq_start + actual_len, :, :]
                        valid_slice = c_valid[seq_start:seq_start + actual_len, :]

                        time_vec = np.zeros(l_max, dtype=np.float32)
                        time_vec[pad_len:] = frac_years_f32[seq_start:seq_start + actual_len]
                        T_seq_xai = torch.from_numpy(time_vec).float().view(
                            1, l_max, 1
                        ).to(device)

                        # Process one pixel at a time for Sensitivity metric
                        # (Sensitivity internally perturbs and re-runs, which
                        # requires consistent wrapper state per sample)
                        for pix_idx in anom_pix:
                            feat_sub = feat_slice[:, pix_idx:pix_idx+1, :]
                            valid_sub = valid_slice[:, pix_idx:pix_idx+1]

                            # Replace any NaNs in the invalid parts of feat_sub with 0.0 
                            # to prevent them from contaminating the gradients in XAI.
                            feat_sub_clean = np.nan_to_num(feat_sub, nan=0.0)

                            if pad_len > 0:
                                fp = np.zeros(
                                    (l_max, 1, alft_dim), dtype=np.float32
                                )
                                fp[pad_len:, :, :] = feat_sub_clean
                                vp = np.zeros((l_max, 1), dtype=bool)
                                vp[pad_len:, :] = valid_sub
                            else:
                                fp = feat_sub_clean
                                vp = valid_sub

                            # (1, L_MAX, ALFT_DIM)
                            x_xai = torch.tensor(
                                np.ascontiguousarray(fp.transpose(1, 0, 2)),
                                dtype=torch.float32, device=device
                            ).requires_grad_(True)

                            m_xai = torch.tensor(
                                np.ascontiguousarray(~vp.T),
                                dtype=torch.bool, device=device
                            )
                            t_xai = T_seq_xai.clone()

                            # Create per-sample wrapper
                            wrapper = PnPXAIWrapper(model, t_xai, m_xai)
                            explainer = IntegratedGradients(wrapper)

                            eval_target = torch.zeros(1, dtype=torch.long, device=device)

                            # Compute attributions
                            attr = explainer.attribute(
                                inputs=x_xai, targets=eval_target
                            )  # (1, L_MAX, ALFT_DIM)

                            # Aggregate attributions by window:
                            # Sum absolute attribution across sequence length,
                            # then partition by window feature indices
                            attr_np = attr.detach().cpu().numpy()  # (1, L_MAX, ALFT_DIM)
                            abs_attr_seq = np.sum(np.abs(attr_np[0]), axis=0)  # (ALFT_DIM,)

                            fpw = features_per_window
                            for w_idx in range(num_windows):
                                w_start = w_idx * fpw
                                w_end = w_start + fpw
                                l_window_attr[t, pix_idx, w_idx] = (
                                    np.sum(abs_attr_seq[w_start:w_end])
                                )

                            # Evaluate attribution quality
                            metric_sens = Sensitivity(
                                model=wrapper, explainer=explainer
                            )
                            metric_mufid = MuFidelity(
                                model=wrapper, explainer=explainer
                            )
                            metric_comp = Complexity(
                                model=wrapper, explainer=explainer
                            )

                            # Sensitivity calls explainer.attribute internally, so it needs gradients
                            l_xai_sens[t, pix_idx] = metric_sens.evaluate(
                                x_xai, targets=eval_target, attributions=attr
                            ).item()

                            # MuFidelity and Complexity only do forward passes or operations on the 
                            # already-computed attributions, so they don't need gradients.
                            with torch.no_grad():
                                l_xai_mufid[t, pix_idx] = metric_mufid.evaluate(
                                    x_xai, targets=eval_target, attributions=attr
                                ).item()
                                l_xai_comp[t, pix_idx] = metric_comp.evaluate(
                                    x_xai, targets=eval_target, attributions=attr
                                ).item()

                            xai_pbar.update(1)

                    xai_pbar.close()
                    torch.set_grad_enabled(False)

                # Write chunk to output arrays
                score_map[:, y_start:y_end, x_start:x_end] = (
                    l_scores.reshape(num_frames, ch, cw)
                )
                status_map[:, y_start:y_end, x_start:x_end] = (
                    l_status.reshape(num_frames, ch, cw)
                )
                first_drift_ts[y_start:y_end, x_start:x_end] = (
                    l_first_drift.reshape(ch, cw)
                )
                drift_count_map[y_start:y_end, x_start:x_end] = (
                    l_drift_count.reshape(ch, cw)
                )

                if enable_xai:
                    window_attr_map[:, y_start:y_end, x_start:x_end, :] = (
                        l_window_attr.reshape(num_frames, ch, cw, num_windows)
                    )
                    xai_sensitivity_map[:, y_start:y_end, x_start:x_end] = (
                        l_xai_sens.reshape(num_frames, ch, cw)
                    )
                    xai_mu_fidelity_map[:, y_start:y_end, x_start:x_end] = (
                        l_xai_mufid.reshape(num_frames, ch, cw)
                    )
                    xai_complexity_map[:, y_start:y_end, x_start:x_end] = (
                        l_xai_comp.reshape(num_frames, ch, cw)
                    )

                pbar.update(1)

    pbar.close()

    if enable_xai:
        return (score_map, status_map, first_drift_ts, drift_count_map,
                window_attr_map, xai_sensitivity_map, xai_mu_fidelity_map,
                xai_complexity_map)
    return score_map, status_map, first_drift_ts, drift_count_map
