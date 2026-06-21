import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    """Continuous temporal encoding (Kazemi et al., 2019).

    Maps absolute fractional year to a learned periodic + linear representation.
    The linear component captures non-periodic temporal trends.
    The periodic components (sine activations) capture multi-scale temporal patterns.
    """
    def __init__(self, out_features):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(1, 1))
        self.p0 = nn.Parameter(torch.randn(1, 1))
        self.W = nn.Parameter(torch.randn(1, out_features - 1))
        self.P = nn.Parameter(torch.randn(1, out_features - 1))

    def forward(self, tau):
        """
        Args:
            tau: (Batch, L, 1) - absolute fractional year
        Returns:
            (Batch, L, out_features) - temporal embedding
        """
        v1 = tau * self.w0 + self.p0             # Linear component
        v2 = torch.sin(tau * self.W + self.P)    # Periodic components
        return torch.cat([v1, v2], dim=-1)

class OOD_Anomaly_Detector(nn.Module):
    """
    Transformer-based Out-of-Distribution anomaly detector with Deep SVDD head.

    Architecture:
        Input projection: Linear(alft_dim + time_dim -> d_model)
        Layer 2: Time2Vec continuous temporal encoding
        Layer 3: Masked Transformer Encoder (bidirectional for retrospective)
        Layer 4: Deep SVDD projection head -> distance from hypersphere center
        
    ─────────────────────────────────────────────────────────────────────────────
    EXPLAINABILITY INTEGRATION POINTS (deferred to future implementation):

    [ATTENTION ROLLOUT] (Abnar & Zuidema, 2020)
        After self.transformer forward pass, extract attention weights per layer:
            # Register forward hooks on each TransformerEncoderLayer's self_attn
            attn_weights = []
            for layer in self.transformer.layers:
                hook = layer.self_attn.register_forward_hook(capture_attention)
                attn_weights.append(hook_output)
            # Compute rollout: product of (0.5*I + 0.5*A_l) across layers
            # Result maps which historical tokens influenced the target anomaly score
            # Short-range attention -> transient noise
            # Long-range attention -> structural baseline violation

    [INTEGRATED GRADIENTS on SVDD HEAD] (Sundararajan et al., 2017)
        Compute path integral of gradients of anomaly_score w.r.t. ALFT features:
            # Custom target: score = ||f(x) - c||^2
            # baseline = mean ALFT vector from training set (stored during training)
            # attr = IntegratedGradients(model).attribute(
            #     inputs=X_alft, baselines=baseline_alft, target=svdd_distance
            # )
            # Per-feature attributions identify which harmonic coefficient
            # (beta_cos, beta_sin, RMSE from which window scale) drove the score

    [DETERMINISTIC HARMONIC RECONSTRUCTION]
        For anomalous tokens, reconstruct the periodic baseline from ALFT features:
            # Extract: beta_0, beta_cos, beta_sin per window from alft_features[t, y, x]
            # Reconstruct: y_hat(t) = beta_0 + sum_k(beta_cos_k*cos(Omega_k*t)
            #                                       + beta_sin_k*sin(Omega_k*t))
            # Plot y_hat(t) vs raw observations for visual diagnostic
            # NOTE: Requires storing per-pixel dominant frequencies (Omega_active)
            # from the ALFT extraction step. Currently not persisted.
    ─────────────────────────────────────────────────────────────────────────────
    """
    def __init__(self, alft_dim, time_dim, d_model=64, num_heads=8,
                 num_layers=4, svdd_dim=64):
        super().__init__()
        self.time_embed = Time2Vec(time_dim)

        raw_dim = alft_dim + time_dim
        self.input_proj = nn.Linear(raw_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Bidirectional attention (retrospective evaluation):
        # No causal mask is passed during forward(), so attention is naturally
        # bidirectional. For online/real-time detection, pass is_causal=True
        # to the forward() call instead.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Deep SVDD Projection Head
        # bias=False prevents hypersphere collapse (Ruff et al., 2018)
        self.svdd_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model // 2, svdd_dim, bias=False)
        )
        self.register_buffer('hypersphere_center', torch.zeros(svdd_dim))

    def forward(self, X_alft, T_seq, padding_mask):
        """
        Args:
            X_alft: (Batch, L_MAX, alft_dim) - ALFT features (NaN at padded positions)
            T_seq: (Batch, L_MAX, 1) - absolute fractional years
            padding_mask: (Batch, L_MAX) - True = ignore token

        Returns:
            anomaly_scores: (Batch,) - squared distance from hypersphere center
        """
        # Replace NaN with 0 for padded positions (masked in attention anyway)
        X_clean = torch.nan_to_num(X_alft, nan=0.0)

        T_emb = self.time_embed(T_seq)  # (Batch, L, time_dim)
        X_fused = torch.cat([X_clean, T_emb], dim=-1)

        X_proj = self.input_proj(X_fused)   # (Batch, L, d_model)
        X_normed = self.input_norm(X_proj)

        # src_key_padding_mask True = ignore token in attention
        latent_seq = self.transformer(
            X_normed, src_key_padding_mask=padding_mask
        )

        # Target token = last position in sequence (most recent timestep)
        target_latent = latent_seq[:, -1, :]  # (Batch, d_model)
        z = self.svdd_head(target_latent)      # (Batch, svdd_dim)

        anomaly_scores = torch.sum(
            (z - self.hypersphere_center) ** 2, dim=1
        )  # (Batch,)
        return anomaly_scores

    def get_embeddings(self, X_alft, T_seq, padding_mask):
        """Returns SVDD embeddings without scoring (for center initialization)."""
        X_clean = torch.nan_to_num(X_alft, nan=0.0)
        T_emb = self.time_embed(T_seq)
        X_fused = torch.cat([X_clean, T_emb], dim=-1)
        X_proj = self.input_proj(X_fused)
        X_normed = self.input_norm(X_proj)
        latent_seq = self.transformer(
            X_normed, src_key_padding_mask=padding_mask
        )
        target_latent = latent_seq[:, -1, :]
        z = self.svdd_head(target_latent)
        return z


class BatchedStreamingDriftDetector:
    """
    Dual-threshold (Warning/Drift) statistical detector using
    Exponential Moving Averages (EMA), optimized for batched GPU execution.

    Implements streaming concept drift detection (Lectures 11-12):
    - Warning signal: score > mean + WARNING_SIGMA * std
    - Drift signal: score > mean + DRIFT_SIGMA * std for
      CONSECUTIVE_ANOMALIES consecutive valid steps
    - Conditional EMA update: only Normal/Warmup scores update baseline
      (prevents anomalous values from poisoning the statistical tracker)
    - Model adaptation: reset statistics on confirmed Drift to begin
      tracking the new environmental baseline

    Status Codes:
        0 = Normal      (within baseline bounds)
        1 = Warning     (exceeds warning threshold, potential virtual drift)
        2 = Drift       (confirmed structural change after consecutive violations)
        3 = Warmup      (insufficient history for threshold computation)
    """
    STATUS_NORMAL = 0
    STATUS_WARNING = 1
    STATUS_DRIFT = 2
    STATUS_WARMUP = 3

    def __init__(self, batch_size, warning_sigma=2.0, drift_sigma=3.0, 
                 consecutive_anomalies=3, alpha=0.05, warmup_period=20, 
                 device='cuda'):
        self.batch_size = batch_size
        self.warning_sigma = warning_sigma
        self.drift_sigma = drift_sigma
        self.consecutive_anomalies = consecutive_anomalies
        self.alpha = alpha
        self.warmup_period = warmup_period

        self.means = torch.zeros(batch_size, device=device)
        self.stds = torch.full((batch_size,), 1e-5, device=device)
        self.drift_counters = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )
        self.step_counts = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )

    def update(self, scores, valid_mask):
        """
        Updates running statistics and returns drift status per pixel.

        Args:
            scores: (batch_size,) - positive OOD distance metrics
            valid_mask: (batch_size,) bool - True for pixels with valid target tokens

        Returns:
            status: (batch_size,) int32 - status codes (0-3)
        """
        # 1. Compute thresholds from current running statistics
        warning_thresh = self.means + (self.warning_sigma * self.stds)
        drift_thresh = self.means + (self.drift_sigma * self.stds)

        # 2. Evaluate current step
        is_warmup = self.step_counts < self.warmup_period
        is_drift_cand = scores > drift_thresh
        is_warning = (scores > warning_thresh) & (~is_drift_cand)
        is_normal = scores <= warning_thresh

        # 3. Update sequential drift counters
        # Reset counter if score drops below drift threshold
        self.drift_counters = torch.where(
            is_drift_cand & valid_mask & ~is_warmup,
            self.drift_counters + 1,
            torch.where(
                valid_mask,
                torch.zeros_like(self.drift_counters),
                self.drift_counters
            )
        )

        # 4. Assign final status flags
        status = torch.full(
            (self.batch_size,), self.STATUS_NORMAL,
            dtype=torch.int32, device=scores.device
        )
        status = torch.where(is_warning, self.STATUS_WARNING, status)
        status = torch.where(
            self.drift_counters >= self.consecutive_anomalies,
            self.STATUS_DRIFT, status
        )
        status = torch.where(is_warmup, self.STATUS_WARMUP, status)
        # Invalid pixels retain NORMAL status
        status = torch.where(~valid_mask, self.STATUS_NORMAL, status)

        # 5. Conditional EMA update: only Normal/Warmup scores update baseline
        # This prevents streaming noise or confirmed drifts from corrupting
        # the statistical baseline.
        update_mask = (is_normal | is_warmup) & valid_mask

        diff = scores - self.means
        new_means = self.means + self.alpha * diff
        new_stds = torch.sqrt(
            (1 - self.alpha) * self.stds ** 2 + self.alpha * diff ** 2
        )

        self.means = torch.where(update_mask, new_means, self.means)
        self.stds = torch.where(update_mask, new_stds, self.stds)

        self.step_counts += valid_mask.int()

        return status

    def reset_model_adaptation(self, reset_mask):
        """
        Triggered on confirmed True Concept Drift to reset statistical
        tracking for specific pixels, allowing them to adapt to a new baseline.
        """
        zero_f = torch.zeros_like(self.means)
        one_f = torch.ones_like(self.stds)
        zero_i = torch.zeros_like(self.drift_counters)

        self.means = torch.where(reset_mask, zero_f, self.means)
        self.stds = torch.where(reset_mask, one_f, self.stds)
        self.drift_counters = torch.where(reset_mask, zero_i, self.drift_counters)
        self.step_counts = torch.where(reset_mask, zero_i, self.step_counts)
