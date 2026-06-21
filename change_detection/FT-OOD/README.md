# FT-OOD: Fourier Transform Out-of-Distribution Anomaly Detection

A continuous change detection pipeline for irregularly sampled satellite time series that combines deterministic spectral analysis with deep latent sequence modeling.

## Architecture

This pipeline replaces the 1D-CNN Inception architecture (`../1D-CNN-general/`) which failed due to fixed Conv1d kernel lengths on irregularly sampled data. The ALFT encoder natively handles irregular sampling via the Non-Uniform Discrete Fourier Transform (NDFT) design matrix.

### Layer 1: Multi-Scale ALFT Encoder (Deterministic, Frozen)
- **Method**: Batched Orthogonal Matching Pursuit on NDFT matrices
- **Windows**: Configurable parallel continuous temporal windows (default: 0.5, 1.0, 3.0 years)
- **Output**: Per-window harmonic coefficients [β₀, β_cos, β_sin, RMSE] — fixed-dimension feature vector regardless of sampling density
- **Design Matrix**: `[1, cos(Ω₁t), ..., cos(Ωₖt), sin(Ω₁t), ..., sin(Ωₖt)]` (no linear trend term)
- **Logic**: Adapted from `dhr_main_pytorch.py` — identical NDFT + OMP + MAD-based robust RMSE

### Layer 2: Time2Vec Temporal Encoding (Learnable)
- **Input**: Absolute fractional year (continuous, not delta-based)
- **Output**: Learned periodic + linear temporal representation
- **Reference**: Kazemi et al. (2019). *Time2Vec: Learning a Vector Representation of Time.*

### Layer 3: Masked Transformer Encoder (Learnable)
- **Attention**: Bidirectional (retrospective evaluation, `is_causal=False`)
- **Masking**: `src_key_padding_mask` forces attention to ignore NaN-padded and invalid tokens
- **Sequence**: Fixed L_MAX window sliding chronologically through the time series

### Layer 4: Deep SVDD Projection Head (Learnable)
- **Objective**: One-class classification — map normal sequences near a fixed hypersphere center
- **Anomaly Score**: Squared Euclidean distance from center `||f(x) - c||²`
- **Reference**: Ruff et al. (2018). *Deep One-Class Classification.* ICML.

### Layer 5: Streaming Drift Detector (Statistical)
- **Method**: Exponential Moving Average (EMA) of anomaly scores per pixel
- **Warning**: Score > mean + 2σ (potential virtual drift)
- **Drift**: Score > mean + 3σ for N consecutive valid observations (confirmed structural change)
- **Anti-Poisoning**: Only Normal/Warmup scores update the EMA baseline
- **Adaptation**: On confirmed drift, statistics reset to track new environmental baseline

## Configuration

Key parameters in `FT_OOD_main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WINDOWS` | `[0.5, 1.0, 3.0]` | ALFT lookback windows in years |
| `K_FREQUENCIES` | `2` | Dominant frequencies per window |
| `L_MAX` | `256` | Maximum Transformer sequence length |
| `TRAIN_END_DATE` | `"2024-01-01"` | Temporal cutoff for training data |
| `WARNING_SIGMA` | `2.0` | Warning threshold multiplier |
| `DRIFT_SIGMA` | `3.0` | Drift threshold multiplier |
| `CONSECUTIVE_ANOMALIES` | `3` | Consecutive violations for confirmed drift |

## Execution

The pipeline runs in 3 phases from the IDE (no command-line execution required):

1. **ALFT Feature Extraction** — Pre-computes multi-scale harmonic features for all pixels/timesteps
2. **Deep SVDD Training** — Trains Transformer + SVDD head on pre-change (normal) sequences
3. **Retrospective Inference** — Slides through full time series, scoring anomalies and tracking drift

Set `SKIP_TRAIN = True` after initial training to skip Phase 2 on subsequent runs.

## Output

HDF5 file with spatial map format (matching DHR pipeline conventions):

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `anomaly_scores` | (T, H, W) | float32 | Raw SVDD distance scores |
| `drift_status` | (T, H, W) | uint8 | Status codes (0=Normal, 1=Warning, 2=Drift, 3=Warmup) |
| `first_drift_timestamp` | (H, W) | float64 | UNIX epoch of first confirmed drift |
| `drift_count` | (H, W) | int32 | Total drift events per pixel |

## References

- Kazemi, S. M., et al. (2019). Time2Vec: Learning a Vector Representation of Time. *arXiv:1907.05321*.
- Ruff, L., et al. (2018). Deep One-Class Classification. *ICML*.
- VanderPlas, J. T. (2018). Understanding the Lomb-Scargle Periodogram. *ApJS*, 236(1), 16.
- Zhu, Z., & Woodcock, C. E. (2014). Continuous Change Detection and Classification. *RSE*, 144, 152-171.
- Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. *ACL*.
- Sundararajan, M., et al. (2017). Axiomatic Attribution for Deep Networks. *ICML*.
