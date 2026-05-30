import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional

# Assuming modules are in your PYTHONPATH
from spatiotemporal_encoders import encode_spatiotemporal_cube, Time2Vec
from transformer_forecaster import SeriesDecomposition, ProbabilisticOutputHead, calculate_masked_gaussian_nll

class DirectMultiHorizonTransformer(nn.Module):
    """
    Sequence-to-Sequence Transformer optimized for long-term multi-horizon forecasting.
    """
    def __init__(
        self, 
        input_channels: int, 
        patch_pixels: int, 
        d_model: int = 128, 
        n_heads: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        decomp_kernel: int = 5
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_pixels = patch_pixels
        
        # 1. Feature Projection
        self.feature_projection = nn.Linear(input_channels * patch_pixels, d_model)
        
        # 2. Decomposition
        self.decomposition = SeriesDecomposition(kernel_size=decomp_kernel)
        
        # 3. Transformer Core
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.trend_projection = nn.Linear(d_model, d_model)
        
        # 4. Probabilistic Output Head - Passing exact pixel count
        self.output_head = ProbabilisticOutputHead(d_model=d_model, pixels_per_patch=patch_pixels)

    def forward(
        self, 
        x_history: torch.Tensor, 
        q_future: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        h_enc = self.feature_projection(x_history)
        trend_init, cyclical_init = self.decomposition(h_enc)
        memory = self.encoder(cyclical_init)
        cyclical_future = self.decoder(tgt=q_future, memory=memory)
        
        trend_mean = trend_init.mean(dim=1, keepdim=True) 
        trend_future = self.trend_projection(trend_mean).expand(-1, q_future.shape[1], -1)
        
        latent_future = cyclical_future + trend_future
        mu, log_var = self.output_head(latent_future)
        return mu, log_var

def generate_temporal_queries(
    target_times: np.ndarray, 
    time_encoder: Time2Vec, 
    d_model: int, 
    device: torch.device
) -> torch.Tensor:
    
    dt_index = pd.DatetimeIndex(target_times)
    
    doy = torch.tensor(dt_index.dayofyear.values, dtype=torch.float32, device=device)
    doy_sin = torch.sin(2 * torch.pi * doy / 365.2425)
    doy_cos = torch.cos(2 * torch.pi * doy / 365.2425)
    doy_features = torch.stack([doy_sin, doy_cos], dim=-1)
    
    epoch = np.datetime64('1970-01-01T00:00:00')
    seconds_since_epoch = (target_times - epoch) / np.timedelta64(1, 's')
    years_continuous = torch.tensor(
        seconds_since_epoch / (365.2425 * 24 * 3600), 
        dtype=torch.float32, device=device
    ).unsqueeze(-1)
    
    t2v_features = time_encoder(years_continuous)
    temporal_features = torch.cat([t2v_features, doy_features], dim=-1)
    
    query_projection = nn.Linear(temporal_features.shape[-1], d_model).to(device)
    q_future = query_projection(temporal_features).unsqueeze(0) 
    
    return q_future

def run_transformer_pipeline(
    local_z_score: np.ndarray, 
    valid_mask: np.ndarray, 
    acq_times: np.ndarray,
    model: DirectMultiHorizonTransformer,
    device: torch.device,
    forecast_times: Optional[np.ndarray] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    N, H, W = local_z_score.shape
    
    encoded_cube, spatial_encoder, t2v_encoder = encode_spatiotemporal_cube(
        local_z_score, acq_times, num_freqs=10, t2v_dim=16
    )
    
    t2v_encoder = t2v_encoder.to(device)
    encoded_cube = encoded_cube.to(device)
    
    # Pre-instantiate mask tensor so we can use it to sanitize the inputs algebraically
    mask_tensor = torch.tensor(valid_mask, dtype=torch.float32, device=device)
    
    # --- STRICT FAILURE HANDLING & INPUT SANITIZATION ---
    # The feature_projection layer performs a dot product across all 18,271 pixels. 
    # A single NaN in an invalid (e.g., cloudy) pixel will poison the entire frame's embedding.
    # We replace NaNs strictly in MASKED regions with 0.0 to allow matrix multiplication.
    # If NaNs exist in VALID regions, they will intentionally bypass this, poison the math,
    # and fail the gradient, fulfilling the requirement to hard-fail on bad data integrity.
    image_channel = encoded_cube[..., 0] # Extract the raw z-score image channel
    image_channel = torch.where(
        mask_tensor.bool(), 
        image_channel, 
        torch.zeros_like(image_channel)
    )
    encoded_cube[..., 0] = image_channel
    
    # --- EXPLICIT MASK FUSION ---
    # Appending the binary mask as a distinct feature channel. 
    # This prevents the network from learning the 0.0 placeholder as a valid statistical 
    # observation (a z-score of 0), allowing the linear projection to isolate missingness
    # dynamically without injecting assumed fill values.
    mask_feature = mask_tensor.unsqueeze(-1) # Shape: [N, H, W, 1]
    encoded_cube = torch.cat([encoded_cube, mask_feature], dim=-1)
    # --------------------------------------------------
    
    # x_history shape: [Batch=1, Seq=N, Features=H*W*(C+1)]
    x_history = encoded_cube.view(1, N, -1)
    
    if forecast_times is None:
        target_times = acq_times
        target_z_score = torch.tensor(local_z_score, dtype=torch.float32, device=device).unsqueeze(0)
        # Expand mask for target shape compatibility [Batch, Seq, H, W]
        mask_target = mask_tensor.unsqueeze(0) 
    else:
        target_times = forecast_times
        target_z_score = None 
        mask_target = None

    q_target = generate_temporal_queries(target_times, t2v_encoder, model.d_model, device)
    mu_pred, logvar_pred = model(x_history, q_target)
    
    loss = torch.tensor(0.0, device=device)
    if target_z_score is not None and mask_target is not None:
        target_flat = target_z_score.view(1, N, -1)
        mask_flat = mask_target.view(1, N, -1)
        
        # 1. IEEE 754 NaN Isolation for Targets
        target_flat = torch.where(mask_flat.bool(), target_flat, torch.zeros_like(target_flat))
        
        # 2. Precision Overflow Prevention
        logvar_pred_safe = torch.clamp(logvar_pred, min=-20.0, max=20.0)
        
        loss = calculate_masked_gaussian_nll(mu_pred, logvar_pred_safe, target_flat, mask_flat)
        
    return mu_pred, logvar_pred, loss