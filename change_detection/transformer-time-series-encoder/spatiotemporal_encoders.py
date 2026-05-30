import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Any

class FourierSpatialEncoding(nn.Module):
    """
    Maps 2D spatial coordinates to high-frequency Fourier features.
    
    This technique mitigates the spectral bias of Multilayer Perceptrons (MLPs), 
    allowing Implicit Neural Representations to learn high-frequency spatial 
    patterns (e.g., fine-grained satellite imagery features).
    
    Reference:
        Tancik et al. (2020) "Fourier Features Let Networks Learn 
        High Frequency Functions in Low Dimensional Domains" (NeurIPS).
    """
    def __init__(self, num_frequencies: int = 10):
        super().__init__()
        # Generates frequencies: [1, 2, 4, 8, ... 2^(L-1)] * pi
        # Registered as a buffer so it moves to the correct device with the module
        frequencies = 2 ** torch.arange(num_frequencies) * torch.pi
        self.register_buffer('frequencies', frequencies)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coordinates: Tensor of shape (..., 2) containing normalized 
                         spatial coordinates in the range [-1, 1].
        Returns:
            Tensor of shape (..., 4 * num_frequencies) containing concatenated 
            sine and cosine features for both spatial dimensions.
        """
        encodings = []
        for freq in self.frequencies:
            encodings.append(torch.sin(coordinates * freq))
            encodings.append(torch.cos(coordinates * freq))
        return torch.cat(encodings, dim=-1)


class Time2Vec(nn.Module):
    """
    Learns a vector representation of continuous time.
    
    This layer creates a hybrid representation of time containing one linear
    non-periodic feature (for secular trends) and multiple periodic features 
    (for latent cycles), where frequencies and phase shifts are learned 
    directly from the data.
    
    Reference:
        Kazemi et al. (2019) "Time2Vec: Learning a Vector Representation of Time"
    """
    def __init__(self, time_dim: int = 1, embed_dim: int = 16):
        super().__init__()
        # Linear component parameters (trend)
        self.w0 = nn.Parameter(torch.randn(time_dim, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        
        # Periodic component parameters (cycles)
        self.w = nn.Parameter(torch.randn(time_dim, embed_dim - 1))
        self.b = nn.Parameter(torch.randn(embed_dim - 1))

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tau: Tensor of shape (..., time_dim) representing continuous time 
                 (e.g., fractional years).
        Returns:
            Tensor of shape (..., embed_dim) representing the learned temporal embedding.
        """
        linear = tau @ self.w0 + self.b0
        periodic = torch.sin(tau @ self.w + self.b)
        return torch.cat([linear, periodic], dim=-1)


def encode_spatiotemporal_cube(
    image_cube: np.ndarray, 
    timestamps_utc: np.ndarray, 
    num_freqs: int = 10, 
    t2v_dim: int = 16
) -> Tuple[torch.Tensor, FourierSpatialEncoding, Time2Vec]:
    """
    Encodes an image cube and irregularly sampled temporal metadata into a 
    feature-rich tensor suitable for Implicit Neural Representations.
    
    Note on failure handling: Missing dates (NaT) or invalid pixel data (NaN) 
    are intentionally propagated as NaN float values in the resulting tensor 
    to prevent artificial assumptions from contaminating the data analysis.
    
    Args:
        image_cube: numpy array of shape (N, H, W) containing monochromatic pixel values.
        timestamps_utc: numpy array of datetime64 objects of shape (N,).
        num_freqs: Number of frequency bands for spatial Fourier encoding.
        t2v_dim: Embedding dimension for the Time2Vec continuous time encoding.
        
    Returns:
        final_cube: Tensor of shape (N, H, W, C), where C is the concatenated feature dimension:
                    1 (image) + (4 * num_freqs) (spatial) + t2v_dim (continuous time) + 2 (DOY).
        spatial_encoder: The instantiated FourierSpatialEncoding module.
        t2v_encoder: The instantiated Time2Vec module.
    """
    N, H, W = image_cube.shape

    # ---------------------------------------------------------
    # 1. Spatial Encoding
    # ---------------------------------------------------------
    # Generate normalized coordinate grid [-1, 1]
    y = torch.linspace(-1, 1, steps=H)
    x = torch.linspace(-1, 1, steps=W)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([grid_y, grid_x], dim=-1) # Shape: (H, W, 2)

    spatial_encoder = FourierSpatialEncoding(num_frequencies=num_freqs)
    spatial_features = spatial_encoder(coords) # Shape: (H, W, 4 * num_freqs)

    # Broadcast to match the N dimension: (N, H, W, C_spatial)
    spatial_features = spatial_features.unsqueeze(0).expand(N, -1, -1, -1)

    # ---------------------------------------------------------
    # 2. Temporal Encoding (DOY + Continuous Time)
    # ---------------------------------------------------------
    dt_index = pd.DatetimeIndex(timestamps_utc)

    # A. DOY Trigonometric Encoding
    doy = torch.tensor(dt_index.dayofyear.values, dtype=torch.float32)
    doy_sin = torch.sin(2 * torch.pi * doy / 365.2425)
    doy_cos = torch.cos(2 * torch.pi * doy / 365.2425)
    doy_features = torch.stack([doy_sin, doy_cos], dim=-1) # Shape: (N, 2)

    # B. Continuous Time Mapping for Time2Vec
    epoch = np.datetime64('1970-01-01T00:00:00')
    seconds_since_epoch = (timestamps_utc - epoch) / np.timedelta64(1, 's')
    
    # Fractional years since epoch
    years_continuous = torch.tensor(
        seconds_since_epoch / (365.2425 * 24 * 3600), 
        dtype=torch.float32
    ).unsqueeze(-1) # Shape: (N, 1)

    t2v_encoder = Time2Vec(time_dim=1, embed_dim=t2v_dim)
    t2v_features = t2v_encoder(years_continuous) # Shape: (N, t2v_dim)

    # Combine temporal features: (N, t2v_dim + 2)
    temporal_features = torch.cat([t2v_features, doy_features], dim=-1)

    # Broadcast temporal features across the spatial grid: (N, H, W, C_temporal)
    temporal_features = temporal_features.view(N, 1, 1, -1).expand(-1, H, W, -1)

    # ---------------------------------------------------------
    # 3. Final Integration
    # ---------------------------------------------------------
    images_tensor = torch.tensor(image_cube, dtype=torch.float32).unsqueeze(-1)

    # Concatenate along the channel dimension
    final_cube = torch.cat([images_tensor, spatial_features, temporal_features], dim=-1)

    return final_cube, spatial_encoder, t2v_encoder