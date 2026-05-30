import torch
import torch.nn as nn
from typing import Tuple

class SeriesDecomposition(nn.Module):
    """
    Decomposes a latent sequence into trend and cyclical components.
    Reference: Wu et al. (2021) "Autoformer" (NeurIPS).
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_transpose = x.permute(0, 2, 1)
        trend = self.moving_avg(x_transpose)
        if trend.shape[2] > x_transpose.shape[2]:
            trend = trend[:, :, :-1]
            
        trend = trend.permute(0, 2, 1)
        cyclical = x - trend
        return trend, cyclical

class ProbabilisticOutputHead(nn.Module):
    """
    Projects latent features back to the 1D flattened pixel dimensions, 
    outputting parameters for a Gaussian distribution per pixel.
    """
    def __init__(self, d_model: int, pixels_per_patch: int):
        super().__init__()
        self.pixels_per_patch = pixels_per_patch
        
        # Outputs 2 values per pixel: Mean and Log-Variance
        self.projection = nn.Linear(d_model, self.pixels_per_patch * 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [Batch, Sequence_length, d_model]
        output = self.projection(x)
        
        # Reshape back to separate mean and log-variance
        # Shape: [Batch, Sequence, Pixels_per_patch, 2]
        output = output.view(x.shape[0], x.shape[1], self.pixels_per_patch, 2)
        
        mu = output[..., 0]
        log_var = output[..., 1]
        
        return mu, log_var

def calculate_masked_gaussian_nll(
    mu: torch.Tensor, 
    log_var: torch.Tensor, 
    target: torch.Tensor, 
    valid_mask: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the Gaussian Negative Log-Likelihood, strictly ignoring 
    pixels flagged as invalid in the observation.
    """
    squared_error = (target - mu) ** 2
    precision = torch.exp(-log_var) 
    
    pixel_loss = 0.5 * (squared_error * precision + log_var)
    masked_loss = pixel_loss * valid_mask
    
    valid_pixel_count = valid_mask.sum()
    
    if valid_pixel_count == 0:
        return torch.tensor(0.0, requires_grad=True).to(mu.device)
        
    return masked_loss.sum() / valid_pixel_count