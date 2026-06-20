import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import logging
import math

# Configure logging for data transformation transparency
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
FILL_VALUE = -9999.0

class HDF5TemporalWindowDataset(torch.utils.data.Dataset):
    """
    Loads spatial patches and temporal windows from an HDF5 dataset.
    Explicitly handles missing data via non-physical fill values.
    """
    def __init__(self, h5_filepath, patch_size=64, window_size=100, samples_per_epoch=1000):
        super().__init__()
        self.h5_filepath = h5_filepath
        self.patch_size = patch_size
        self.window_size = window_size
        self.samples_per_epoch = samples_per_epoch
        
        # Load into RAM to bypass HDF5 chunking bottlenecks during random access
        with h5py.File(self.h5_filepath, 'r') as f:
            raw_data = f['data'][:]       
            quality_mask = f['mask'][:]   
            self.timestamps = f['time'][:] 
            
        self.time_len, self.height, self.width = raw_data.shape
        
        # 1. Unify implicit NaNs and explicit quality masks
        nan_mask = np.isnan(raw_data)
        self.valid_mask = (~nan_mask) & (~quality_mask)
        
        # 2. Apply non-physical fill value explicitly
        raw_data[~self.valid_mask] = FILL_VALUE
        self.data = raw_data
        
        logging.info(f"DATA INTEGRITY LOG: Replaced {np.count_nonzero(~self.valid_mask)} "
                     f"invalid/NaN samples with fill value {FILL_VALUE}.")
        
        # 3. Time Normalization (required to prevent gradient explosion in Time2Vec)
        t_min, t_max = self.timestamps.min(), self.timestamps.max()
        self.norm_timestamps = (self.timestamps - t_min) / (t_max - t_min)
        logging.info(f"DATA TRANSFORMATION LOG: Normalized timestamps from "
                     f"[{t_min}, {t_max}] to [0.0, 1.0] for Time2Vec stability.")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        y = np.random.randint(0, self.height - self.patch_size)
        x = np.random.randint(0, self.width - self.patch_size)
        
        t_start = np.random.randint(0, self.time_len - self.window_size)
        t_end = t_start + self.window_size
        
        patch_data = self.data[t_start:t_end, y:y+self.patch_size, x:x+self.patch_size]
        patch_mask = self.valid_mask[t_start:t_end, y:y+self.patch_size, x:x+self.patch_size]
        window_timestamps = self.norm_timestamps[t_start:t_end]
        
        patch_data = torch.from_numpy(patch_data).unsqueeze(1).float()
        patch_mask = torch.from_numpy(patch_mask).unsqueeze(1).bool()
        timestamps = torch.from_numpy(window_timestamps).float()
        
        return timestamps, patch_data, patch_mask


class Time2Vec(nn.Module):
    """
    Continuous-time positional encoding.
    Ref: Kazemi, S. M., et al. (2019). "Time2Vec: Learning a Vector Representation of Time."
    """
    def __init__(self, out_features):
        super().__init__()
        self.w0 = nn.parameter.Parameter(torch.randn(1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))

    def forward(self, tau):
        tau = tau.unsqueeze(-1) 
        v1 = self.w0 * tau + self.b0
        v2 = torch.sin(self.w * tau + self.b)
        return torch.cat([v1, v2], dim=-1) 


class MaskedContinuousTransformer(nn.Module):
    """
    Continuous-Time Transformer optimized for Masked Reconstruction.
    Ref: Zerveas, G., et al. (2021). "A Transformer-based Framework for Multivariate Time Series Representation Learning."
    """
    def __init__(self, feature_dim=1, embed_dim=64, num_heads=4, num_layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.input_projection = nn.Linear(feature_dim, embed_dim)
        self.time2vec = Time2Vec(embed_dim)
        
        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim*4,
            batch_first=False 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Generative decoder to map the latent sequence back to the 1D physical feature space
        self.decoder = nn.Linear(embed_dim, feature_dim)

    def generate_block_mask(self, T, N, block_size=10, mask_ratio=0.2, device='cpu'):
        """
        Generates a boolean mask indicating which temporal indices should be artificially
        masked for reconstruction training. Applies masking in temporal blocks.
        """
        num_blocks = max(1, int((T * mask_ratio) // block_size))
        artificial_mask = torch.zeros(T, N, dtype=torch.bool, device=device)
        
        for _ in range(num_blocks):
            start_idx = torch.randint(0, T - block_size, (1,)).item()
            artificial_mask[start_idx:start_idx + block_size, :] = True
            
        return artificial_mask

    def forward(self, timestamps, data_seq, valid_mask_seq, training_mode=True):
        """
        data_seq: (T, N, 1)
        valid_mask_seq: (T, N, 1) - True where natural data is VALID.
        timestamps: (T)
        """
        T, N, C = data_seq.shape
        device = data_seq.device
        
        # 1. Nullify extreme fill values for numerical stability (FP16/AMP safety)
        # We replace the -9999.0 with 0.0 specifically to prevent linear projection overflow.
        # The src_key_padding_mask guarantees these tokens are completely ignored by the attention mechanism.
        safe_data_seq = data_seq.masked_fill(~valid_mask_seq, 0.0)
        
        # 2. Project input features
        x = self.input_projection(safe_data_seq) # (T, N, embed_dim)
        
        # 3. Apply Artificial Block Masking (Only during training)
        artificial_mask = None
        if training_mode:
            artificial_mask = self.generate_block_mask(T, N, block_size=8, mask_ratio=0.2, device=device)
            # Only apply artificial mask where data actually exists
            artificial_mask = artificial_mask & valid_mask_seq.squeeze(-1)
            
            # Replace selected blocks with the [MASK] token
            mask_expanded = self.mask_token.expand(T, N, -1)
            x[artificial_mask] = mask_expanded[artificial_mask]
        
        # 4. Add Continuous Temporal Embeddings
        t_expand = timestamps.unsqueeze(1).expand(T, N) 
        t_embed = self.time2vec(t_expand) 
        x = x + t_embed
        
        # 5. Prepend [CLS] token for spatial latent representation
        cls_tokens = self.cls_token.expand(1, N, -1) 
        x = torch.cat((cls_tokens, x), dim=0) # (T+1, N, embed_dim)
        
        # 6. Construct Attention Key Padding Mask for NATURALLY missing data
        # We must NOT pad out the artificial mask, as the model needs to predict it.
        # Invert valid_mask_seq to conform to PyTorch padding logic (True = ignore)
        natural_padding_mask = ~(valid_mask_seq.squeeze(-1).transpose(0, 1)) # (N, T)
        cls_pad = torch.zeros(N, 1, dtype=torch.bool, device=device)
        full_padding_mask = torch.cat((cls_pad, natural_padding_mask), dim=1) # (N, T+1)
        
        # 7. Transformer Pass
        out = self.transformer(x, src_key_padding_mask=full_padding_mask)
        
        # 8. Route outputs
        cls_representation = out[0, :, :] # (N, embed_dim) - Used for spatial correlation
        trajectory_out = out[1:, :, :]    # (T, N, embed_dim) - Used for sequence reconstruction
        
        # 9. Decode to feature space
        predictions = self.decoder(trajectory_out) # (T, N, 1)
        
        return predictions, cls_representation, artificial_mask


def compute_masked_loss(predictions, targets, artificial_mask, cls_representation, spatial_shape, lambda_spatial=0.1):
    """
    Computes Huber Loss strictly on artificially masked points, plus latent spatial regularization.
    
    predictions: (T, N, 1)
    targets: (T, N, 1)
    artificial_mask: (T, N) - True where we hid valid data.
    cls_representation: (N, embed_dim)
    spatial_shape: Tuple (H, W) corresponding to N
    """
    H, W = spatial_shape
    assert artificial_mask.any(), "CRITICAL: No artificial masks generated. Loss cannot be computed."
    
    # 1. Block Reconstruction Loss (Huber Loss)
    # We only compute loss where artificial_mask == True. 
    # Huber loss is robust to localized single-sample noise artifacts in the target array.
    unreduced_temporal_loss = F.smooth_l1_loss(predictions, targets, reduction='none')
    
    # Isolate loss to the artificially masked regions
    artificial_mask_expanded = artificial_mask.unsqueeze(-1)
    masked_temporal_loss = unreduced_temporal_loss.masked_fill(~artificial_mask_expanded, 0.0)
    
    reconstruction_loss = masked_temporal_loss.sum() / artificial_mask_expanded.sum()
    
    # 2. Latent Spatial Regularization (Graph Laplacian via Tensor Shifts)
    B = 1 # Assuming single patch processing for shape mechanics
    D = cls_representation.shape[-1]
    latent_grid = cls_representation.view(B, H, W, D).permute(0, 3, 1, 2)
    
    padded_Z = F.pad(latent_grid, (1, 1, 1, 1), mode='replicate')
    Z_up    = padded_Z[..., :-2, 1:-1]
    Z_down  = padded_Z[..., 2:, 1:-1]
    Z_left  = padded_Z[..., 1:-1, :-2]
    Z_right = padded_Z[..., 1:-1, 2:]
    
    spatial_loss = (
        F.mse_loss(latent_grid, Z_up) +
        F.mse_loss(latent_grid, Z_down) +
        F.mse_loss(latent_grid, Z_left) +
        F.mse_loss(latent_grid, Z_right)
    ) / 4.0
    
    total_loss = reconstruction_loss + (lambda_spatial * spatial_loss)
    return total_loss


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (timestamps, patch_data, patch_mask) in enumerate(dataloader):
        t_seq = timestamps[0].to(device) 
        B, T, C, H, W = patch_data.shape
        
        # Flatten spatial grid into sequence batch size N
        data_seq = patch_data.to(device).permute(1, 0, 3, 4, 2).reshape(T, B * H * W, 1)
        mask_seq = patch_mask.to(device).permute(1, 0, 3, 4, 2).reshape(T, B * H * W, 1)
        
        assert mask_seq.any(), "CRITICAL: Received batch with zero valid samples. Upstream pipeline failure."
        
        optimizer.zero_grad()
        
        # Forward pass applies dynamic artificial block masking
        predictions, cls_repr, artificial_mask = model(t_seq, data_seq, mask_seq, training_mode=True)
        
        # Compute loss
        loss = compute_masked_loss(
            predictions, 
            targets=data_seq, 
            artificial_mask=artificial_mask, 
            cls_representation=cls_repr, 
            spatial_shape=(H, W),
            lambda_spatial=0.1
        )
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)