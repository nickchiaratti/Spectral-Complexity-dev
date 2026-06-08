import torch
import torch.nn as nn

class MultiScaleSITSNet(nn.Module):
    def __init__(self):
        super(MultiScaleSITSNet, self).__init__()
        
        # Inception Block (Parallel Branches) Input: (Batch, 6, 20)
        # Features: [doy_sin, doy_cos, tod_sin, tod_cos, dt_log, z_score]
        self.branch1 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=3, padding='same')
        self.branch2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5, padding='same')
        self.branch3 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=7, padding='same')
        
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        # Secondary Extractor. Input channels = 16 * 3 = 48
        self.conv_sec = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, padding='same')
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        # Regression Head
        # Sequence length is now dynamic, so we use Global Average Pooling
        # resulting in a fixed 64 channels.
        spatial_dim = 40
        
        self.linear1 = nn.Linear(64 + spatial_dim, 128)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(128, 3)

    def forward(self, X_seq, X_spatial, seq_mask=None):
        # Permute X_seq from (Batch, SeqLen, 6) to (Batch, 6, SeqLen)
        x = X_seq.permute(0, 2, 1)
        
        if seq_mask is not None:
            # (Batch, SeqLen) -> (Batch, 1, SeqLen)
            m = seq_mask.unsqueeze(1).float()
        else:
            m = torch.ones(x.size(0), 1, x.size(2), device=x.device)
        
        # Inception branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        # Concatenate along channel dimension (dim=1)
        x = torch.cat([out1, out2, out3], dim=1) # (Batch, 48, SeqLen)
        x = self.relu(x)
        x = x * m # Apply mask before pooling
        
        x = self.maxpool1(x) 
        m = self.maxpool1(m) # Shrink the mask identically
        
        # Secondary Extractor
        x = self.conv_sec(x)
        x = self.relu(x)
        x = x * m # Apply mask again
        
        x = self.maxpool2(x) 
        m = self.maxpool2(m)
        
        # Global Average Pooling (masked)
        sum_x = torch.sum(x * m, dim=2)
        count_valid = torch.sum(m, dim=2).clamp(min=1e-5)
        x = sum_x / count_valid # (Batch, 64)
        
        # Regression Head
        x = torch.cat([x, X_spatial], dim=1) # (Batch, 64 + 40 = 104)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) # (Batch, 3)
        return x
