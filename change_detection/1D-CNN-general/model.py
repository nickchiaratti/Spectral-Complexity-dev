# model.py
import torch
import torch.nn as nn

class MultiScaleSITSNet(nn.Module):
    def __init__(self, in_channels=15, out_features=3, target_features_dim=56):
        super(MultiScaleSITSNet, self).__init__()
        self.target_features_dim = target_features_dim
        
        # Inception Block
        self.branch1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding='same')
        self.branch2 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding='same')
        self.branch3 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=7, padding='same')
        
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        # Secondary Extractor
        self.conv_sec = nn.Conv1d(in_channels=96, out_channels=64, kernel_size=3, padding='same')
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        self.dropout1d = nn.Dropout1d(0.2)
        
        # Regression Head
        # Removed spatial_dim. Input is strictly the 64 channels from GAP + target_features_dim
        self.linear1 = nn.Linear(64 + self.target_features_dim, 128)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(128, out_features)

    def forward(self, X_seq, X_targets, seq_mask=None): # X_spatial parameter removed
        # Permute X_seq from (Batch, SeqLen, in_channels) to (Batch, in_channels, SeqLen)
        x = X_seq.permute(0, 2, 1)
        
        if seq_mask is not None:
            m = seq_mask.unsqueeze(1).float()
        else:
            m = torch.ones(x.size(0), 1, x.size(2), device=x.device)
        
        # Inception branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        x = torch.cat([out1, out2, out3], dim=1) 
        x = self.relu(x)
        x = x * m 
        
        x = self.maxpool1(x) 
        m = self.maxpool1(m) 
        x = self.dropout1d(x)
        
        # Secondary Extractor
        x = self.conv_sec(x)
        x = self.relu(x)
        x = x * m 
        
        x = self.maxpool2(x) 
        m = self.maxpool2(m)
        x = self.dropout1d(x)
        
        # Global Average Pooling (masked)
        sum_x = torch.sum(x * m, dim=2)
        count_valid = torch.sum(m, dim=2).clamp(min=1e-5)
        x = sum_x / count_valid # (Batch, 64)
        
        x = torch.cat([x, X_targets], dim=1) # (Batch, 64 + target_features_dim)
        
        # Regression Head
        # Concatenation of X_spatial removed
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) 
        return x