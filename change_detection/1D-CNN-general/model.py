# model.py
import torch
import torch.nn as nn

class MultiScaleSITSNet(nn.Module):
    def __init__(self, out_features=3):
        super(MultiScaleSITSNet, self).__init__()
        
        # Inception Block
        self.branch1 = nn.Conv1d(in_channels=11, out_channels=16, kernel_size=3, padding='same')
        self.branch2 = nn.Conv1d(in_channels=11, out_channels=16, kernel_size=5, padding='same')
        self.branch3 = nn.Conv1d(in_channels=11, out_channels=16, kernel_size=7, padding='same')
        
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)
        
        # Secondary Extractor
        self.conv_sec = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=3, padding='same')
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        
        # Regression Head
        # Removed spatial_dim. Input is strictly the 64 channels from GAP.
        self.linear1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(128, out_features)

    def forward(self, X_seq, seq_mask=None): # X_spatial parameter removed
        # Permute X_seq from (Batch, SeqLen, 11) to (Batch, 11, SeqLen)
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
        
        # Secondary Extractor
        x = self.conv_sec(x)
        x = self.relu(x)
        x = x * m 
        
        x = self.maxpool2(x) 
        m = self.maxpool2(m)
        
        # Global Average Pooling (masked)
        sum_x = torch.sum(x * m, dim=2)
        count_valid = torch.sum(m, dim=2).clamp(min=1e-5)
        x = sum_x / count_valid # (Batch, 64)
        
        # Regression Head
        # Concatenation of X_spatial removed
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) 
        return x