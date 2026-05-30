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
        
        self.flatten = nn.Flatten()
        
        # Regression Head
        # After maxpool twice, length will be 20 -> 10 -> 5.
        # Wait, maxpool of size 2, stride 2 (default). length 20 -> 10 -> 5.
        # Flattened dim = 64 channels * 5 = 320
        flattened_dim = 64 * 5
        spatial_dim = 40
        
        self.linear1 = nn.Linear(flattened_dim + spatial_dim, 128)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(128, 3)

    def forward(self, X_seq, X_spatial):
        # Permute X_seq from (Batch, 20, 6) to (Batch, 6, 20)
        x = X_seq.permute(0, 2, 1)
        
        # Inception branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        # Concatenate along channel dimension (dim=1)
        x = torch.cat([out1, out2, out3], dim=1) # (Batch, 48, 20)
        x = self.relu(x)
        x = self.maxpool1(x) # (Batch, 48, 10)
        
        # Secondary Extractor
        x = self.conv_sec(x)
        x = self.relu(x)
        x = self.maxpool2(x) # (Batch, 64, 5)
        
        x = self.flatten(x) # (Batch, 320)
        
        # Regression Head
        x = torch.cat([x, X_spatial], dim=1) # (Batch, 360)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) # (Batch, 3)
        return x
