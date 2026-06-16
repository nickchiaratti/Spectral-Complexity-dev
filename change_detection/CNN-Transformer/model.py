# model.py
import torch
import torch.nn as nn

class MultiScaleSITSNet(nn.Module):
    def __init__(self, in_channels=15, out_features=3, target_features_dim=56):
        super(MultiScaleSITSNet, self).__init__()
        self.target_features_dim = target_features_dim
        
        # Transformer Block
        self.hidden_dim = 96
        self.feature_projection = nn.Linear(in_channels, self.hidden_dim)
        self.self_attention = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=4, 
            dim_feedforward=128, 
            batch_first=True,
            dropout=0.1
        )
        
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
        x = self.feature_projection(X_seq) # (Batch, SeqLen, hidden_dim)
        
        if seq_mask is not None:
            # PyTorch Transformer uses True for elements that should be IGNORED (padded)
            # seq_mask has 0.0 for padded elements.
            padding_mask = (seq_mask == 0.0)
            m = seq_mask.unsqueeze(1).float()
        else:
            padding_mask = None
            m = torch.ones(x.size(0), 1, x.size(1), device=x.device)
            
        # Temporal Self-Attention
        x = self.self_attention(x, src_key_padding_mask=padding_mask)
        
        # Permute from (Batch, SeqLen, hidden_dim) to (Batch, hidden_dim, SeqLen) for CNN Extractor
        x = x.permute(0, 2, 1)
        
        # Apply fractional temporal decay weights
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