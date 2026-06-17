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
        
        # Regression Head
        # Removed spatial_dim. Input is strictly the hidden_dim channels from GAP + target_features_dim
        self.linear1 = nn.Linear(self.hidden_dim + self.target_features_dim, 128)
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
        
        # Permute from (Batch, SeqLen, hidden_dim) to (Batch, hidden_dim, SeqLen) for GAP
        x = x.permute(0, 2, 1)
        
        # Global Average Pooling (masked)
        sum_x = torch.sum(x * m, dim=2)
        count_valid = torch.sum(m, dim=2).clamp(min=1e-5)
        x = sum_x / count_valid # (Batch, hidden_dim)
        
        x = torch.cat([x, X_targets], dim=1) # (Batch, hidden_dim + target_features_dim)
        
        # Regression Head
        # Concatenation of X_spatial removed
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x) 
        return x