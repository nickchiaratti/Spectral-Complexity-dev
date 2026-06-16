import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from dataset import SITSDataset
from model import MultiScaleSITSNet

# PnPXAI Imports
from pnpxai.explainers import IntegratedGradients, LRPEpsilonPlus, DeepLiftShap, AttentionRollout, GradientShap, SmoothGrad
from pnpxai.evaluator.metrics import Sensitivity, Complexity

# ==========================================
# BENCHMARK CONFIGURATION
# ==========================================
LOCATION = "Tait"
TRAIN_END_YEAR = "2025"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
WEIGHTS_PATH = f"C:/satelliteImagery/HLST30/CNN-Transformer-{LOCATION}-TrainEnd{TRAIN_END_YEAR}/CNN-Transformer_{LOCATION}_baseline_weights_pre{TRAIN_END_YEAR}.pth"
BATCH_SIZE = 16  # Small batch for heavy explainer benchmarking

class PnPXAIWrapper(nn.Module):
    def __init__(self, base_model, targets, mask):
        super().__init__()
        self.base_model = base_model
        self.targets = targets
        self.mask = mask
        
    def forward(self, x):
        batch_ratio = x.size(0) // self.targets.size(0)
        if batch_ratio > 1:
            t = self.targets.repeat(batch_ratio, *([1] * (self.targets.dim() - 1)))
            m = self.mask.repeat(batch_ratio, *([1] * (self.mask.dim() - 1)))
        else:
            t = self.targets
            m = self.mask
        return self.base_model(x, t, m)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(WEIGHTS_PATH):
        print(f"Error: Weights not found at {WEIGHTS_PATH}. Please wait for training to complete.")
        return

    print("Loading Dataset for Benchmarking...")
    # We load in monitoring mode just to grab a few valid sequences
    dataset = SITSDataset(
        H5_PATH, 
        mode='monitoring', 
        train_end_date=f"{TRAIN_END_YEAR}-01-01",
        consecutive_anomalies=4, 
        time_window_years=3.0,
        enable_elastic_window=True, 
        max_elastic_window_years=5.0, 
        min_samples=38
    )
    
    if len(dataset) == 0:
        print("No data found for benchmarking.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    batch = next(iter(loader))
    
    X_seq = batch['X_seq'].to(device)
    X_targets = batch['X_targets'].to(device)
    seq_mask = batch['seq_mask'].to(device)
    y_true = batch['Y_target'].to(device)
    
    in_channels = len(dataset.temporal_periods) * 2 + 8
    target_features_dim = X_targets.shape[-1]
    
    model = MultiScaleSITSNet(in_channels=in_channels, out_features=1, target_features_dim=target_features_dim).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model.eval()
    
    print("Model loaded. Wrapping for PnPXAI...")
    wrapped_model = PnPXAIWrapper(model, X_targets, seq_mask)
    eval_targets = torch.zeros(X_seq.size(0), dtype=torch.long, device=device)
    
    # DeepLiftShap and GradientShap require a baseline distribution at initialization
    baselines = torch.zeros_like(X_seq)
    
    # Define Explainers to Benchmark
    explainers = {
        "Integrated Gradients": IntegratedGradients(wrapped_model),
        "LRP Epsilon Plus": LRPEpsilonPlus(wrapped_model),
        "DeepLiftSHAP": DeepLiftShap(wrapped_model, baselines),
        "Attention Rollout": AttentionRollout(wrapped_model),
        "Gradient SHAP": GradientShap(wrapped_model, baselines),
        "SmoothGrad": SmoothGrad(wrapped_model)
    }
    
    results = {}
    
    print("\n--- Starting Benchmarking Suite ---")
    for name, explainer in explainers.items():
        print(f"Benchmarking {name}...")
        try:
            # 1. Compute Attributions
            attrs = explainer.attribute(inputs=X_seq, targets=eval_targets)
                
            # 2. Evaluate using Sensitivity and Complexity
            # These metrics evaluate how robust the explainer is to noise (Sensitivity)
            # and how sparse/interpretable the explanation is (Complexity)
            sens_scores = []
            comp_scores = []
            
            for i in range(len(X_seq)):
                x_i = X_seq[i:i+1]
                t_i = X_targets[i:i+1]
                m_i = seq_mask[i:i+1]
                a_i = attrs[i:i+1] if isinstance(attrs, torch.Tensor) else attrs[0][i:i+1]
                tgt_i = eval_targets[i:i+1]
                
                w_single = PnPXAIWrapper(model, t_i, m_i)
                
                if name in ["DeepLiftSHAP", "Gradient SHAP"]:
                    baselines_single = torch.zeros_like(x_i)
                    expl_single = explainers[name].__class__(w_single, baselines_single)
                else:
                    expl_single = explainers[name].__class__(w_single)
                
                metric_sens = Sensitivity(model=w_single, explainer=expl_single)
                metric_comp = Complexity(model=w_single, explainer=expl_single)
                
                sens = metric_sens.evaluate(x_i, targets=tgt_i, attributions=a_i).item()
                comp = metric_comp.evaluate(x_i, targets=tgt_i, attributions=a_i).item()
                
                sens_scores.append(sens)
                comp_scores.append(comp)
                
            results[name] = {
                "Sensitivity": np.nanmean(sens_scores),
                "Complexity": np.nanmean(comp_scores)
            }
            print(f"  -> Sensitivity (Lower is better): {results[name]['Sensitivity']:.4f}")
            print(f"  -> Complexity (Lower is better):  {results[name]['Complexity']:.4f}")
            
        except Exception as e:
            print(f"  -> [FAILED] {name} could not be evaluated: {str(e)}")
            
    print("\n==========================================")
    print(" BENCHMARK RESULTS (RANKED)")
    print("==========================================")
    
    # Rank by a combined score (both lower is better)
    print("\n--- Ranked by Sensitivity (Robustness) ---")
    ranked_sens = sorted(results.items(), key=lambda item: item[1]['Sensitivity'])
    for i, (name, metrics) in enumerate(ranked_sens):
        print(f"{i+1}. {name}: {metrics['Sensitivity']:.4f}")
        
    print("\n--- Ranked by Complexity (Interpretability) ---")
    ranked_comp = sorted(results.items(), key=lambda item: item[1]['Complexity'])
    for i, (name, metrics) in enumerate(ranked_comp):
        print(f"{i+1}. {name}: {metrics['Complexity']:.4f}")

if __name__ == "__main__":
    main()
