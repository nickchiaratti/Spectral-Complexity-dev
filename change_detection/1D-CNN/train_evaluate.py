import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MultiScaleSITSNet
from dataset import SITSDataset
import numpy as np
import h5py
import os
from tqdm import tqdm

def enable_mc_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

def train_and_evaluate(h5_path, output_h5='inference_results.h5', weights_path='sits_baseline_weights_pre2024.pth', train_end_date="2024-01-01", skip_training=False, mc_samples=50, confidence_multiplier=3.0, consecutive_anomalies=3, time_window_years=3.0, enable_elastic_window=True, max_elastic_window_years=5.0, min_samples=38):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Calibration Dataset...")
    cal_dataset = SITSDataset(h5_path, mode='calibration', train_end_date=train_end_date,
                              consecutive_anomalies=consecutive_anomalies, time_window_years=time_window_years,
                              enable_elastic_window=enable_elastic_window, max_elastic_window_years=max_elastic_window_years,
                              min_samples=min_samples)
    
    if len(cal_dataset) == 0:
        print("No valid calibration data found.")
        return
        
    cal_loader = DataLoader(cal_dataset, batch_size=4096, shuffle=True, num_workers=16, pin_memory=True)
    
    in_channels = cal_dataset[0]['X_seq'].shape[-1]
    model = MultiScaleSITSNet(in_channels=in_channels, out_features=consecutive_anomalies).to(device)
    
    if skip_training and os.path.exists(weights_path):
        print(f"Skipping training. Loading existing weights from {weights_path}...")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.HuberLoss(delta=1.0)
        
        print(f"Training on {len(cal_dataset)} sequences...")
        model.train()
        epochs = 10 
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in tqdm(cal_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                X_seq = batch['X_seq'].to(device, non_blocking=True)
                X_spatial = batch['X_spatial'].to(device, non_blocking=True)
                seq_mask = batch['seq_mask'].to(device, non_blocking=True)
                y = batch['Y_target'].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                preds = model(X_seq, X_spatial, seq_mask)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_seq.size(0)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(cal_dataset):.4f}")
            
        torch.save(model.state_dict(), weights_path)
        print(f"Weights saved to {weights_path}")
    
    # Calculate Baseline Uncertainties
    print("Calculating Baseline Uncertainties (Per-Pixel)...")
    model.eval()
    model.apply(enable_mc_dropout)
    baseline_loader = DataLoader(cal_dataset, batch_size=4096, shuffle=False, num_workers=16, pin_memory=True)
    
    H, W = cal_dataset.h, cal_dataset.w
    pixel_residual_sq_sum = np.zeros((H, W), dtype=np.float64)
    pixel_residual_count = np.zeros((H, W), dtype=np.int32)
    pixel_epistemic_sum = np.zeros((H, W), dtype=np.float64)
    pixel_epistemic_count = np.zeros((H, W), dtype=np.int32)
    
    with torch.no_grad():
        for batch in tqdm(baseline_loader, desc="Calculating Baseline Uncertainties"):
            X_seq = batch['X_seq'].to(device, non_blocking=True)
            X_spatial = batch['X_spatial'].to(device, non_blocking=True)
            seq_mask = batch['seq_mask'].to(device, non_blocking=True)
            y = batch['Y_target'].to(device, non_blocking=True)
            batch_sz = X_seq.size(0)
            
            stoc_preds = torch.zeros((mc_samples, batch_sz, consecutive_anomalies), device=device)
            for i in range(mc_samples):
                stoc_preds[i] = model(X_seq, X_spatial, seq_mask)
            
            stds = stoc_preds.std(dim=0).cpu().numpy()
            mean_preds = stoc_preds.mean(dim=0)
            residuals = torch.abs(mean_preds - y).cpu().numpy()
            
            meta = batch['metadata']
            py = meta[0].numpy()
            px = meta[1].numpy()
            
            stds_mean = stds.mean(axis=1)
            res_sq_mean = (residuals**2).mean(axis=1)
            
            np.add.at(pixel_residual_sq_sum, (py, px), res_sq_mean)
            np.add.at(pixel_residual_count, (py, px), 1)
            np.add.at(pixel_epistemic_sum, (py, px), stds_mean)
            np.add.at(pixel_epistemic_count, (py, px), 1)
            
    with np.errstate(divide='ignore', invalid='ignore'):
        pixel_aleatoric_rmse = np.sqrt(pixel_residual_sq_sum / pixel_residual_count)
        pixel_epistemic_std = pixel_epistemic_sum / pixel_epistemic_count
        
    # Per your user rule, we will NOT apply a global smoothing fill value.
    # Pixels with 0 valid baseline sequences will remain NaN.
    
    # Phase 2: Full Inference

    print("Loading Evaluation Dataset...")
    eval_dataset = SITSDataset(h5_path, mode='all', train_end_date=train_end_date,
                               consecutive_anomalies=consecutive_anomalies, time_window_years=time_window_years,
                               enable_elastic_window=enable_elastic_window, max_elastic_window_years=max_elastic_window_years,
                               min_samples=min_samples)
    if len(eval_dataset) == 0:
        print("No evaluation data found.")
        return
        
    eval_loader = DataLoader(eval_dataset, batch_size=4096, shuffle=False, num_workers=16, pin_memory=True)
    
    print(f"Evaluating {len(eval_dataset)} sequences...")
    dt_fields = [
        ('Pixel_X', 'int32'),
        ('Pixel_Y', 'int32'),
        ('Timestamp_T21', 'float64'),
        ('Timestamp_T_Last', 'float64')
    ]
    for i in range(1, consecutive_anomalies + 1):
        dt_fields.append((f'Pred_{i}', 'float32'))
    for i in range(1, consecutive_anomalies + 1):
        dt_fields.append((f'Std_{i}', 'float32'))
    for i in range(1, consecutive_anomalies + 1):
        dt_fields.append((f'Actual_{i}', 'float32'))
    dt_fields.append(('Anomaly_Flag', 'uint8'))
    dt_fields.append(('Attr_DoY', 'float32'))
    dt_fields.append(('Attr_ToD', 'float32'))
    dt_fields.append(('Attr_dt', 'float32'))
    dt_fields.append(('Attr_ZScore', 'float32'))
    dt_fields.append(('Attr_Spatial', 'float32'))
    dt = np.dtype(dt_fields)

    total_anomalies = 0
    print(f"Saving inference results incrementally to {output_h5}...")
    with h5py.File(output_h5, 'w') as f:
        if 'inference_results' in f:
            del f['inference_results']
        dset = f.create_dataset('inference_results', shape=(len(eval_dataset),), dtype=dt)
        dset.attrs['train_end_date'] = str(train_end_date)
        dset.attrs['confidence_multiplier'] = float(confidence_multiplier)
        
        if 'baseline_aleatoric_map' in f:
            del f['baseline_aleatoric_map']
        if 'baseline_epistemic_map' in f:
            del f['baseline_epistemic_map']
            
        f.create_dataset('baseline_aleatoric_map', data=pixel_aleatoric_rmse.astype(np.float32))
        f.create_dataset('baseline_epistemic_map', data=pixel_epistemic_std.astype(np.float32))
        
        curr_idx = 0
        with torch.no_grad():
            model.eval()
            model.apply(enable_mc_dropout)
            
            for batch in tqdm(eval_loader, desc="Evaluating inference results"):
                X_seq = batch['X_seq'].to(device, non_blocking=True)
                X_spatial = batch['X_spatial'].to(device, non_blocking=True)
                seq_mask = batch['seq_mask'].to(device, non_blocking=True)
                y_tensor = batch['Y_target'].to(device, non_blocking=True)
                
                batch_size = X_seq.size(0)
                stochastic_preds = torch.zeros((mc_samples, batch_size, consecutive_anomalies), device=device)
                
                for i in range(mc_samples):
                    stochastic_preds[i] = model(X_seq, X_spatial, seq_mask)
                    
                mean_preds = stochastic_preds.mean(dim=0)
                std_preds = stochastic_preds.std(dim=0)
                
                meta = batch['metadata']
                py = meta[0].numpy()
                px = meta[1].numpy()
                ts21 = meta[2].numpy()
                ts_last = meta[3].numpy()
                
                batch_aleatoric = pixel_aleatoric_rmse[py, px][:, None]
                
                preds_np = mean_preds.cpu().numpy()
                stds_np = std_preds.cpu().numpy()
                actuals_np = y_tensor.cpu().numpy()
                
                residuals_np = np.abs(preds_np - actuals_np)
                
                # Bayesian Total Uncertainty Bound
                total_uncertainty = np.sqrt(stds_np**2 + batch_aleatoric**2)
                uncertainty_threshold = total_uncertainty * confidence_multiplier
                
                # Anomaly condition: Does the residual break the total uncertainty bound?
                # (np.isnan comparisons safely evaluate to False)
                is_anomaly = (residuals_np > uncertainty_threshold).any(axis=1)
                
                anomaly_flags = is_anomaly.astype(np.uint8)
                
                # Preserve raw reality: Assign 255 if baseline was NaN (missing data)
                has_nan_baseline = np.isnan(total_uncertainty).any(axis=1)
                anomaly_flags[has_nan_baseline] = 255
                
                # Only count true anomalies
                total_anomalies += np.sum(anomaly_flags == 1)
                
                # --- Captum GradientShap Attribution ---
                attr_doy = np.full(batch_size, np.nan, dtype=np.float32)
                attr_tod = np.full(batch_size, np.nan, dtype=np.float32)
                attr_dt = np.full(batch_size, np.nan, dtype=np.float32)
                attr_zscore = np.full(batch_size, np.nan, dtype=np.float32)
                attr_spatial = np.full(batch_size, np.nan, dtype=np.float32)
                
                anom_idx = np.where(anomaly_flags == 1)[0]
                if len(anom_idx) > 0:
                    if 'gradient_shap' not in locals():
                        from captum.attr import GradientShap
                        gradient_shap = GradientShap(model)
                        
                    X_seq_anom = X_seq[anom_idx]
                    X_spat_anom = X_spatial[anom_idx]
                    seq_mask_anom = seq_mask[anom_idx]
                    
                    base_seq = torch.zeros_like(X_seq_anom)
                    base_spat = torch.zeros_like(X_spat_anom)
                    
                    attrs = gradient_shap.attribute(inputs=(X_seq_anom, X_spat_anom),
                                                    baselines=(base_seq, base_spat),
                                                    additional_forward_args=(seq_mask_anom,),
                                                    target=0)
                    
                    attr_X_seq = attrs[0].cpu().numpy()
                    attr_X_spatial = attrs[1].cpu().numpy()
                    
                    abs_seq = np.sum(np.abs(attr_X_seq), axis=1) # Sum over SeqLen
                    
                    attr_doy[anom_idx] = np.sum(abs_seq[:, 0:2], axis=1)
                    attr_tod[anom_idx] = np.sum(abs_seq[:, 2:4], axis=1)
                    attr_dt[anom_idx] = np.sum(abs_seq[:, 4:-1], axis=1)
                    attr_zscore[anom_idx] = abs_seq[:, -1]
                    attr_spatial[anom_idx] = np.sum(np.abs(attr_X_spatial), axis=1)
                
                # Extracted metadata above
                
                batch_results = np.empty(batch_size, dtype=dt)
                batch_results['Pixel_X'] = px
                batch_results['Pixel_Y'] = py
                batch_results['Timestamp_T21'] = ts21
                batch_results['Timestamp_T_Last'] = ts_last
                
                for k in range(consecutive_anomalies):
                    batch_results[f'Pred_{k+1}'] = preds_np[:, k]
                    batch_results[f'Std_{k+1}'] = total_uncertainty[:, k]
                    batch_results[f'Actual_{k+1}'] = meta[4 + k].numpy()
                    
                batch_results['Anomaly_Flag'] = anomaly_flags
                batch_results['Attr_DoY'] = attr_doy
                batch_results['Attr_ToD'] = attr_tod
                batch_results['Attr_dt'] = attr_dt
                batch_results['Attr_ZScore'] = attr_zscore
                batch_results['Attr_Spatial'] = attr_spatial
                
                dset[curr_idx:curr_idx + batch_size] = batch_results
                curr_idx += batch_size

    anomaly_rate = (total_anomalies / len(eval_dataset)) * 100 if len(eval_dataset) > 0 else 0
    print("\n--- Evaluation Report ---")
    print(f"Total sequences evaluated: {len(eval_dataset)}")
    print(f"Total Anomalies flagged: {total_anomalies}")
    print(f"Anomaly rate: {anomaly_rate:.2f}%")
    print("Done!")

if __name__ == "__main__":
    h5_path = "C:/satelliteImagery/HLST30/HLST_Malibu_Harmonized_SC_EM-7_Norm-bandCount.h5"
    train_and_evaluate(h5_path)
