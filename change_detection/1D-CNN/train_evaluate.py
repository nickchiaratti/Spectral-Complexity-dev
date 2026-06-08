import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MultiScaleSITSNet
from dataset import SITSDataset
import numpy as np
import h5py
import os

def train_and_evaluate(h5_path, output_h5='inference_results.h5', weights_path='sits_baseline_weights_pre2024.pth', train_end_date="2024-01-01", skip_training=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Calibration Dataset...")
    cal_dataset = SITSDataset(h5_path, mode='calibration', train_end_date=train_end_date)
    
    if len(cal_dataset) == 0:
        print("No valid calibration data found.")
        return
        
    cal_loader = DataLoader(cal_dataset, batch_size=4096, shuffle=True, num_workers=16, pin_memory=True)
    
    model = MultiScaleSITSNet().to(device)
    
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
            for batch in cal_loader:
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
    
    # Baseline RMSE Calculation
    print("Calculating Baseline RMSE...")
    model.eval()
    baseline_loader = DataLoader(cal_dataset, batch_size=4096, shuffle=False, num_workers=16, pin_memory=True)
    sq_errs = []
    with torch.no_grad():
        for batch in baseline_loader:
            X_seq = batch['X_seq'].to(device, non_blocking=True)
            X_spatial = batch['X_spatial'].to(device, non_blocking=True)
            seq_mask = batch['seq_mask'].to(device, non_blocking=True)
            y = batch['Y_target'].to(device, non_blocking=True)
            preds = model(X_seq, X_spatial, seq_mask)
            sq_errs.append(((preds - y)**2).cpu().numpy())
    
    baseline_rmse = np.sqrt(np.mean(np.concatenate(sq_errs)))
    print(f"Baseline RMSE (Pre-2024): {baseline_rmse:.4f}")
    
    # Phase 2: Full Inference
    print("Loading Evaluation Dataset...")
    eval_dataset = SITSDataset(h5_path, mode='all', train_end_date=train_end_date)
    if len(eval_dataset) == 0:
        print("No evaluation data found.")
        return
        
    eval_loader = DataLoader(eval_dataset, batch_size=4096, shuffle=False, num_workers=16, pin_memory=True)
    
    print(f"Evaluating {len(eval_dataset)} sequences...")
    dt = np.dtype([
        ('Pixel_X', 'int32'),
        ('Pixel_Y', 'int32'),
        ('Timestamp_T21', 'float64'),
        ('Timestamp_T23', 'float64'),
        ('Pred_1', 'float32'),
        ('Pred_2', 'float32'),
        ('Pred_3', 'float32'),
        ('Actual_1', 'float32'),
        ('Actual_2', 'float32'),
        ('Actual_3', 'float32'),
        ('Mean_Residual', 'float32'),
        ('Anomaly_Flag', 'uint8')
    ])

    total_anomalies = 0
    print(f"Saving inference results incrementally to {output_h5}...")
    with h5py.File(output_h5, 'w') as f:
        if 'inference_results' in f:
            del f['inference_results']
        dset = f.create_dataset('inference_results', shape=(len(eval_dataset),), dtype=dt)
        dset.attrs['train_end_date'] = str(train_end_date)
        
        curr_idx = 0
        with torch.no_grad():
            for batch in eval_loader:
                X_seq = batch['X_seq'].to(device, non_blocking=True)
                X_spatial = batch['X_spatial'].to(device, non_blocking=True)
                seq_mask = batch['seq_mask'].to(device, non_blocking=True)
                y = batch['Y_target'].numpy()
                
                preds = model(X_seq, X_spatial, seq_mask).cpu().numpy()
                batch_size = len(preds)
                
                res = np.abs(preds - y)
                mean_res = np.mean(res, axis=1)
                is_anomaly = mean_res > (3.0 * baseline_rmse)
                total_anomalies += np.sum(is_anomaly)
                
                meta = batch['metadata']
                py = meta[0].numpy()
                px = meta[1].numpy()
                ts21 = meta[2].numpy()
                ts23 = meta[3].numpy()
                act1 = meta[4].numpy()
                act2 = meta[5].numpy()
                act3 = meta[6].numpy()
                
                batch_results = np.empty(batch_size, dtype=dt)
                batch_results['Pixel_X'] = px
                batch_results['Pixel_Y'] = py
                batch_results['Timestamp_T21'] = ts21
                batch_results['Timestamp_T23'] = ts23
                batch_results['Pred_1'] = preds[:, 0]
                batch_results['Pred_2'] = preds[:, 1]
                batch_results['Pred_3'] = preds[:, 2]
                batch_results['Actual_1'] = act1
                batch_results['Actual_2'] = act2
                batch_results['Actual_3'] = act3
                batch_results['Mean_Residual'] = mean_res
                batch_results['Anomaly_Flag'] = is_anomaly.astype(np.uint8)
                
                dset[curr_idx:curr_idx + batch_size] = batch_results
                curr_idx += batch_size

    anomaly_rate = (total_anomalies / len(eval_dataset)) * 100 if len(eval_dataset) > 0 else 0
    print("\n--- Evaluation Report ---")
    print(f"Baseline RMSE (Pre-2024): {baseline_rmse:.4f}")
    print(f"Total sequences evaluated: {len(eval_dataset)}")
    print(f"Total Anomalies flagged: {total_anomalies}")
    print(f"Anomaly rate: {anomaly_rate:.2f}%")
    print("Done!")

if __name__ == "__main__":
    h5_path = "C:/satelliteImagery/HLST30/HLST_Malibu_Harmonized_SC_EM-7_Norm-bandCount.h5"
    train_and_evaluate(h5_path)
