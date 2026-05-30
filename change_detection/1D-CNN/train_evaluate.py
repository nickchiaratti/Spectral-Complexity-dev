import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MultiScaleSITSNet
from dataset import SITSDataset
import numpy as np
import h5py
import os

def train_and_evaluate(h5_path, output_h5='inference_results.h5', weights_path='sits_baseline_weights_pre2024.pth', train_end_date="2024-01-01"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Loading Calibration Dataset...")
    cal_dataset = SITSDataset(h5_path, mode='calibration', train_end_date=train_end_date)
    
    if len(cal_dataset) == 0:
        print("No valid calibration data found.")
        return
        
    cal_loader = DataLoader(cal_dataset, batch_size=4096, shuffle=True, num_workers=16, pin_memory=True)
    
    model = MultiScaleSITSNet().to(device)
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
            y = batch['Y_target'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            preds = model(X_seq, X_spatial)
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
            y = batch['Y_target'].to(device, non_blocking=True)
            preds = model(X_seq, X_spatial)
            sq_errs.append(((preds - y)**2).cpu().numpy())
    
    baseline_rmse = np.sqrt(np.mean(np.concatenate(sq_errs)))
    print(f"Baseline RMSE (Pre-2024): {baseline_rmse:.4f}")
    
    # Phase 2: Monitoring Inference
    print("Loading Monitoring Dataset...")
    mon_dataset = SITSDataset(h5_path, mode='monitoring', train_end_date=train_end_date)
    if len(mon_dataset) == 0:
        print("No monitoring data found.")
        return
        
    mon_loader = DataLoader(mon_dataset, batch_size=4096, shuffle=False, num_workers=16, pin_memory=True)
    
    print(f"Evaluating {len(mon_dataset)} monitoring sequences...")
    all_results = [] # list of tuples
    total_anomalies = 0
    with torch.no_grad():
        for batch in mon_loader:
            X_seq = batch['X_seq'].to(device, non_blocking=True)
            X_spatial = batch['X_spatial'].to(device, non_blocking=True)
            y = batch['Y_target'].numpy()
            
            preds = model(X_seq, X_spatial).cpu().numpy()
            
            for i in range(len(preds)):
                res = np.abs(preds[i] - y[i])
                mean_res = np.mean(res)
                is_anomaly = mean_res > (3.0 * baseline_rmse)
                if is_anomaly:
                    total_anomalies += 1
                
                meta = batch['metadata']
                # metadata elements: (y, x, ts21, ts23, actual1, actual2, actual3)
                py = meta[0][i].item()
                px = meta[1][i].item()
                ts21 = meta[2][i].item()
                ts23 = meta[3][i].item()
                act1 = meta[4][i].item()
                act2 = meta[5][i].item()
                act3 = meta[6][i].item()
                
                pr1, pr2, pr3 = preds[i]
                
                all_results.append((px, py, ts21, ts23, pr1, pr2, pr3, act1, act2, act3, mean_res, is_anomaly))
                
    anomaly_rate = (total_anomalies / len(mon_dataset)) * 100 if len(mon_dataset) > 0 else 0
    print("\n--- Monitoring Report ---")
    print(f"Baseline RMSE (Pre-2024): {baseline_rmse:.4f}")
    print(f"Total sequences evaluated in Monitoring Set: {len(mon_dataset)}")
    print(f"Total Anomalies flagged: {total_anomalies}")
    print(f"Anomaly rate: {anomaly_rate:.2f}%")
    
    # Save results to HDF5
    print(f"Saving inference results to {output_h5}...")
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
    
    arr = np.array(all_results, dtype=dt)
    with h5py.File(output_h5, 'w') as f:
        # Create dataset to maintain structure
        if 'inference_results' in f:
            del f['inference_results']
        dset = f.create_dataset('inference_results', data=arr)
        dset.attrs['train_end_date'] = str(train_end_date)
    print("Done!")

if __name__ == "__main__":
    h5_path = "C:/satelliteImagery/HLST30/HLST_Malibu_Harmonized_SC_EM-7_Norm-bandCount.h5"
    train_and_evaluate(h5_path)
