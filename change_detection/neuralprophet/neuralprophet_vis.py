import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

LOCATION = "Tait"
TARGET_METRIC = 'sliding_volume_z_score'
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-None.h5"
RESULTS_PATH = f"C:/satelliteImagery/HLST30/CCD/{LOCATION}_CCD_neuralprophet_Change_Detection_{TARGET_METRIC}.h5"

def on_click(event):
    if event.inaxes is None or event.button != 1:
        return
    x, y = int(round(event.xdata)), int(round(event.ydata))
    print(f"Clicked on pixel ({x}, {y})")
    
    with h5py.File(H5_PATH, 'r') as f_in, h5py.File(RESULTS_PATH, 'r') as f_out:
        data_grp = f_in['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        metric_ds = data_grp[TARGET_METRIC]
        acq_times = metric_ds.attrs['acquisition_time'][:]
        y_data = metric_ds[:, y, x]
        common_mask = data_grp['common_mask'][:, y, x]
        
        predicted = f_out['predicted_series'][:, y, x]
        rmse = f_out['rmse_series'][:, y, x] # We stored band width here
        anomaly_flags = f_out['anomaly_flags'][:, y, x]
        
    valid_mask = (common_mask == 0) & ~np.isnan(y_data)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) == 0:
        print("No valid data for this pixel.")
        return

    sig_acq = acq_times[valid_indices]
    sig_y = y_data[valid_indices]
    sig_pred = predicted[valid_indices]
    sig_rmse = rmse[valid_indices]
    sig_anom = anomaly_flags[valid_indices]
    
    sort_idx = np.argsort(sig_acq)
    sig_acq = sig_acq[sort_idx]
    sig_y = sig_y[sort_idx]
    sig_pred = sig_pred[sort_idx]
    sig_rmse = sig_rmse[sort_idx]
    sig_anom = sig_anom[sort_idx]
    
    dates = pd.to_datetime(sig_acq, unit='s')
    
    plt.figure(figsize=(12, 6))
    
    # Plot raw data
    plt.plot(dates, sig_y, 'o-', color='black', label=f'{TARGET_METRIC} (Observed)', alpha=0.5, markersize=4)
    
    # Plot prediction
    valid_pred = ~np.isnan(sig_pred)
    if valid_pred.any():
        plt.plot(dates[valid_pred], sig_pred[valid_pred], '-', color='blue', label='NeuralProphet Yhat')
        # Uncertainty band
        yhat_low = sig_pred[valid_pred] - (sig_rmse[valid_pred] / 2) # approx
        yhat_high = sig_pred[valid_pred] + (sig_rmse[valid_pred] / 2)
        plt.fill_between(dates[valid_pred], yhat_low, yhat_high, color='blue', alpha=0.2, label='90% Interval')
    
    # Plot anomalies
    anom_idx = np.where(sig_anom == 1)[0]
    if len(anom_idx) > 0:
        plt.plot(dates[anom_idx], sig_y[anom_idx], 'x', color='red', markersize=10, markeredgewidth=2, label='Detected Anomaly / Changepoint')
        for d in dates[anom_idx]:
            plt.axvline(d, color='red', linestyle='--', alpha=0.3)

    plt.title(f'NeuralProphet Anomaly Detection at ({x}, {y})')
    plt.xlabel('Date')
    plt.ylabel(TARGET_METRIC)
    plt.legend()
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    print(f"Loading {RESULTS_PATH}...")
    try:
        with h5py.File(RESULTS_PATH, 'r') as f_out:
            change_count = f_out['change_count'][:]
    except FileNotFoundError:
        print(f"Results file not found: {RESULTS_PATH}")
        print("Please run neuralprophet_main.py first.")
        return
        
    plt.figure(figsize=(10, 8))
    plt.imshow(change_count, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Number of Detected Events')
    plt.title('Click a pixel to view NeuralProphet Time Series Analysis')
    
    plt.gcf().canvas.mpl_connect('button_press_event', on_click)
    plt.show()

if __name__ == '__main__':
    main()
