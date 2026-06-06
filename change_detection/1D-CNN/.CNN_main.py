import os
from train_evaluate import train_and_evaluate
from visualization import plot_spatial_anomaly_overlay

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==========================================
# IDE CONFIGURATION
# ==========================================
H5_PATH = "C:/satelliteImagery/HLST30/HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-bandCount.h5"
OUTPUT_DIR = "c:/satelliteImagery/HLST30/1D-CNN-Rochesterv2-TrainEnd2023"
TRAIN_END_DATE = "2023-01-01"
SKIP_TRAIN = True

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    weights_path = os.path.join(OUTPUT_DIR, 'sits_baseline_weights_pre2024.pth')
    inference_h5 = os.path.join(OUTPUT_DIR, 'inference_results.h5')
    
    print(f"Starting pipeline: skip_training={SKIP_TRAIN}...")
    train_and_evaluate(H5_PATH, output_h5=inference_h5, weights_path=weights_path, train_end_date=TRAIN_END_DATE, skip_training=SKIP_TRAIN)
    if os.path.exists(inference_h5):
        print("Launching Spatial Anomaly Overlay visualization...")
        plot_spatial_anomaly_overlay(H5_PATH, inference_h5, TRAIN_END_DATE)
    else:
        print(f"Error: Inference results not found at {inference_h5}. Please run without SKIP_TRAIN=True first.")

if __name__ == "__main__":
    main()
