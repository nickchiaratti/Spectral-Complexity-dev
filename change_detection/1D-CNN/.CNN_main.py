import os
from train_evaluate import train_and_evaluate
from visualization import plot_spatial_anomaly_overlay

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==========================================
# IDE CONFIGURATION
# ==========================================
LOCATION = "Rochesterv2"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
OUTPUT_DIR = f"C:/satelliteImagery/HLST30/1D-CNN-{LOCATION}-TrainEnd2024"
TRAIN_END_DATE = "2024-01-01"
SKIP_TRAIN = False

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
