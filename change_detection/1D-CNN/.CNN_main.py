import os
from train_evaluate import train_and_evaluate
from visualization import plot_spatial_anomaly_overlay

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ==========================================
# IDE CONFIGURATION
# ==========================================
LOCATION = "Tait"
TRAIN_END_YEAR = "2025"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-bandCount.h5"
OUTPUT_DIR = f"C:/satelliteImagery/HLST30/1D-CNN-{LOCATION}-TrainEnd{TRAIN_END_YEAR}"
TRAIN_END_DATE = f"{TRAIN_END_YEAR}-01-01"
SKIP_TRAIN = False
MC_SAMPLES = 4
CONFIDENCE_MULTIPLIER = 3.0

CONSECUTIVE_ANOMALIES = 4
TIME_WINDOW_YEARS = 3.0
ENABLE_ELASTIC_WINDOW = False  # Allows window to expand backwards to meet MIN_SAMPLES
MAX_ELASTIC_WINDOW_YEARS = TIME_WINDOW_YEARS + 2.0  # Maximum span to expand backwards
MIN_SAMPLES = 38

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    weights_path = os.path.join(OUTPUT_DIR, f'CNN_{LOCATION}_baseline_weights_pre{TRAIN_END_YEAR}.pth')
    inference_h5 = os.path.join(OUTPUT_DIR, f'CNN_{LOCATION}_baseline_weights_pre{TRAIN_END_YEAR}.h5')
    
    print(f"Starting pipeline: skip_training={SKIP_TRAIN}...")
    train_and_evaluate(
        H5_PATH, 
        output_h5=inference_h5, 
        weights_path=weights_path, 
        train_end_date=TRAIN_END_DATE, 
        skip_training=SKIP_TRAIN,
        mc_samples=MC_SAMPLES,
        confidence_multiplier=CONFIDENCE_MULTIPLIER,
        consecutive_anomalies=CONSECUTIVE_ANOMALIES,
        time_window_years=TIME_WINDOW_YEARS,
        enable_elastic_window=ENABLE_ELASTIC_WINDOW,
        max_elastic_window_years=MAX_ELASTIC_WINDOW_YEARS,
        min_samples=MIN_SAMPLES
    )
    if os.path.exists(inference_h5):
        print("Launching Spatial Anomaly Overlay visualization...")
        plot_spatial_anomaly_overlay(H5_PATH, inference_h5)
    else:
        print(f"Error: Inference results not found at {inference_h5}. Please run without SKIP_TRAIN=True first.")

if __name__ == "__main__":
    main()
