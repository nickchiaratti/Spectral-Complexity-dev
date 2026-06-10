import os
import sys
import yaml
import subprocess
from pathlib import Path

import HLS30.HLS30_earthAccess_to_hdf5 as HLS30_earthAccess_to_hdf5
import Harmonized_SC.HLST_constellation_to_hdf5 as HLST_constellation_to_hdf5
import Harmonized_SC.HLST_SC_calculations as HLST_SC_calculations
import Harmonized_SC.HLST_specComplex_viewer as HLST_specComplex_viewer

# ==========================================
# PIPELINE CONFIGURATION
# ==========================================
script_dir = Path(__file__).resolve().parent
# If TARGET_LOCATION is None, it will fall back to 'current_run' in the YAML
TARGET_LOCATION = 'Tait'
CONFIG_FILE_PATH = os.path.join(script_dir, "locations_config.yaml")
SKIP_VIEW = True  # Set to True to skip the interactive mgrs_view step
SATELLITE_DATA_DIR = "C:/satelliteImagery/HLST30"

def load_config(config_path="locations_config.yaml", location=None):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if location is None:
        location = config.get("current_run", {}).get("location")
    
    if not location or location not in config.get("locations", {}):
        print(f"Error: Location '{location}' not found in configuration.")
        sys.exit(1)
        
    loc_config = config["locations"][location]
    loc_config["LocationName"] = location
    return loc_config

def main():

    # Load configuration
    loc_config = load_config(CONFIG_FILE_PATH, TARGET_LOCATION)
    location_name = loc_config["LocationName"]
    print(f"Starting pipeline for location: {location_name}")
    
    # Save the current run to the yaml if it was passed explicitly
    if TARGET_LOCATION:
        with open(CONFIG_FILE_PATH, 'r') as f:
            full_config = yaml.safe_load(f)
        full_config.setdefault("current_run", {})["location"] = TARGET_LOCATION
        with open(CONFIG_FILE_PATH, 'w') as f:
            yaml.dump(full_config, f, sort_keys=False)

    # Define the execution order
    steps = [
        ("mgrs_view.py", None),
        ("HLS30_earthAccess_to_hdf5", HLS30_earthAccess_to_hdf5.main),
        ("HLST_constellation_to_hdf5", HLST_constellation_to_hdf5.main),
        ("HLST_SC_calculations", HLST_SC_calculations.main),
        ("HLST_specComplex_viewer", HLST_specComplex_viewer.main)
    ]

    for name, func in steps:
        print(f"\n{'='*50}")
        print(f"Executing: {name}")
        print(f"{'='*50}")
        
        if name == "mgrs_view.py":
            if SKIP_VIEW:
                print("Skipping mgrs_view.py as requested.")
                continue
            else:
                script_path = os.path.join(script_dir, "HLSX30", "mgrs_view.py")
                print(f"Launching Streamlit interface for {script_path}...")
                print("Please interact with the browser window, then click 'Close App & Continue Pipeline' to proceed.")
                subprocess.run([sys.executable, "-m", "streamlit", "run", script_path], check=True)
                continue
        elif name == "HLST_specComplex_viewer":
            print(f"Launching SpecComplex Viewer for {location_name}...")
            # Compute file path based on default naming convention in calculation script
            file_path = f"{SATELLITE_DATA_DIR}/HLST_{location_name}_Harmonized_SC_EM-7_Norm-bandCount.h5"
            
            # Extract start and end year from config
            start_date_str = loc_config.get("START_DATE", "2024-01-01")
            end_date_str = loc_config.get("END_DATE", "2025-01-01")
            start_year = int(start_date_str.split("-")[0])
            end_year = int(end_date_str.split("-")[0])
            
            func(target_location=TARGET_LOCATION, file_path=file_path, start_year=start_year, end_year=end_year)
            continue
            
        func(target_location=TARGET_LOCATION)

    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()
