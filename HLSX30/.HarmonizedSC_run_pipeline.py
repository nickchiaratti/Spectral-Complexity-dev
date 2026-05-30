import os
import sys
import yaml
import subprocess
from pathlib import Path

# ==========================================
# PIPELINE CONFIGURATION
# ==========================================
script_dir = Path(__file__).resolve().parent
# If TARGET_LOCATION is None, it will fall back to 'current_run' in the YAML
TARGET_LOCATION = None
CONFIG_FILE_PATH = os.path.join(script_dir, "locations_config.yaml")
SKIP_VIEW = False  # Set to True to skip the interactive mgrs_view step

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

    scripts_to_run = [
        os.path.join(script_dir,"mgrs_view.py"),
        os.path.join(script_dir,"HLS30-earthAccess-to-hdf5.py"),
        os.path.join(script_dir,"HLST-constellation-to-hdf5.py"),
        os.path.join(script_dir,"HLST_SC_calculations.py"),
        os.path.join(script_dir,"HLST_specComplex_viewer.py")
    ]

    for script in scripts_to_run:
        print(f"\n{'='*50}")
        print(f"Executing: {script}")
        print(f"{'='*50}")
        
        if os.path.basename(script) == "mgrs_view.py":
            if SKIP_VIEW:
                print("Skipping mgrs_view.py as requested.")
                continue
            else:
                print(f"Launching Streamlit interface for {script}...")
                print("Please interact with the browser window, then click 'Close App & Continue Pipeline' to proceed.")
                subprocess.run([sys.executable, "-m", "streamlit", "run", script], check=True)
                continue
        elif os.path.basename(script) == "HLST_specComplex_viewer.py":
            print(f"Launching SpecComplex Viewer for {location_name}...")
            # Compute file path based on default naming convention in calculation script
            file_path = f"C:/satelliteImagery/HLST30/HLST_{location_name}_Harmonized_SC_EM-7_Norm-bandCount.h5"
            
            # Extract start and end year from config
            start_date_str = loc_config.get("START_DATE", "2024-01-01")
            end_date_str = loc_config.get("END_DATE", "2025-01-01")
            start_year = int(start_date_str.split("-")[0])
            end_year = int(end_date_str.split("-")[0])
            
            try:
                subprocess.run([sys.executable, script, "--file", file_path, "--start_year", str(start_year), "--end_year", str(end_year)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing {script}. Pipeline halted.")
                sys.exit(e.returncode)
            continue
            
        try:
            # We run the script. The script itself will read from locations_config.yaml -> current_run -> location
            subprocess.run([sys.executable, script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}. Pipeline halted.")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            print("\nPipeline aborted by user.")
            sys.exit(0)

    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()
