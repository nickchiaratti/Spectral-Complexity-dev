import os
import sys
import subprocess

def run_pipeline(location):
    print(f"Running Consolidated Native-Grid Pipeline for {location}...")
    
    # Step 1: Native Stackers
    subprocess.run(["python", "HLS30/HLS_native_stacker.py", "--location", location])
    subprocess.run(["python", "Tanager-1/Tanager_native_stacker.py", "--location", location])
    subprocess.run(["python", "enmap/EnMAP_native_stacker.py", "--location", location])
    subprocess.run(["python", "Dragonette/Dragonette_native_stacker.py", "--location", location])
    
    # Step 1.5: Mask Distribution
    subprocess.run(["python", "Harmonized_SC/distribute_water_mask.py", "--location", location])
    
    # Step 2: Native SC Calculations
    sensors = [
        f"C:/satelliteImagery/HLS30/HLS_{location}_Native_Stack.h5", # simplified
        f"C:/satelliteImagery/Tanager/Tanager_Native_Stack_{location}.h5",
        f"C:/satelliteImagery/enmap/EnMAP_Native_Stack_{location}.h5",
        f"C:/satelliteImagery/dragonette/Dragonette_Native_Stack_{location}.h5"
    ]
    for s in sensors:
        if os.path.exists(s):
            subprocess.run(["python", "Harmonized_SC/native_SC_calculations.py", "--file", s])
            
    # Step 3: Constellation
    subprocess.run(["python", "Harmonized_SC/consolidated_SC_constellation.py", "--location", location])
    
    print("Pipeline Complete!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", default="Tait")
    args = parser.parse_args()
    run_pipeline(args.location)
