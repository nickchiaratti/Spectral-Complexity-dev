import os
import h5py
import json
import numpy as np

# JSON encoder to handle numpy data types gracefully
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            # convert NaNs and Infs to None/null to be JSON compliant
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

def extract_pixels_to_json(h5_path, location, pixels, json_path="z-score-samples.json"):
    """
    Extracts time-series data for a list of pixels and saves to a structured JSON file.
    This function is modular and can be imported into visualization scripts.
    
    Args:
        h5_path (str): Path to the source HDF5 file.
        location (str): The name of the location (e.g., 'Tait').
        pixels (list of tuples): List of pixels in format (x, y, category).
            Example: [(50, 100, 'structural change'), (20, 30, 'stable periodic patterns')]
        json_path (str): Output JSON file path.
    """
    if not os.path.exists(h5_path):
        print(f"Error: HDF5 file not found at {h5_path}")
        return

    print(f"Extracting {len(pixels)} pixels from {h5_path}...")
    
    # Load existing JSON if it exists
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data_dict = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {json_path} is corrupted or empty. Starting fresh.")
            data_dict = {}
    else:
        data_dict = {}
        
    with h5py.File(h5_path, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        metric_ds = data_grp['sliding_volume_z_score']
        mask_ds = data_grp['common_mask']
        
        acq_times = metric_ds.attrs['acquisition_time'][:]
        
        for x, y, category in pixels:
            # Note: HDF5 standard is [time, y, x] where y is height/row and x is width/col
            z_scores = metric_ds[:, y, x]
            masks = mask_ds[:, y, x]
            
            # Ensure category exists in dictionary
            if category not in data_dict:
                data_dict[category] = {}
                
            # Create a unique key for the pixel to easily overwrite duplicates
            pixel_key = f"{location}_{x}_{y}"
            
            # Save or overwrite pixel data
            data_dict[category][pixel_key] = {
                "Location": location,
                "X": int(x),
                "Y": int(y),
                "timestamp": acq_times.tolist(),
                # Convert np.nan to None (JSON null) for valid JSON format
                "sliding_volume_z_score": [None if np.isnan(val) else val for val in z_scores.tolist()],
                "common_mask": masks.tolist()
            }
            
    # Save back to JSON
    with open(json_path, 'w') as f:
        json.dump(data_dict, f, cls=NumpyEncoder, indent=2)
        
    print(f"Successfully appended/updated pixel data in {json_path}")

def append_single_pixel(h5_path, location, x, y, category, json_path="z-score-samples.json"):
    """
    Convenience function to extract and append a single pixel.
    """
    extract_pixels_to_json(h5_path, location, [(x, y, category)], json_path)

if __name__ == "__main__":
    # ==========================================
    # Example Usage / Manual Execution
    # ==========================================
    # You can run this script directly to extract your manually identified pixels.
    
    # Configuration
    H5_FILE = "C:/satelliteImagery/HLST30/HLST_Tait_Harmonized_SC_EM-7_Norm-bandCount.h5"
    LOCATION_NAME = "Tait"
    OUTPUT_JSON = "z-score-samples.json"
    
    # Populate this list with your identified pixels
    # Format: (x_coordinate, y_coordinate, 'category')
    pixels_of_interest = [
        # Uncomment and modify these examples with your actual coordinates:
        (100, 100, 'structural change'),
        # (110, 120, 'transient event'),
        # (150, 200, 'stable periodic patterns'),
        # (50, 50, 'noisy data'),
        # (10, 10, 'indeterminate data')
    ]
    
    if pixels_of_interest:
        extract_pixels_to_json(H5_FILE, LOCATION_NAME, pixels_of_interest, OUTPUT_JSON)
    else:
        print("Please add coordinates to 'pixels_of_interest' to run as a standalone script.")
