import os
import re
import math
import rasterio
import numpy as np
from datetime import datetime

# --- Configuration ---
SOURCE_DIR = r"C:\satelliteImagery\OSK-Ghost\SourceData"

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the great-circle distance between two points on the Earth's surface.
    Returns distance in meters.
    """
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_igm_files(source_dir):
    igm_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('_igm'):
                igm_path = os.path.join(root, file)
                match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)', file)
                if match:
                    dt_str = match.group(1)
                    dt = datetime.strptime(dt_str, "%Y-%m-%dT%H-%M-%SZ")
                    igm_files.append({'dt': dt, 'dt_str': dt_str, 'path': igm_path})
    igm_files.sort(key=lambda x: x['dt'])
    return igm_files

def main():
    print(f"Scanning for OSK-Ghost _igm files in: {SOURCE_DIR}")
    igm_files = get_igm_files(SOURCE_DIR)
    
    if not igm_files:
        print("No valid _igm files found. Exiting.")
        return

    print(f"\nFound {len(igm_files)} IGM files. Calculating raw telemetry centroids...")
    print("-" * 65)
    
    centroids = []
    
    for item in igm_files:
        with rasterio.open(item['path']) as src:
            # Strict read. If the file is corrupted, allow the crash.
            lat_data = src.read(1)
            lon_data = src.read(2)
            
            # Extract the geometric center index of the raw array
            center_row = lat_data.shape[0] // 2
            center_col = lat_data.shape[1] // 2
            
            c_lat = lat_data[center_row, center_col]
            c_lon = lon_data[center_row, center_col]
            
            centroids.append({
                'dt_str': item['dt_str'],
                'lat': c_lat,
                'lon': c_lon
            })
            
    for i in range(len(centroids) - 1):
        c1 = centroids[i]
        c2 = centroids[i+1]
        
        dist_m = haversine_distance(c1['lat'], c1['lon'], c2['lat'], c2['lon'])
        
        print(f"Frame {i} vs Frame {i+1} ({c1['dt_str']} -> {c2['dt_str']}):")
        print(f"  Raw IGM Center 1: {c1['lat']:.6f}, {c1['lon']:.6f}")
        print(f"  Raw IGM Center 2: {c2['lat']:.6f}, {c2['lon']:.6f}")
        print(f"  Absolute Drift:   {dist_m:.2f} meters")
        print("-" * 65)

if __name__ == "__main__":
    main()