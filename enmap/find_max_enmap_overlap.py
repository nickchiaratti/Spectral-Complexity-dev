import requests
import json
import numpy as np
import math

# Configuration
STAC_API_URL = 'https://geoservice.dlr.de/eoc/ogc/stac/v1'
COLLECTION_ID = 'ENMAP_HSI_L2A'
CENTER_LAT = 36.7783
CENTER_LON = -119.4179
SEARCH_RADIUS_DEG = 2.0 # 4x4 degree search window (Central California)

# Rochesterv2 BBox for scale comparison
ROCH_LON_MIN, ROCH_LON_MAX = -77.770166, -77.376776
ROCH_LAT_MIN, ROCH_LAT_MAX = 42.961778, 43.342135

def compute_area_sqkm(lon_min, lon_max, lat_min, lat_max):
    """Approximate area in sq km for a given bounding box."""
    mean_lat = (lat_min + lat_max) / 2.0
    lat_dist = (lat_max - lat_min) * 111.32
    lon_dist = (lon_max - lon_min) * 111.32 * math.cos(math.radians(mean_lat))
    return lat_dist * lon_dist

def main():
    print(f"Searching EnMAP STAC for max overlaps around ({CENTER_LAT}, {CENTER_LON})...")
    
    # 1. Fetch all items in the large search window
    bbox_search = [CENTER_LON - SEARCH_RADIUS_DEG, CENTER_LAT - SEARCH_RADIUS_DEG, 
                   CENTER_LON + SEARCH_RADIUS_DEG, CENTER_LAT + SEARCH_RADIUS_DEG]
    params = {'bbox': ','.join(map(str, bbox_search)), 'limit': 1000}
    
    all_features = []
    url = f'{STAC_API_URL}/collections/{COLLECTION_ID}/items'
    
    while url:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        all_features.extend(data.get('features', []))
        
        # Follow pagination if there are more than 'limit' results
        links = data.get('links', [])
        next_link = next((link['href'] for link in links if link['rel'] == 'next'), None)
        url = next_link
        params = None # Params are embedded in the next_link
        
    print(f'Found {len(all_features)} total frames in search region.')

    # 2. Rasterize the bounding boxes onto a grid to find the peak overlap
    res_deg = 0.005 # ~500m grid resolution
    grid_lon = np.arange(bbox_search[0], bbox_search[2], res_deg)
    grid_lat = np.arange(bbox_search[1], bbox_search[3], res_deg)
    counts = np.zeros((len(grid_lat), len(grid_lon)), dtype=int)

    for f in all_features:
        f_bbox = f.get('bbox')
        if not f_bbox: continue
        f_lon_min, f_lat_min, f_lon_max, f_lat_max = f_bbox
        
        # Mask out the grid cells that fall within the feature's BBox
        lon_mask = (grid_lon >= f_lon_min) & (grid_lon <= f_lon_max)
        lat_mask = (grid_lat >= f_lat_min) & (grid_lat <= f_lat_max)
        counts[np.ix_(lat_mask, lon_mask)] += 1

    max_count = np.max(counts)
    y_idx, x_idx = np.where(counts == max_count)
    best_lat = grid_lat[y_idx[len(y_idx)//2]]
    best_lon = grid_lon[x_idx[len(x_idx)//2]]
    
    print(f'\n--- RESULTS ---')
    print(f'Peak Overlapping Frames: {max_count}')
    print(f'Peak Center: Lat {best_lat:.6f}, Lon {best_lon:.6f}')

    # 3. Size the bounding box to be exactly 1.25x the area of Rochesterv2
    roch_area = compute_area_sqkm(ROCH_LON_MIN, ROCH_LON_MAX, ROCH_LAT_MIN, ROCH_LAT_MAX)
    max_area = roch_area * 1.25 # 25% larger than Rochesterv2
    
    # Maintain Rochesterv2 aspect ratio for the new box
    roch_lat_dist = (ROCH_LAT_MAX - ROCH_LAT_MIN) * 111.32
    roch_lon_dist = (ROCH_LON_MAX - ROCH_LON_MIN) * 111.32 * math.cos(math.radians((ROCH_LAT_MIN + ROCH_LAT_MAX) / 2))
    aspect = roch_lon_dist / roch_lat_dist
    
    target_h_km = math.sqrt(max_area / aspect)
    target_w_km = target_h_km * aspect
    
    target_lat_diff = target_h_km / 111.32
    target_lon_diff = target_w_km / (111.32 * math.cos(math.radians(best_lat)))

    best_bbox = [
        best_lon - target_lon_diff/2,
        best_lat - target_lat_diff/2,
        best_lon + target_lon_diff/2,
        best_lat + target_lat_diff/2
    ]

    print(f'\nProposed BBox config for exactly +25% pixels ({max_area:.2f} sq km):')
    print(f'ROI_LON_MIN: {best_bbox[0]:.6f}')
    print(f'ROI_LON_MAX: {best_bbox[2]:.6f}')
    print(f'ROI_LAT_MIN: {best_bbox[1]:.6f}')
    print(f'ROI_LAT_MAX: {best_bbox[3]:.6f}')

if __name__ == "__main__":
    main()
