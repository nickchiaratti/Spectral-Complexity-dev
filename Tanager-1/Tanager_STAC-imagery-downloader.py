import os
import requests
import json
from pathlib import Path
from urllib.parse import urljoin

# --- Pre-defined Regions of Interest (Bounding Boxes: [min_lon, min_lat, max_lon, max_lat]) ---
REGIONS = {
    "Southern_California": [-121.5, 32, -113, 35.5],
    "Utah": [-114.05, 37.0, -109.0, 42.5],
    "Rochester_NY": [-77.72, 43.04, -77.44, 43.28],
    "Buenos_Aires": [ -65.0, -70.0, -41.0, -30.0],
    "Global": [-180.0, -90.0, 180.0, 90.0]
}

# --- Download Jobs Configuration ---
# To build temporal series across different catalogs, point multiple jobs to the same 'output_dir'
# and use the same 'include_bboxes'. The native stacker will combine them automatically.
DOWNLOAD_JOBS = [
    #{
    #    "job_name": "Palisades_fire",
    #    "collection_url": "https://www.planet.com/data/stac/tanager-core-imagery/fire/collection.json",
    #    "output_dir": r"C:\satelliteImagery\Tanager\Palisades_SourceData",
    #    "include_bboxes": [REGIONS["Southern_California"]],
    #    "exclude_bboxes": [REGIONS["Utah"]],
    #    "target_assets": ['ortho_sr_hdf5','ortho_visual'] 
    #},
    #{
    #    "job_name": "BuenosAires_Urban",
    #    "collection_url": "https://www.planet.com/data/stac/tanager-core-imagery/urban/collection.json",
    #    "output_dir": r"C:\satelliteImagery\Tanager\BuenosAires_SourceData",
    #    "include_bboxes": [REGIONS["Buenos_Aires"]],
    #    "exclude_bboxes": [],
    #    "target_assets": ['ortho_sr_hdf5','ortho_visual'] 
    #},
    {
        "job_name": "ROCX_Rochester",
        "collection_url": "https://www.planet.com/data/stac/tanager-core-imagery/ROCX2025/collection.json",
        "output_dir": "C:/satelliteImagery/Tanager/ROCX_SourceData",
        "include_bboxes": [REGIONS["Rochester_NY"]],
        "exclude_bboxes": [],
        "target_assets": ['ortho_sr_hdf5','ortho_visual']
    }
]

def intersects(bbox1, bbox2):
    """Evaluates whether two [min_lon, min_lat, max_lon, max_lat] bounding boxes intersect."""
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def passes_spatial_filters(item_bbox, include_bboxes, exclude_bboxes):
    """
    Evaluates an item's bounding box against explicit inclusion and exclusion regions.
    """
    if not item_bbox:
        return False

    # 1. Strict Exclusions (e.g., dropping overlapping Utah footprints)
    for ex_box in exclude_bboxes:
        if intersects(item_bbox, ex_box):
            return False

    # 2. Inclusions
    if not include_bboxes:
        return True # If no inclusion regions are specified, accept all non-excluded items
        
    for inc_box in include_bboxes:
        if intersects(item_bbox, inc_box):
            return True

    return False

def download_file(url, destination_path):
    """Streams file download to disk to handle large multi-gigabyte HDF5 assets efficiently."""
    if destination_path.exists():
        print(f"    -> Skipping {destination_path.name} (Already exists)")
        return True

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(destination_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"    -> Failed to download {url}: {e}")
        return False

def execute_job(job_config):
    """Processes a single download job dictionary."""
    job_name = job_config["job_name"]
    collection_url = job_config["collection_url"]
    out_dir = Path(job_config["output_dir"])
    includes = job_config.get("include_bboxes", [])
    excludes = job_config.get("exclude_bboxes", [])
    target_assets = job_config.get("target_assets", ['ortho_sr_hdf5'])

    print(f"\n{'='*50}")
    print(f"Executing Job: {job_name}")
    print(f"Target Directory: {out_dir}")
    print(f"{'='*50}")

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(collection_url)
        response.raise_for_status()
        collection_data = response.json()
    except Exception as e:
        print(f"CRITICAL: Failed to fetch catalog at {collection_url}: {e}")
        return

    item_links = [link['href'] for link in collection_data.get('links', []) if link.get('rel') == 'item']
    print(f"Discovered {len(item_links)} items in catalog.")
    
    matched_items = 0

    for idx, item_href in enumerate(item_links):
        item_url = urljoin(collection_url, item_href)
        
        try:
            item_resp = requests.get(item_url)
            item_resp.raise_for_status()
            item_data = item_resp.json()
        except Exception as e:
            print(f"  [Error] Failed to fetch item metadata at {item_url}: {e}")
            continue

        item_id = item_data.get('id', f'item_{idx}')
        item_bbox = item_data.get('bbox')
        
        # Apply Spatial Filtering
        if not passes_spatial_filters(item_bbox, includes, excludes):
            continue

        matched_items += 1
        print(f"\n  [{item_id}] Matched spatial criteria. Processing assets...")
        
        # Route to common folder structure to support the native stacker
        item_folder = out_dir / item_id
        item_folder.mkdir(exist_ok=True)

        # Download STAC JSON Metadata (Required for extracting exact Acquisition Time)
        json_file_name = f"{item_id}.json"
        json_dest = item_folder / json_file_name
        download_file(item_url, json_dest)

        # Download Specific Assets
        assets = item_data.get('assets', {})
        for asset_key in target_assets:
            if asset_key in assets:
                asset_url = assets[asset_key].get('href')
                if not asset_url.startswith('http'):
                    asset_url = urljoin(item_url, asset_url)
                
                file_name = os.path.basename(asset_url)
                dest_path = item_folder / file_name
                print(f"    -> Downloading {asset_key}...")
                download_file(asset_url, dest_path)
            else:
                print(f"    -> Asset '{asset_key}' not present in this item.")

    print(f"\nJob '{job_name}' completed. Successfully downloaded {matched_items} scenes.")

def main():
    print("Initializing Tanager STAC Job Queue...\n")
    for job in DOWNLOAD_JOBS:
        execute_job(job)
    print("\nAll download jobs finished.")

if __name__ == "__main__":
    main()