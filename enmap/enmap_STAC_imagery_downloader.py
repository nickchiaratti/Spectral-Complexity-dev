"""
Downloads EnMAP HSI L2A data for specified ROIs and time ranges from the DLR EOC Geoservice STAC.
Uses pystac-client for querying and requests with HTTP Basic Auth for downloading.
"""
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pathlib import Path
from datetime import datetime
import getpass
import json
import browser_cookie3
import yaml

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Target collection
STAC_API_URL = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
COLLECTION_ID = "ENMAP_HSI_L2A"
TARGET_LOCATION = 'Malibu'

def auto_cas_login(session, user, password):
    """Automatically logs into DLR's CAS SSO to establish download session cookies."""
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin
    print("Authenticating with DLR CAS using credentials from .env...")
    
    # We use a known protected endpoint to trigger the CAS redirect
    trigger_url = 'https://download.geoservice.dlr.de/ENMAP/files/L2A/2026/08/08/DT0000208491/01/ENMAP01-____L2A-DT0000208491_20260808T004733Z_001_V010506_20260809T184648Z/ENMAP01-____L2A-DT0000208491_20260808T004733Z_001_V010506_20260809T184648Z-METADATA.XML'
    
    r_get = session.get(trigger_url)
    soup = BeautifulSoup(r_get.text, 'html.parser')
    form = soup.find('form', id='fm1')
    
    if not form:
        print("Warning: CAS login form not found. Assuming session is already valid.")
        return session
        
    action = form.get('action', 'login')
    post_url = urljoin(r_get.url, action)
    
    payload = {}
    for inp in form.find_all('input'):
        n = inp.get('name')
        v = inp.get('value', '')
        if n:
            payload[n] = v
            
    payload['username'] = user
    payload['password'] = password
    payload['_eventId'] = 'submit'
    
    xsrf_token = session.cookies.get('XSRF-TOKEN')
    headers = {}
    if xsrf_token:
        headers['X-XSRF-TOKEN'] = xsrf_token
        payload['_csrf'] = xsrf_token
        
    r_post = session.post(post_url, data=payload, headers=headers, allow_redirects=True)
    if 'ticket=' in r_post.url or 'download.geoservice.dlr.de' in r_post.url:
        print("Successfully authenticated with DLR download server.")
    else:
        print(f"Warning: Unexpected post-login URL: {r_post.url}")
        
    return session

def create_retry_session():
    """Creates a robust requests Session equipped with exponential backoff and automated CAS authentication."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    env_path = Path(__file__).parent.parent / '.env'
    user, pwd = None, None
    env_cookie = None
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    k, v = line.strip().split('=', 1)
                    if k.strip() == 'ENMAP_USERNAME': user = v.strip()
                    if k.strip() == 'ENMAP_PASSWORD': pwd = v.strip()
                    if k.strip() == 'ENMAP_COOKIE': env_cookie = v.strip().strip('"').strip("'")
    
    if user and pwd:
        session = auto_cas_login(session, user, pwd)
    elif env_cookie:
        print("Using session cookie provided in .env file (ENMAP_COOKIE).")
        session.headers.update({'Cookie': env_cookie})
    else:
        print("Extracting session cookies for dlr.de from your Chrome/local browser...")
        try:
            try:
                cj = browser_cookie3.chrome(domain_name='dlr.de')
            except Exception:
                cj = browser_cookie3.load(domain_name='dlr.de')
            session.cookies.update(cj)
        except Exception as e:
            print(f"Warning: Could not extract browser cookies ({e}).")

    retry_strategy = Retry(
        total=10,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=2,
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def download_file(url, destination_path, session):
    """Streams file download to disk to handle large multi-gigabyte assets efficiently."""
    if destination_path.exists():
        print(f"    -> Skipping {destination_path.name} (Already exists)")
        return True

    print(f"    -> Downloading to {destination_path.name}...")
    try:
        with session.get(url, stream=True) as r:
            if r.status_code in [401, 403]:
                print(f"    -> Authentication failed or access denied (HTTP {r.status_code}) for {url}")
                raise PermissionError("EnMAP authentication token is expired or invalid. Please update ENMAP_COOKIE in your .env file.")
            r.raise_for_status()

            # Verify response is not an HTML login/error page instead of valid binary/data asset
            content_type = r.headers.get('Content-Type', '').lower()
            if 'text/html' in content_type and not destination_path.name.endswith('.html'):
                print(f"    -> Download returned an HTML login page instead of asset file. Session authentication required for {url}")
                raise PermissionError("EnMAP authentication token is expired or invalid. Please update ENMAP_COOKIE in your .env file.")

            with open(destination_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except PermissionError:
        # Clean up partial file if we somehow created one
        if destination_path.exists():
            destination_path.unlink()
        raise  # Bubble up to abort the pipeline
    except Exception as e:
        print(f"    -> Error downloading {url}: {e}")
        # Clean up partial file
        if destination_path.exists():
            destination_path.unlink()
        return False

def execute_job(job_config, session):
    job_name = job_config["job_name"]
    bbox = job_config["bbox"]
    dt_range = job_config["datetime"]
    target_assets = job_config.get("target_assets", ['image','metadata','quality_classes'])
    
    out_dir = job_config["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Executing Job: {job_name}")
    print(f"BBox: {bbox}")
    print(f"Time: {dt_range}")
    print(f"Target Directory: {out_dir}")
    print(f"{'='*50}")
    
    # 1. Query STAC
    dt_safe = dt_range.replace('/', '_').replace(':', '')
    cache_file = out_dir / f"STAC_Search_Results_{job_name}_{dt_safe}.json"
    
    if cache_file.exists():
        print(f"Loading STAC results from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            items = json.load(f)
    else:
        print("Querying DLR EOC STAC...")
        
        # Target the collection directly instead of using the global /search endpoint
        collection_items_url = f"{STAC_API_URL}/collections/{COLLECTION_ID}/items"
        
        params = {
            "bbox": ",".join(map(str, bbox)),
            "datetime": dt_range,
            "limit": 500
        }
        
        try:
            response = session.get(collection_items_url, params=params)
            response.raise_for_status()
            feature_collection = response.json()
            items = feature_collection.get("features", [])
            
            # Handle simple pagination if there are more results
            next_link = next((link['href'] for link in feature_collection.get('links', []) if link['rel'] == 'next'), None)
            while next_link:
                resp = session.get(next_link)
                resp.raise_for_status()
                fc = resp.json()
                items.extend(fc.get("features", []))
                next_link = next((link['href'] for link in fc.get('links', []) if link['rel'] == 'next'), None)
                
            # Save STAC results to cache
            with open(cache_file, 'w') as f:
                json.dump(items, f, indent=4)
                print(f"Saved STAC results cache to: {cache_file}")
                
        except Exception as e:
            print(f"Failed to query STAC API: {e}")
            return

    filtered_items = items
    print(f"Found {len(filtered_items)} items matching query criteria.")
    
    matched_items = 0
    for idx, item in enumerate(filtered_items):
        matched_items += 1
        item_id = item.get('id', f'item_{idx}')
        print(f"\n  [{idx+1}/{len(filtered_items)}] Processing {item_id}...")
        
        item_folder = out_dir / item_id
        item_folder.mkdir(exist_ok=True)
        
        # Save STAC metadata locally
        json_file_name = f"{item_id}.json"
        json_dest = item_folder / json_file_name
        if not json_dest.exists():
            with open(json_dest, 'w') as f:
                json.dump(item, f, indent=2)
                
        # Download assets
        assets = item.get('assets', {})
        if not assets:
            print("    -> No assets found for this item.")
            continue
            
        for asset_key, asset in assets.items():
            if target_assets and asset_key not in target_assets:
                continue
                
            href = asset.get('href')
            if not href:
                continue
            
            # WORKAROUND: DLR STAC Catalog has a bug where older items have invalid hrefs missing the item_id folder.
            # Example Bad:  .../DT0000163932/05/ENMAP01-...-METADATA.XML
            # Example Good: .../DT0000163932/05/ENMAP01-.../ENMAP01-...-METADATA.XML
            # We automatically fix the URL by injecting the item_id folder if it's missing.
            file_name = os.path.basename(href.split("?")[0])
            
            # Skip VNIR_COG.TIF and SWIR_COG.TIF assets (and thumbnails/quicklooks)
            upper_name = file_name.upper()
            if "VNIR_COG.TIF" in upper_name or "SWIR_COG.TIF" in upper_name:
                continue

            if f"/{item_id}/" not in href:
                parent_dir_path = href.rsplit('/', 1)[0]
                href = f"{parent_dir_path}/{item_id}/{file_name}"

            dest_path = item_folder / file_name
            
            download_file(href, dest_path, session)
            
    print(f"\nJob '{job_name}' completed. Successfully processed {matched_items} scenes.")

def main(target_location=None):
    print("Initializing EnMAP STAC Job Queue...\n")
    
    script_dir = Path(__file__).resolve().parent
    config_path = os.path.join(script_dir.parent, "locations_config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: Could not find configuration file at {config_path}")
        return
        
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    if target_location is None:
        target_location = config_data.get("current_run", {}).get("location")
        
    download_jobs = []
    locations_to_process = [target_location] if target_location else config_data.get("locations", {}).keys()
    
    for loc_name in locations_to_process:
        if loc_name not in config_data.get("locations", {}):
            print(f"Warning: Location '{loc_name}' not found in locations_config.yaml")
            continue
            
        loc_config = config_data["locations"][loc_name]
        
        min_lon = loc_config["ROI_LON_MIN"]
        max_lon = loc_config["ROI_LON_MAX"]
        min_lat = loc_config["ROI_LAT_MIN"]
        max_lat = loc_config["ROI_LAT_MAX"]
        
        start_date = loc_config.get("START_DATE", "2023-01-01")
        end_date = loc_config.get("END_DATE", "2025-12-31")
        
        source_cache = loc_config.get("SOURCE_CACHE") or loc_name
        
        # Enforce valid bbox format [minx, miny, maxx, maxy]
        bbox = [min(min_lon, max_lon), min(min_lat, max_lat), max(min_lon, max_lon), max(min_lat, max_lat)]
        
        out_dir = Path(f"C:/satelliteImagery/Enmap/{source_cache}_SourceData")
        
        download_jobs.append({
            "job_name": f"{loc_name}_EnMAP",
            "bbox": bbox,
            "datetime": f"{start_date}/{end_date}",
            "target_assets": [],
            "out_dir": out_dir
        })
    
    session = create_retry_session()
    
    for job in download_jobs:
        execute_job(job, session)
        
    print("\nAll download jobs finished.")

if __name__ == "__main__":
    main(target_location=TARGET_LOCATION)
