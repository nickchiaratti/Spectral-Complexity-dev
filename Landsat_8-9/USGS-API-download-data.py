# Refactored script to download Level 2 SR data AND Level 1 Band 8 Panchromatic data
import json
import requests
import sys
import time
import argparse
import datetime
import threading
import re
import os

Location = "Rochester"


# --- CONFIGURATION ---
path = "C:/satelliteImagery/LANDSAT/SourceData/" + Location 
maxthreads = 5 
sema = threading.Semaphore(value=maxthreads)
threads = []
username = 'nchiaratti'
token = 'replace with your USGS-M2M token'
serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"

def sendRequest(url, data, apiKey=None):  
    pos = url.rfind('/') + 1
    endpoint = url[pos:]
    json_data = json.dumps(data)
    
    if apiKey == None:
        response = requests.post(url, json_data)
    else:
        headers = {'X-Auth-Token': apiKey}              
        response = requests.post(url, json_data, headers=headers)    
    
    try:
        httpStatusCode = response.status_code 
        output = json.loads(response.text)	
        if output.get('errorCode') is not None:
            print(f"Error in {endpoint}: {output['errorCode']} - {output['errorMessage']}")
            sys.exit()
        return output.get('data')
    except Exception as e: 
        print(f"Failed to parse request {endpoint}. Error: {e}")
        sys.exit()

def downloadFile(url, fileNameId):
    """Downloads a file, skipping if it already exists. Uses displayId for filename."""
    sema.acquire()
    global path
    try:
        # Get filename from headers
        head_response = requests.head(url, allow_redirects=True)
        disposition = head_response.headers.get('content-disposition', '')
        
        filename = ""
        if "filename=" in disposition:
            filename = re.findall("filename=(.+)", disposition)[0].strip("\"")
        
        # FORCE: If filename is generic "gen-bundle", missing, or is an entityId (doesn't start with LC08/LC09), use fileNameId
        # USGS entityIds usually look like LC8... whereas displayIds look like LC08...
        if not filename or "gen-bundle" in filename.lower() or not filename.startswith("LC0"):
            # Preserve extension if found in disposition, otherwise default to .tar
            ext = ".tar"
            if filename and "." in filename:
                ext = "." + filename.split(".")[-1]
            filename = f"{fileNameId}{ext}" 

        full_path = os.path.join(path, filename)

        # Skip if already exists
        if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
            print(f"Skipping {filename} - already exists.")
            return

        print(f"Starting download: {filename} ...")
        response = requests.get(url, stream=True)
        with open(full_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                if chunk:
                    f.write(chunk)
        print(f"Successfully downloaded {filename}")
        
    except Exception as e:
        print(f"Failed to download from {url}. Error: {e}")
    finally:
        sema.release()
    
def runDownload(url, fileNameId):
    """Utility to trigger a threaded download."""
    thread = threading.Thread(target=downloadFile, args=(url, fileNameId))
    threads.append(thread)
    thread.start()

def process_downloads(downloads_to_request, download_to_name_map, entity_to_name_map, apiKey):
    """Handles the actual download request and retrieval process."""
    if not downloads_to_request:
        return

    label = "download_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    requestResults = sendRequest(serviceUrl + "download-request", {'downloads' : downloads_to_request, 'label' : label}, apiKey)
    
    # 1. Handle products available immediately
    for d in requestResults.get('availableDownloads', []):
        d_id_str = str(d.get('downloadId', ''))
        e_id = d.get('entityId')
        # Check mapping by download ID first, then by entity ID
        name_id = download_to_name_map.get(d_id_str) or entity_to_name_map.get(e_id) or e_id or "unknown_scene"
        runDownload(d['url'], name_id)
    
    # 2. Handle products being prepared (on-demand)
    preparing = requestResults.get('preparingDownloads', [])
    if preparing:
        print(f"Waiting for {len(preparing)} products to be prepared...")
        total_to_wait = len(preparing)
        found_ids = set()
        
        for attempt in range(12): # Polling for roughly 6 minutes
            print(f"Polling prepared downloads (Attempt {attempt+1}/12)...")
            time.sleep(30) 
            retrieveResults = sendRequest(serviceUrl + "download-retrieve", {'label' : label}, apiKey)
            
            for d in retrieveResults.get('available', []):
                d_id = str(d['downloadId'])
                if d_id not in found_ids:
                    found_ids.add(d_id)
                    # Look up the displayId we stored for this download/product ID
                    name_id = download_to_name_map.get(d_id) or d.get('displayId') or d.get('entityId') or "unknown_scene"
                    runDownload(d['url'], name_id)
            
            if len(found_ids) >= total_to_wait:
                break

if __name__ == '__main__': 
    if not os.path.exists(path):
        os.makedirs(path)

    # Login
    apiKey = sendRequest(serviceUrl + "login-token", {'username' : username, 'token' : token})
    print(f"Logged in successfully.\n")

    # Search parameters
    if Location == "Rochester":
        spatialFilter =  {'filterType' : "mbr",
            'lowerLeft' : {'latitude' : 43.072486, 'longitude' : -77.688990},
            'upperRight' : { 'latitude' : 43.159168, 'longitude' : -77.660365}}  #RIT Campus
    elif Location == "Davis":
        spatialFilter =  {'filterType' : "mbr",
            'lowerLeft' : {'latitude' : 38.522226, 'longitude' : -121.792782},
            'upperRight' : { 'latitude' : 38.547101, 'longitude' : -121.746691}}  #UC Davis Campus
    else:
        print("Invalid Location")
        sys.exit()

    temporalFilter = {'start' : '2023-01-01', 'end' : '2025-12-31'}
    cloudCoverFilter = {'min' : 0, 'max' : 25}

    target_datasets = ["landsat_ot_c2_l2"]
    
    all_downloads_to_request = []
    all_download_to_name_map = {} # Maps product/download ID (string) to displayId
    all_entity_to_name_map = {}   # Maps entityId to displayId
    found_display_ids = set()

    for datasetName in target_datasets:
        print(f"Searching Dataset: {datasetName}...")
        
        search_payload = {
            'datasetName' : datasetName,
            'sceneFilter' : {
                'spatialFilter' : spatialFilter,
                'acquisitionFilter' : temporalFilter,
                'cloudCoverFilter' : cloudCoverFilter
            }
        }
        
        scenes_data = sendRequest(serviceUrl + "scene-search", search_payload, apiKey)
    
        if scenes_data and scenes_data['recordsReturned'] > 0:
            # Create mapping from entityId to displayId for search results
            entity_to_display = {res['entityId']: res['displayId'] for res in scenes_data['results']}
            all_entity_to_name_map.update(entity_to_display)
            
            sceneIds = list(entity_to_display.keys())
            found_display_ids.update(entity_to_display.values())
            
            # Get download options for the found scenes
            downloadOptions = sendRequest(serviceUrl + "download-options", {'datasetName' : datasetName, 'entityIds' : sceneIds}, apiKey)
        
            for product in downloadOptions:
                p_name = product['productName']
                if product['available']:
                    is_match = False
                    if datasetName == "landsat_ot_c2_l2":
                        # Target the standard Level 2 Product Bundle
                        if "BUNDLE" in p_name.upper() or "LEVEL 2 PRODUCT" in p_name.upper():
                             is_match = True
                    
                    if is_match:
                        # Add to request list
                        all_downloads_to_request.append({'entityId' : product['entityId'], 'productId' : product['id']})
                        
                        # Store the mapping between the product/download ID and the scene's displayId
                        d_name = entity_to_display.get(product['entityId'], product['entityId'])
                        all_download_to_name_map[str(product['id'])] = d_name

    if found_display_ids:
        print("\n--- Search Results ---")
        print(f"Found {len(found_display_ids)} unique scenes matching criteria:")
        for did in sorted(list(found_display_ids)):
            print(f"  - {did}")
        print(f"Total products identified for download: {len(all_downloads_to_request)}")
        
        confirm = input("\nWould you like to proceed with the download? (y/n): ").strip().lower()
        if confirm == 'y':
            process_downloads(all_downloads_to_request, all_download_to_name_map, all_entity_to_name_map, apiKey)
            
            print("\nWaiting for all downloads to complete...")
            for t in threads:
                t.join()
            print("\nAll downloads finished.")
        else:
            print("Download cancelled by user.")
    else:
        print("No scenes found matching search filters.")

    sendRequest(serviceUrl + "logout", None, apiKey)
    print("Logged out.")