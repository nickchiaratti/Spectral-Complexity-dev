import os

# --- CRITICAL FIX FOR RESILIO/NETWORK DRIVES ---
# Disables HDF5 file locking to prevent OSError: [Errno 0]
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE' 

import glob
import shutil
import tempfile
import json
import re
import numpy as np
import h5py
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from osgeo import gdal, osr

# ================= CONFIGURATION =================
Location = "Tait"
SOURCE_DIR = r"C:\satelliteImagery\OSK-Ghost\imagery"
OUTPUT_FILE = os.path.join(SOURCE_DIR, f"OSK_Ghost_TimeSeries_{Location}.h5")

if Location == "Rochester":
    ROI_LON_MIN, ROI_LON_MAX = -77.72, -77.50
    ROI_LAT_MIN, ROI_LAT_MAX = 43.08, 43.28
elif Location == "Tait":
    ROI_LON_MIN, ROI_LON_MAX = -77.516127, -77.461968
    ROI_LAT_MIN, ROI_LAT_MAX = 43.127698, 43.159168
elif Location == "Tait-Tight":
    ROI_LON_MIN, ROI_LON_MAX = -77.510594, -77.497333
    ROI_LAT_MIN, ROI_LAT_MAX = 43.137844, 43.148929
elif Location == "RIT":
    ROI_LON_MIN, ROI_LON_MAX = -77.688990, -77.660365
    ROI_LAT_MIN, ROI_LAT_MAX = 43.072486, 43.093298
elif Location == "Seabreeze":
    ROI_LON_MIN, ROI_LON_MAX = -77.556403, -77.522049
    ROI_LAT_MIN, ROI_LAT_MAX = 43.223786, 43.242186
elif Location == "LakeOntario":
    ROI_LON_MIN, ROI_LON_MAX = -77.560544, -77.520075
    ROI_LAT_MIN, ROI_LAT_MAX = 43.223786, 43.241007

EPSG_CODE = 32618        # UTM Zone 18N
DEFAULT_IGM_LON_BAND = 2
DEFAULT_IGM_LAT_BAND = 1
# =================================================

gdal.UseExceptions()

def parse_xml_value(root, key):
    """Helper to safely extract text from XML MDI keys."""
    for mdi in root.findall(".//Metadata[@domain='ENVI']/MDI"):
        if mdi.get('key') == key:
            return mdi.text.strip()
    return None

def parse_aux_xml_for_wavelengths(xml_path):
    wavelengths = []
    units = "Unknown"
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        wl_str = parse_xml_value(root, 'wavelength')
        if wl_str:
            clean_str = wl_str.replace('{', '').replace('}', '')
            wavelengths = [float(x) for x in clean_str.split(',') if x.strip()]
        
        u_str = parse_xml_value(root, 'wavelength_units')
        if u_str: units = u_str
    except Exception as e:
        print(f"Warning: Could not parse XML wavelengths: {e}")
    return wavelengths, units

def parse_resolution_from_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        val = parse_xml_value(tree.getroot(), 'max_gsd')
        if val: return float(val)
    except: pass
    return None

def detect_igm_bands(xml_path):
    lon_idx = DEFAULT_IGM_LON_BAND
    lat_idx = DEFAULT_IGM_LAT_BAND
    try:
        tree = ET.parse(xml_path)
        val = parse_xml_value(tree.getroot(), 'band_names')
        if val:
            names = [x.strip().lower() for x in val.replace('{','').replace('}','').split(',')]
            found_lat = False
            found_lon = False
            for i, name in enumerate(names):
                if 'lat' in name:
                    lat_idx = i + 1
                    found_lat = True
                elif 'long' in name or 'lon' in name:
                    lon_idx = i + 1
                    found_lon = True
            if found_lat and found_lon:
                print(f"  Detected IGM Band Order: Lon={lon_idx}, Lat={lat_idx}")
                return lon_idx, lat_idx
    except: pass
    
    print(f"  Using Default IGM Order: Lon={lon_idx}, Lat={lat_idx}")
    return lon_idx, lat_idx

def create_temp_igm_with_hdr(igm_bin_path, xml_path, temp_dir, idx):
    """
    Copies the IGM binary to temp and writes a real .hdr file.
    Includes 'data ignore value = 0' to silence GDAL warnings about nodata.
    Returns path to the temp IGM dataset.
    """
    try:
        # 1. Parse Metadata
        tree = ET.parse(xml_path)
        root = tree.getroot()
        lines = parse_xml_value(root, 'lines')
        samples = parse_xml_value(root, 'samples')
        bands = parse_xml_value(root, 'bands')
        data_type = parse_xml_value(root, 'data_type')
        interleave = parse_xml_value(root, 'interleave')
        byte_order = parse_xml_value(root, 'byte_order')
        
        # Defaults if missing (standard ENVI)
        if not interleave: interleave = 'bsq'
        if not byte_order: byte_order = '0' # Little Endian
        
        # 2. Copy Binary to Temp
        temp_bin_name = f"temp_igm_{idx}.img"
        temp_bin_path = os.path.join(temp_dir, temp_bin_name)
        shutil.copy2(igm_bin_path, temp_bin_path)
        
        # 3. Write .hdr file
        hdr_content = f"""ENVI
description = {{Temporary IGM Header}}
samples = {samples}
lines   = {lines}
bands   = {bands}
header offset = 0
file type = ENVI Standard
data type = {data_type}
interleave = {interleave}
byte order = {byte_order}
band names = {{Lat, Lon}}
data ignore value = 0
"""
        temp_hdr_path = os.path.join(temp_dir, f"temp_igm_{idx}.hdr")
        with open(temp_hdr_path, 'w') as f:
            f.write(hdr_content)
            
        return temp_bin_path
        
    except Exception as e:
        raise RuntimeError(f"Failed to create temp IGM/HDR: {e}")

def get_roi_bounds_projected(lon_min, lon_max, lat_min, lat_max, resolution, epsg):
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4326)
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    tgt_srs = osr.SpatialReference()
    tgt_srs.ImportFromEPSG(epsg)

    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    corners = [
        transform.TransformPoint(lon_min, lat_max),
        transform.TransformPoint(lon_max, lat_max),
        transform.TransformPoint(lon_max, lat_min),
        transform.TransformPoint(lon_min, lat_min)
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]

    min_x = np.floor(min(xs) / resolution) * resolution
    max_x = np.ceil(max(xs) / resolution) * resolution
    min_y = np.floor(min(ys) / resolution) * resolution
    max_y = np.ceil(max(ys) / resolution) * resolution

    width = int((max_x - min_x) / resolution)
    height = int((max_y - min_y) / resolution)
    return (min_x, min_y, max_x, max_y), (width, height)

def generate_struct_metadata(grid_name, width, height, ul_mtrs, lr_mtrs, n_times, n_bands):
    odl = []
    odl.append("GROUP=SwathStructure")
    odl.append("END_GROUP=SwathStructure")
    odl.append("GROUP=GridStructure")
    odl.append(f"\tGROUP=GRID_1")
    odl.append(f"\t\tGridName=\"{grid_name}\"")
    odl.append(f"\t\tXDim={width}")
    odl.append(f"\t\tYDim={height}")
    odl.append(f"\t\tUpperLeftPointMtrs=({ul_mtrs[0]:.6f},{ul_mtrs[1]:.6f})")
    odl.append(f"\t\tLowerRightMtrs=({lr_mtrs[0]:.6f},{lr_mtrs[1]:.6f})")
    odl.append(f"\t\tProjection=GCTP_UTM")
    odl.append(f"\t\tZoneCode={EPSG_CODE % 100}")
    odl.append(f"\t\tSphereCode=12")
    odl.append(f"\t\tProjParams=(0,0,0,0,0,0,0,0,0,0,0,0,0)")
    odl.append("\t\tGROUP=Dimension")
    dims = [("Time", n_times), ("Band", n_bands), ("YDim", height), ("XDim", width)]
    for i, (name, size) in enumerate(dims):
        odl.append(f"\t\t\tOBJECT=Dimension_{i+1}")
        odl.append(f"\t\t\t\tDimensionName=\"{name}\"")
        odl.append(f"\t\t\t\tSize={size}")
        odl.append(f"\t\t\tEND_OBJECT=Dimension_{i+1}")
    odl.append("\t\tEND_GROUP=Dimension")
    odl.append("\t\tGROUP=DataField")
    odl.append(f"\t\t\tOBJECT=DataField_1")
    odl.append(f"\t\t\t\tDataFieldName=\"Spectra\"")
    odl.append(f"\t\t\t\tDataType=DFNT_FLOAT32")
    odl.append(f"\t\t\t\tDimList=(\"Time\",\"Band\",\"YDim\",\"XDim\")")
    odl.append(f"\t\t\tEND_OBJECT=DataField_1")
    odl.append("\t\tEND_GROUP=DataField")
    odl.append("\t\tGROUP=MergedFields")
    odl.append("\t\tEND_GROUP=MergedFields")
    odl.append(f"\tEND_GROUP=GRID_1")
    odl.append("END_GROUP=GridStructure")
    return "\n".join(odl)

def create_geoloc_vrt(hsi_path, igm_dataset_path, output_vrt, lon_band, lat_band):
    """
    Creates the Geoloc VRT.
    """
    ds = gdal.Open(hsi_path)
    if ds is None: raise ValueError(f"Could not open HSI: {hsi_path}")
    driver = gdal.GetDriverByName('VRT')
    vrt_ds = driver.CreateCopy(output_vrt, ds)
    
    safe_igm_path = igm_dataset_path.replace("\\", "/")

    # Robust WKT retrieval
    try:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wkt_4326 = srs.ExportToWkt()
    except Exception as e:
        print(f"  Warning: OSR import failed ({e}), using hardcoded WKT")
        wkt_4326 = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH],AUTHORITY["EPSG","4326"]]'

    # Check for empty WKT
    if not wkt_4326:
         wkt_4326 = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Longitude",EAST],AXIS["Latitude",NORTH],AUTHORITY["EPSG","4326"]]'

    geoloc_metadata = {
        'SRS': wkt_4326,
        'X_DATASET': safe_igm_path,
        'X_BAND': str(lon_band),
        'Y_DATASET': safe_igm_path,
        'Y_BAND': str(lat_band),
        'PIXEL_OFFSET': '0',
        'LINE_OFFSET': '0',
        'PIXEL_STEP': '1',
        'LINE_STEP': '1'
    }
    vrt_ds.SetMetadata(geoloc_metadata, 'GEOLOCATION')
    vrt_ds = None; ds = None
    return output_vrt

def process_stack():
    print("--- Starting ENVI+IGM Stacking Process (Robust WKT) ---")
    
    subdirs = [f.path for f in os.scandir(SOURCE_DIR) if f.is_dir()]
    if not subdirs: raise FileNotFoundError("No subdirectories found.")
    print(f"Found {len(subdirs)} subdirectories.")
    
    def get_date(fname):
        match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)', fname)
        return match.group(1) if match else "0000"
    
    subdirs.sort(key=lambda x: get_date(os.path.basename(x)))

    # --- DETERMINE RESOLUTION ---
    target_res = None
    print("Scanning for resolution from metadata...")
    for subdir_path in subdirs:
        files = os.listdir(subdir_path)
        igm_file_name = next((f for f in files if '_igm' in f and not f.endswith('.hdr') and not f.endswith('.xml')), None)
        if igm_file_name:
            igm_xml_name = igm_file_name + '.aux.xml'
            local_igm_xml = os.path.join(subdir_path, igm_xml_name)
            if not os.path.exists(local_igm_xml):
                 local_igm_xml = os.path.join(subdir_path, igm_file_name.replace('.igm', '.aux.xml'))
            if os.path.exists(local_igm_xml):
                res = parse_resolution_from_xml(local_igm_xml)
                if res:
                    target_res = res
                    print(f"  Detected native resolution: {target_res} meters")
                    break
    if target_res is None: target_res = 11.0

    (min_x, min_y, max_x, max_y), (width, height) = get_roi_bounds_projected(
        ROI_LON_MIN, ROI_LON_MAX, ROI_LAT_MIN, ROI_LAT_MAX, target_res, EPSG_CODE
    )
    print(f"Canvas: {width}x{height} pixels. Bounds: {min_x},{min_y} -> {max_x},{max_y}")

    temp_dir = tempfile.mkdtemp()
    print(f"Temporary working directory: {temp_dir}")
    
    h5_file = None
    h5_dset = None
    wavelengths = []
    units = "Unknown"
    valid_scenes = []

    try:
        for idx, subdir_path in enumerate(subdirs):
            folder_name = os.path.basename(subdir_path)
            print(f"Processing ({idx+1}/{len(subdirs)}): {folder_name}")
            
            files = os.listdir(subdir_path)
            hsi_file_name = next((f for f in files if f.endswith('.hsi')), None)
            igm_file_name = next((f for f in files if '_igm' in f and not f.endswith('.hdr') and not f.endswith('.xml')), None)
            
            if not hsi_file_name or not igm_file_name: raise FileNotFoundError(f"Missing .hsi or _igm in {folder_name}")
            
            local_hsi = os.path.join(subdir_path, hsi_file_name)
            local_igm = os.path.join(subdir_path, igm_file_name)
            
            hsi_xml_name = hsi_file_name + '.aux.xml'
            local_hsi_xml = os.path.join(subdir_path, hsi_xml_name)
            if not os.path.exists(local_hsi_xml):
                local_hsi_xml = os.path.join(subdir_path, hsi_file_name.replace('.hsi', '.aux.xml'))
            
            igm_xml_name = igm_file_name + '.aux.xml'
            local_igm_xml = os.path.join(subdir_path, igm_xml_name)
            if not os.path.exists(local_igm_xml):
                local_igm_xml = os.path.join(subdir_path, igm_file_name.replace('.igm', '.aux.xml'))

            if not wavelengths:
                if os.path.exists(local_hsi_xml):
                    wavelengths, units = parse_aux_xml_for_wavelengths(local_hsi_xml)
            
            lon_band, lat_band = detect_igm_bands(local_igm_xml)

            # --- STRATEGY: CREATE VALID TEMP IGM DATASET WITH HEADER & NODATA ---
            temp_igm_path = create_temp_igm_with_hdr(local_igm, local_igm_xml, temp_dir, idx)

            # Create Geoloc VRT pointing to Valid Temp IGM
            vrt_path = os.path.join(temp_dir, f"geoloc_{idx}.vrt")
            create_geoloc_vrt(local_hsi, temp_igm_path, vrt_path, lon_band, lat_band)
            
            # Warp
            warp_opts = gdal.WarpOptions(
                format='MEM',
                outputBounds=(min_x, min_y, max_x, max_y),
                xRes=target_res, yRes=target_res,
                dstSRS=f'EPSG:{EPSG_CODE}',
                dstNodata=-9999, 
                srcNodata=0,
                geoloc=True,
                resampleAlg=gdal.GRA_Bilinear,
                multithread=True
            )
            
            ds_warp = gdal.Warp('', vrt_path, options=warp_opts)
            if not ds_warp: raise RuntimeError(f"Warp failed for {folder_name}")
            data = ds_warp.ReadAsArray() 

            if data is not None and np.count_nonzero(data > 1e-6) == 0:
                print(f"  WARNING: Image is empty (zeros). {folder_name} likely does not overlap ROI.")

            # HDF5 Init
            if h5_file is None:
                n_bands = data.shape[0]
                n_scenes = len(subdirs)
                
                h5_file = h5py.File(OUTPUT_FILE, 'w')
                grp_grid = h5_file.create_group("HDFEOS/GRIDS/HYP")
                grp_data = grp_grid.create_group("Data Fields")
                
                c_time, c_band = 1, min(10, n_bands)
                c_height, c_width = min(128, height), min(128, width)
                
                print(f"  Creating Dataset: {(n_scenes, n_bands, height, width)}")
                h5_dset = grp_data.create_dataset(
                    "Spectra", shape=(n_scenes, n_bands, height, width),
                    maxshape=(None, n_bands, height, width),
                    dtype='float32', compression="gzip",
                    chunks=(c_time, c_band, c_height, c_width) 
                )
                h5_dset.attrs["_FillValue"] = -9999

            if h5_dset is not None:
                current_idx = len(valid_scenes)
                h5_dset[current_idx, :, :, :] = data
                valid_scenes.append({
                    "filename": folder_name,
                    "timestamp": get_date(folder_name)
                })
                print(f"  Processed scene {current_idx+1}")
            ds_warp = None
            
    finally:
        if h5_file and h5_dset is not None:
            final_count = len(valid_scenes)
            print(f"  Resizing dataset to {final_count} scenes...")
            h5_dset.resize(final_count, axis=0)
            
            h5_file.attrs["Title"] = "Time Series Stack"
            h5_file.attrs["Created"] = datetime.now().isoformat()
            
            if wavelengths:
                h5_dset.attrs["Wavelengths"] = np.array(wavelengths)
                h5_dset.attrs["WavelengthUnits"] = units
            
            struct_meta = generate_struct_metadata(
                "HYP", width, height, (min_x, max_y), (max_x, min_y), 
                final_count, h5_dset.shape[1]
            )
            grp_info = h5_file.create_group("HDFEOS INFORMATION")
            dt_str = h5py.string_dtype(encoding='ascii')
            dset_sm = grp_info.create_dataset("StructMetadata.0", (1,), dtype=dt_str)
            dset_sm[0] = struct_meta
            
            h5_file.close()
            print(f"\nSaved {final_count} scenes to {OUTPUT_FILE}")
        elif h5_file:
             h5_file.close()
        
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    process_stack()