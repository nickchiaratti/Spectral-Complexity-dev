"""
Landsat Source Processor (HDF-EOS5 Compliant)

Description:
This script processes raw Landsat Level-2 Science Product (L2SP) data directly from source files 
(either unzipped folders or .tar archives) into a standardized, multi-temporal HDF-EOS5 compliant file. 
It performs spatial subsetting, radiometric rescaling, and reorganizes data into the HDF-EOS5 Grid model.

Key Changes for HDF-EOS5:
- File Structure: Data is stored under /HDFEOS/GRIDS/<GridName>/Data Fields.
- Metadata: Generates 'StructMetadata.0' ODL block defining dimensions, projections (GCTP), and fields.
- Attributes: Adds 'HDFEOSVersion' and core HDF-EOS information.

Configuration:
- Location: Presets for ROI coordinates.
- INCLUDE_PAN_BAND: Toggle for 15m Panchromatic Band.
- QA_MASK_THRESHOLD_PERCENT: Quality filter threshold.
"""

import os
import numpy as np
import h5py
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from pyproj import Transformer, CRS
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from tqdm import tqdm
import tarfile

# --- CONFIGURATION ---

Location = "Tait-Tight"
INCLUDE_PAN_BAND = False 
QA_MASK_THRESHOLD_PERCENT = 0.0

# Bounding Box (Longitude/Latitude)
if Location == "Rochester":
    ROI_LON_MIN = -77.72; ROI_LON_MAX = -77.50
    ROI_LAT_MIN = 43.08; ROI_LAT_MAX = 43.28
elif Location == "Tait":
    ROI_LON_MIN = -77.516127; ROI_LON_MAX = -77.461968
    ROI_LAT_MIN = 43.127698; ROI_LAT_MAX = 43.159168
elif Location == "Tait-Tight":
    ROI_LON_MIN = -77.510594; ROI_LON_MAX = -77.497333
    ROI_LAT_MIN = 43.137844; ROI_LAT_MAX = 43.148929
elif Location == "RIT":
    ROI_LON_MIN = -77.688990; ROI_LON_MAX = -77.660365
    ROI_LAT_MIN = 43.072486; ROI_LAT_MAX = 43.093298
elif Location == "Seabreeze":
    ROI_LON_MIN = -77.556403; ROI_LON_MAX = -77.522049
    ROI_LAT_MIN = 43.223786; ROI_LAT_MAX = 43.242186
elif Location == "LakeOntario":
    ROI_LON_MIN = -77.560544; ROI_LON_MAX = -77.520075
    ROI_LAT_MIN = 43.223786; ROI_LAT_MAX = 43.241007
elif Location == "Davis":
    ROI_LON_MIN = -121.792782; ROI_LON_MAX = -121.746691
    ROI_LAT_MIN = 38.522226; ROI_LAT_MAX = 38.547101

LANDSAT_WAVELENGTHS = [0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201]
LANDSAT_BAND_NAMES = ["Coastal Aerosol", "Blue", "Green", "Red", "NIR", "SWIR 1", "SWIR 2"]
TARGET_BANDS_LIST = [1, 2, 3, 4, 5, 6, 7]
REFLECTANCE_MULT_BAND_SR = 2.75e-05
REFLECTANCE_ADD_BAND_SR = -0.2

# --- GCTP Constants ---
HE5_GCTP_GEO = 0
HE5_GCTP_UTM = 1
HE5_GCTP_WGS84 = 12

# --- QA Bitmasks ---
# Bit 0: Fill (1 = Fill, 0 = Image)
# Bit 1: Dilated Cloud
# Bit 2: Cirrus
# Bit 3: Cloud
# Bit 4: Cloud Shadow
# Bit 5: Snow
QA_BAD_MASK = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)

def parse_mtl_xml_content(xml_content_str):
    root = ET.fromstring(xml_content_str)
    meta = {}
    contents = root.find('PRODUCT_CONTENTS')
    img_attrs = root.find('IMAGE_ATTRIBUTES')
    proj_attrs = root.find('PROJECTION_ATTRIBUTES')
    l1_record = root.find('LEVEL1_PROCESSING_RECORD')
    l1_rad = root.find('LEVEL1_RADIOMETRIC_RESCALING')
    
    meta['product_id'] = contents.find('LANDSAT_PRODUCT_ID').text
    meta['acquisition_date'] = img_attrs.find('DATE_ACQUIRED').text
    meta['scene_center_time'] = img_attrs.find('SCENE_CENTER_TIME').text
    meta['sensor_id'] = img_attrs.find('SENSOR_ID').text
    meta['spacecraft_id'] = img_attrs.find('SPACECRAFT_ID').text
    meta['cloud_cover'] = float(img_attrs.find('CLOUD_COVER').text)
    meta['sun_azimuth'] = float(img_attrs.find('SUN_AZIMUTH').text)
    meta['sun_elevation'] = float(img_attrs.find('SUN_ELEVATION').text)
    meta['utm_zone'] = proj_attrs.find('UTM_ZONE').text
    
    band_filenames = {}
    for b in TARGET_BANDS_LIST:
        tag_name = f'FILE_NAME_BAND_{b}'
        filename = contents.find(tag_name).text
        band_filenames[b] = filename
    
    qa_filename = contents.find('FILE_NAME_QUALITY_L1_PIXEL')
    if qa_filename is None:
         for child in contents:
             if child.text and child.text.endswith('QA_PIXEL.TIF'):
                 qa_filename = child; break
    meta['qa_filename'] = qa_filename.text if qa_filename is not None else None
    meta['band_filenames'] = band_filenames
    return meta

def calculate_target_grid(roi_bounds, src_crs_wkt):
    lon_min, lat_min, lon_max, lat_max = roi_bounds
    dst_crs = CRS.from_wkt(src_crs_wkt)
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs = [lon_min, lon_max, lon_max, lon_min]
    ys = [lat_min, lat_min, lat_max, lat_max]
    xx, yy = transformer.transform(xs, ys)
    dst_min_x = min(xx); dst_max_x = max(xx)
    dst_min_y = min(yy); dst_max_y = max(yy)
    
    # 30m Grid
    width_30 = int(np.ceil((dst_max_x - dst_min_x) / 30.0))
    height_30 = int(np.ceil((dst_max_y - dst_min_y) / 30.0))
    transform_30 = from_bounds(dst_min_x, dst_min_y, dst_max_x, dst_max_y, width_30, height_30)
    
    return dst_crs, transform_30, width_30, height_30, (dst_min_x, dst_max_y), (dst_max_x, dst_min_y)

def get_raster_handle(source_type, source_path, filename_inside=None):
    if source_type == 'dir':
        return rasterio.open(os.path.join(source_path, filename_inside))
    elif source_type == 'tar':
        return rasterio.open(f"/vsitar/{source_path.replace('\\', '/')}/{filename_inside}")
    return None

def generate_struct_metadata(grid_name, x_dim, y_dim, ul_coords, lr_coords, zone_code, num_bands, num_frames):
    """
    Generates the ODL StructMetadata.0 string required for HDF-EOS5.
    Assumes UTM projection (HE5_GCTP_UTM) and WGS84 (HE5_GCTP_WGS84).
    """
    # 13 projection params, mostly 0 for UTM
    proj_params = tuple([0.0] * 13) 
    
    # ODL Template
    odl = f"""GROUP=SwathStructure
END_GROUP=SwathStructure
GROUP=GridStructure
    GROUP=GRID_1
        GridName="{grid_name}"
        XDim={x_dim}
        YDim={y_dim}
        UpperLeftPointMtrs=({ul_coords[0]:.6f},{ul_coords[1]:.6f})
        LowerRightMtrs=({lr_coords[0]:.6f},{lr_coords[1]:.6f})
        Projection=GCTP_UTM
        ZoneCode={zone_code}
        SphereCode={HE5_GCTP_WGS84}
        ProjParams={str(proj_params).replace(' ', '').replace('(', '').replace(')', '')}
        
        GROUP=Dimension
            OBJECT=Dimension_1
                DimensionName="Time"
                Size={num_frames}
            END_OBJECT=Dimension_1
            OBJECT=Dimension_2
                DimensionName="Bands"
                Size={num_bands}
            END_OBJECT=Dimension_2
            OBJECT=Dimension_3
                DimensionName="YDim"
                Size={y_dim}
            END_OBJECT=Dimension_3
            OBJECT=Dimension_4
                DimensionName="XDim"
                Size={x_dim}
            END_OBJECT=Dimension_4
        END_GROUP=Dimension
        
        GROUP=DataField
            OBJECT=DataField_1
                DataFieldName="surface_reflectance"
                DataType=HDF5T_NATIVE_FLOAT
                DimList=("Time","Bands","YDim","XDim")
            END_OBJECT=DataField_1
            OBJECT=DataField_2
                DataFieldName="QUALITY_L1_PIXEL"
                DataType=HDF5T_NATIVE_UINT16
                DimList=("Time","YDim","XDim")
            END_OBJECT=DataField_2
    """
    odl += """
        END_GROUP=DataField
        GROUP=MergedFields
        END_GROUP=MergedFields
    END_GROUP=GRID_1
END_GROUP=GridStructure
GROUP=PointStructure
END_GROUP=PointStructure
GROUP=ZaStructure
END_GROUP=ZaStructure
END
"""
    return odl

def process_landsat_stack(root_dir, output_path):
    # 1. Find Scenes
    print(f"Scanning {root_dir}...")
    scenes = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith("MTL.xml"): scenes.append({'type': 'dir', 'path': root, 'mtl_file': f})
            elif f.endswith(".tar") and "LC0" in f: scenes.append({'type': 'tar', 'path': os.path.join(root, f)})

    if not scenes: print("No scenes found."); return

    # 2. Parse Metadata
    scenes_data = []
    for scene in tqdm(scenes, desc="Parsing Metadata"):
        try:
            if scene['type'] == 'dir':
                with open(os.path.join(scene['path'], scene['mtl_file']), 'r') as f: content = f.read()
            else:
                with tarfile.open(scene['path'], 'r') as tar:
                    mtl = next(m for m in tar.getmembers() if m.name.endswith("MTL.xml"))
                    content = tar.extractfile(mtl).read().decode('utf-8')
            
            meta = parse_mtl_xml_content(content)
            meta['source_type'] = scene['type']
            meta['source_path'] = scene['path']
            scenes_data.append(meta)
        except Exception as e: print(f"Error parsing {scene['path']}: {e}")

    unique_scenes = {s['product_id']: s for s in scenes_data}.values()
    scenes_data = sorted(list(unique_scenes), key=lambda x: x['acquisition_date'])
    
    if not scenes_data: return

    # 3. Define Grid
    print("\nDefining Grid...")
    first_meta = scenes_data[0]
    with get_raster_handle(first_meta['source_type'], first_meta['source_path'], first_meta['band_filenames'][1]) as src:
        src_crs = src.crs
        src_crs_wkt = src_crs.to_wkt()
        # Extract UTM Zone for metadata
        try:
             # Basic UTM zone extraction for EPSG
             utm_zone = src_crs.to_dict().get('zone', 0)
        except:
             utm_zone = 0

    dst_crs, tf_30, w_30, h_30, ul_coords, lr_coords = calculate_target_grid(
        (ROI_LON_MIN, ROI_LAT_MIN, ROI_LON_MAX, ROI_LAT_MAX), src_crs_wkt)

    # 4. Filter by QA
    valid_scenes = []
    print("\nChecking QA Masks...")
    for scene in tqdm(scenes_data, desc="QA Filtering"):
        if not scene.get('qa_filename'): 
            print(f"Skipping {scene['product_id']} - No QA Pixel file.")
            continue
        try:
            with get_raster_handle(scene['source_type'], scene['source_path'], scene['qa_filename']) as src:
                # Initialize with 0 (Script Background)
                qa = np.zeros((h_30, w_30), dtype='uint16')
                # Warp uses 0 as nodata fill for areas outside source
                reproject(rasterio.band(src, 1), qa, src_transform=src.transform, src_crs=src.crs, 
                          dst_transform=tf_30, dst_crs=dst_crs, resampling=Resampling.nearest)
                
                # --- Valid Pixel Logic ---
                # 0 = Background (from reproject initialization)
                # 1 = Fill (from USGS source)
                # Anything > 1 is a valid pixel with actual data
                valid_mask = (qa > 1)
                
                total_valid_pixels = np.count_nonzero(valid_mask)
                
                # If scene has no valid pixels in ROI, skip it
                if total_valid_pixels == 0:
                    continue
                
                # --- Bad Pixel Logic ---
                # Check for Cloud/Shadow/Snow bits only on valid pixels
                is_bad_quality = (qa & QA_BAD_MASK) != 0
                bad_pixels = np.count_nonzero(valid_mask & is_bad_quality)
                
                bad_pct = (bad_pixels / total_valid_pixels) * 100.0
                
                if bad_pct <= QA_MASK_THRESHOLD_PERCENT: 
                    valid_scenes.append(scene)

        except Exception as e: print(f"QA Error {scene.get('product_id', 'unknown')}: {e}")

    scenes_data = valid_scenes
    print(f"QA Filter Results: {len(scenes_data)} scenes passed.")
    if not scenes_data: print("No scenes passed QA."); return
    
    # 5. Initialize HDF-EOS5 File
    print(f"\nInitializing HDF-EOS5: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # HDFEOS Information
        info_grp = f.create_group("HDFEOS INFORMATION")
        info_grp.attrs["HDFEOSVersion"] = "HDFEOS_5.1.16"
        
        # Structural Metadata
        struct_meta = generate_struct_metadata("LANDSAT", w_30, h_30, ul_coords, lr_coords, utm_zone, len(TARGET_BANDS_LIST), len(scenes_data))
        # HDF-EOS expects string datasets for this, sometimes fixed length
        dt = h5py.string_dtype(encoding='ascii')
        dset_sm = info_grp.create_dataset("StructMetadata.0", shape=(1,), dtype=dt, data=struct_meta)
        
        # Grid Structure
        grid_grp = f.create_group("HDFEOS/GRIDS/LANDSAT")
        data_grp = grid_grp.create_group("Data Fields")
        
        # Data Datasets
        # Use float32, apply scaling, set fill to NaN
        # --- RENAMED: dset_ms -> dset_sr ---
        dset_sr = data_grp.create_dataset('surface_reflectance', 
                                shape=(len(scenes_data), len(TARGET_BANDS_LIST), h_30, w_30),
                                dtype='float32', compression='gzip', fillvalue=np.nan)
        
        # QA Pixel Dataset
        dset_qa = data_grp.create_dataset('QUALITY_L1_PIXEL',
                                shape=(len(scenes_data), h_30, w_30),
                                dtype='uint16', compression='gzip', fillvalue=0)

        # Attach attributes to dataset
        dset_sr.attrs['units'] = "Reflectance"
        dset_sr.attrs['_FillValue'] = np.nan # Updated for float
        dset_sr.attrs['wavelengths'] = LANDSAT_WAVELENGTHS
        
        # --- NEW Attributes for Metadata ---
        acq_times = []
        spacecraft_ids = []
        sun_azimuths = []
        sun_elevations = []

        # 6. Process Data
        for t, scene in enumerate(tqdm(scenes_data, desc="Writing Data")):
            # Collect Metadata
            date_str = scene['acquisition_date']
            time_str = scene['scene_center_time'].replace('Z', '')
            if '.' in time_str:
                parts = time_str.split('.')
                if len(parts[1]) > 6:
                    time_str = parts[0] + '.' + parts[1][:6]
            
            full_timestamp_str = f"{date_str}T{time_str}"
            
            # --- CONVERT TO UNIX TIMESTAMP (FLOAT) ---
            try:
                # Assuming UTC 'Z' or native naive as UTC
                dt = datetime.strptime(full_timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
                dt = dt.replace(tzinfo=timezone.utc)
                unix_ts = dt.timestamp()
            except ValueError:
                # Fallback if no fractional seconds
                dt = datetime.strptime(full_timestamp_str, "%Y-%m-%dT%H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                unix_ts = dt.timestamp()

            acq_times.append(unix_ts)
            spacecraft_ids.append(scene['spacecraft_id'])
            sun_azimuths.append(scene['sun_azimuth'])
            sun_elevations.append(scene['sun_elevation'])
            
            # Multispectral
            for b_idx, band_num in enumerate(TARGET_BANDS_LIST):
                try:
                    with get_raster_handle(scene['source_type'], scene['source_path'], scene['band_filenames'][band_num]) as src:
                        # Temporary UINT16 array for warping
                        temp_uint = np.zeros((h_30, w_30), dtype='uint16')
                        reproject(rasterio.band(src, 1), temp_uint, src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=tf_30, dst_crs=dst_crs, resampling=Resampling.cubic)
                        
                        # Apply scaling and convert to float
                        dest_float = np.full((h_30, w_30), np.nan, dtype='float32')
                        valid_mask = (temp_uint != 0) 
                        
                        if np.any(valid_mask):
                             scaled = (temp_uint[valid_mask].astype(np.float32) * REFLECTANCE_MULT_BAND_SR) + REFLECTANCE_ADD_BAND_SR
                             dest_float[valid_mask] = np.maximum(scaled, 0.0)

                        dset_sr[t, b_idx, :, :] = dest_float
                except Exception as e: print(f"Band error frame {t}: {e}")

            # QA Pixel
            if scene.get('qa_filename'):
                try:
                    with get_raster_handle(scene['source_type'], scene['source_path'], scene['qa_filename']) as src:
                        # Initialize with 0 (Background)
                        qa_uint = np.zeros((h_30, w_30), dtype='uint16')
                        # Important: Nearest neighbor for bitmask
                        reproject(rasterio.band(src, 1), qa_uint, src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=tf_30, dst_crs=dst_crs, resampling=Resampling.nearest)
                        dset_qa[t, :, :] = qa_uint
                except Exception as e: print(f"QA write error frame {t}: {e}")

        # Write collected attributes to dataset
        dset_sr.attrs['acquisition_time'] = np.array(acq_times, dtype='float64') # Stored as float64
        dset_sr.attrs['spacecraft_id'] = np.array(spacecraft_ids, dtype='S20')
        dset_sr.attrs['sun_azimuth'] = np.array(sun_azimuths, dtype='float32')
        dset_sr.attrs['sun_elevation'] = np.array(sun_elevations, dtype='float32')

    print("\nProcessing Complete.")

if __name__ == '__main__':
    in_dir = "C:/satelliteImagery/LANDSAT/SourceData"
    if in_dir:
        out_file = os.path.join(in_dir, f"{Location}_HDFEOS.h5")
        process_landsat_stack(in_dir, out_file)