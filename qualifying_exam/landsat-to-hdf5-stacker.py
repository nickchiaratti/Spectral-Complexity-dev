"""
Landsat Source Processor (HDF-EOS5 Compliant)

Description:
This script processes raw Landsat Level-2 Science Product (L2SP) data directly from source files 
(either unzipped folders or .tar archives) into a standardized, multi-temporal HDF-EOS5 compliant file. 
It performs spatial subsetting, radiometric rescaling, and reorganizes data into the HDF-EOS5 Grid model.

Key Changes for HDF-EOS5:
- File Structure: Data is stored under /HDFEOS/GRIDS/LANDSAT/Data Fields.
- Metadata: Generates 'StructMetadata.0' ODL block defining dimensions, projections (GCTP), and fields.
- Attributes: Adds 'HDFEOSVersion' and core HDF-EOS information.

Configuration:
- Location: Presets for ROI coordinates.
"""

import os
import numpy as np
import h5py
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from pyproj import Transformer, CRS
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
import json
from tqdm import tqdm
import tarfile

# --- CONFIGURATION ---

Location = "Tait-I-490"
if Location == "Rochester":
    ROI_LON_MIN = -77.72; ROI_LON_MAX = -77.4450
    ROI_LAT_MIN = 43.0450; ROI_LAT_MAX = 43.28
elif Location == "Tait":
    ROI_LON_MIN = -77.516127; ROI_LON_MAX = -77.461968
    ROI_LAT_MIN = 43.127698; ROI_LAT_MAX = 43.159168
elif Location == "RIT":
    ROI_LON_MIN = -77.688990; ROI_LON_MAX = -77.660365
    ROI_LAT_MIN = 43.072486; ROI_LAT_MAX = 43.093298
elif Location == "Tait-I-490":
    ROI_LON_MIN = -77.516127; ROI_LON_MAX = -77.4450
    ROI_LAT_MIN = 43.0450; ROI_LAT_MAX = 43.159168

SOURCE_DIR = "C:/satelliteImagery/LANDSAT/SourceData/Rochester"
LANDSAT_WAVELENGTHS = [0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201]
LANDSAT_BAND_NAMES = ["Coastal Aerosol", "Blue", "Green", "Red", "NIR", "SWIR 1", "SWIR 2"]
TARGET_BANDS_LIST = [1, 2, 3, 4, 5, 6, 7]
REFLECTANCE_MULT_BAND_SR = 2.75e-05
REFLECTANCE_ADD_BAND_SR = -0.2

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
    meta['wrs_path'] = int(img_attrs.find('WRS_PATH').text)
    meta['wrs_row'] = int(img_attrs.find('WRS_ROW').text)
    meta['utm_zone'] = int(proj_attrs.find('UTM_ZONE').text)
    
    band_filenames = {}
    for b in TARGET_BANDS_LIST:
        tag_name = f'FILE_NAME_BAND_{b}'
        filename = contents.find(tag_name).text
        band_filenames[b] = filename
    
    meta['qa_pixel_filename'] = contents.find('FILE_NAME_QUALITY_L1_PIXEL').text
    meta['qa_aerosol_filename'] = contents.find('FILE_NAME_QUALITY_L2_AEROSOL').text
    meta['qa_radsat_filename'] = contents.find('FILE_NAME_QUALITY_L1_RADIOMETRIC_SATURATION').text
    meta['band_filenames'] = band_filenames
    return meta

def calculate_utm_zone(lon):
    """Calculates UTM zone from longitude."""
    return int((lon + 180) / 6) + 1

def calculate_target_grid(roi_bounds, force_utm_zone=None):
    """
    Calculates the target Affine transform and dimensions for the grid.
    If force_utm_zone is provided, it constructs the CRS for that zone.
    Otherwise, it calculates the best zone from the ROI center.
    """
    lon_min, lat_min, lon_max, lat_max = roi_bounds
    
    # Determine UTM Zone
    center_lon = (lon_min + lon_max) / 2
    utm_zone = int((center_lon + 180) / 6) + 1
    
    # Construct Destination CRS (WGS84 UTM)
    # Assumes Northern Hemisphere for now based on context (NY state)
    # A robust solution checks latitude.
    is_north = lat_min >= 0
    epsg_code = 32600 + utm_zone
    if not is_north: epsg_code += 100 # Adjust for southern hemisphere if needed
    dst_crs = CRS.from_epsg(epsg_code)

    
    
    # Transform Corners to Destination CRS
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs = [lon_min, lon_max, lon_max, lon_min]
    ys = [lat_min, lat_min, lat_max, lat_max]
    xx, yy = transformer.transform(xs, ys)
    
    dst_min_x = min(xx); dst_max_x = max(xx)
    dst_min_y = min(yy); dst_max_y = max(yy)
    
    # 30m Grid
    width_30 = int(np.ceil((dst_max_x - dst_min_x) / 30.0))
    height_30 = int(np.ceil((dst_max_y - dst_min_y) / 30.0))
    # Note: from_bounds args are (west, south, east, north, width, height)
    # Ensure north is max_y and south is min_y
    transform_30 = from_bounds(dst_min_x, dst_min_y, dst_max_x, dst_max_y, width_30, height_30)
    
    return dst_crs, transform_30, width_30, height_30, (dst_min_x, dst_max_y), (dst_max_x, dst_min_y), utm_zone

def get_raster_handle(source_type, source_path, filename_inside=None):

    return rasterio.open(f"/vsitar/{source_path.replace('\\', '/')}/{filename_inside}")

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
        SphereCode=12
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
            if f.endswith(".tar") and "LC0" in f: 
                scenes.append({'type': 'tar', 'path': os.path.join(root, f)})

    if not scenes: print("No scenes found."); return

    # 2. Parse Metadata
    scenes_data = []
    for scene in tqdm(scenes, desc="Parsing Metadata"):
        try:
            with tarfile.open(scene['path'], 'r') as tar:
                for m in tar.getmembers():
                    if m.name.endswith("MTL.xml"):
                        content = tar.extractfile(m).read().decode('utf-8')
                        break
            
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
    
    # Determine forced UTM zone if applicable
    roi_bounds = (ROI_LON_MIN, ROI_LAT_MIN, ROI_LON_MAX, ROI_LAT_MAX)
    dst_crs, tf_30, w_30, h_30, ul_coords, lr_coords, zone_code = calculate_target_grid(roi_bounds, force_utm_zone=None)
    
    print(f"Target Grid: {w_30}x{h_30} in UTM Zone {zone_code}")

    
    # 4. Initialize HDF-EOS5 File
    print(f"\nInitializing HDF-EOS5: {output_path}")
    
    
    with h5py.File(output_path, 'w') as f:
        # HDFEOS Information
        info_grp = f.create_group("HDFEOS INFORMATION")
        info_grp.attrs["HDFEOSVersion"] = "HDFEOS_5.1.16"
        
        # Structural Metadata
        struct_meta = generate_struct_metadata("LANDSAT", w_30, h_30, ul_coords, lr_coords, zone_code, len(TARGET_BANDS_LIST), len(scenes_data))
        # HDF-EOS expects string datasets for this, sometimes fixed length
        dt = h5py.string_dtype(encoding='ascii')
        dset_sm = info_grp.create_dataset("StructMetadata.0", shape=(1,), dtype=dt, data=struct_meta)
        
        # Grid Structure
        grid_grp = f.create_group("HDFEOS/GRIDS/LANDSAT")
        data_grp = grid_grp.create_group("Data Fields")
        
        # Data Datasets
        # Use float32, apply scaling, set fill to NaN
        dset_sr = data_grp.create_dataset('surface_reflectance', 
                                shape=(len(scenes_data), len(TARGET_BANDS_LIST), h_30, w_30),
                                dtype='float32', compression='gzip', fillvalue=np.nan)
        
        # QA Pixel Dataset
        dset_pixel_qa = data_grp.create_dataset('QUALITY_L1_PIXEL',
                                shape=(len(scenes_data), h_30, w_30),
                                dtype='uint16', compression='gzip', fillvalue=1)

        # QA Aerosol Dataset
        dset_aerosol_qa = data_grp.create_dataset('QUALITY_L2_AEROSOL',
                                shape=(len(scenes_data), h_30, w_30),
                                dtype='uint8', compression='gzip', fillvalue=1)

        # Radiometric Saturation Dataset
        dset_radsat = data_grp.create_dataset('RADIOMETRIC_SATURATION',
                                shape=(len(scenes_data), h_30, w_30),
                                dtype='uint16', compression='gzip', fillvalue=np.nan)

        # Attach attributes to dataset
        dset_sr.attrs['units'] = "Reflectance"
        dset_sr.attrs['_FillValue'] = np.nan # Updated for float
        dset_sr.attrs['wavelengths'] = LANDSAT_WAVELENGTHS
        
        # --- Standardized Spatial Metadata ---
        # Write CRS as a standard UTF-8 string natively
        dset_sr.attrs['spatial_ref'] = dst_crs.to_wkt()
        
        # Write the Transform in native Rasterio Affine order (a, b, c, d, e, f)
        rasterio_transform = [tf_30.a, tf_30.b, tf_30.c, tf_30.d, tf_30.e, tf_30.f]
        dset_sr.attrs['GeoTransform'] = np.array(rasterio_transform, dtype='float64')
        
        # --- NEW Attributes for Metadata ---
        acq_times = []
        spacecraft_ids = []
        sun_azimuths = []
        sun_elevations = []
        wrs_paths = []
        wrs_rows = []
        cloud_covers = []

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
            wrs_paths.append(scene['wrs_path'])
            wrs_rows.append(scene['wrs_row'])
            cloud_covers.append(scene['cloud_cover'])
            
            # Multispectral
            for b_idx, band_num in enumerate(TARGET_BANDS_LIST):
                try:
                    with get_raster_handle(scene['source_type'], scene['source_path'], scene['band_filenames'][band_num]) as src:
                        # Temporary UINT16 array for warping
                        temp_uint = np.zeros((h_30, w_30), dtype='uint16')
                        reproject(rasterio.band(src, 1), temp_uint, src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=tf_30, dst_crs=dst_crs, 
                                  resampling=Resampling.cubic, 
                                  src_nodata=0, dst_nodata=0,)
                        
                        # Apply scaling and convert to float
                        dest_float = np.full((h_30, w_30), np.nan, dtype='float32')
                        valid_mask = (temp_uint != 0) 
                        
                        if np.any(valid_mask):
                             scaled = (temp_uint[valid_mask].astype(np.float32) * REFLECTANCE_MULT_BAND_SR) + REFLECTANCE_ADD_BAND_SR
                             dest_float[valid_mask] = np.clip(scaled, 0.0, 1.0)

                        dset_sr[t, b_idx, :, :] = dest_float
                except Exception as e: print(f"Band error frame {t}: {e}")

            # QA Pixel
            if scene.get('qa_pixel_filename'):
                try:
                    with get_raster_handle(scene['source_type'], scene['source_path'], scene['qa_pixel_filename']) as src:
                        # Initialize with 0 (Background)
                        qa_uint = np.zeros((h_30, w_30), dtype='uint16')
                        # Important: Nearest neighbor for bitmask
                        reproject(rasterio.band(src, 1), qa_uint, src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=tf_30, dst_crs=dst_crs, 
                                  resampling=Resampling.nearest,
                                  src_nodata=1, dst_nodata=1,)
                        dset_pixel_qa[t, :, :] = qa_uint
                except Exception as e: print(f"QA write error frame {t}: {e}")

            # QA Aerosol
            if scene.get('qa_aerosol_filename'):
                try:
                    with get_raster_handle(scene['source_type'], scene['source_path'], scene['qa_aerosol_filename']) as src:
                        # Initialize with 0 (Background)
                        qa_uint = np.zeros((h_30, w_30), dtype='uint8')
                        # Important: Nearest neighbor for bitmask
                        reproject(rasterio.band(src, 1), qa_uint, src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=tf_30, dst_crs=dst_crs, 
                                  resampling=Resampling.nearest,
                                  src_nodata=1, dst_nodata=1,)
                        dset_aerosol_qa[t, :, :] = qa_uint
                except Exception as e: print(f"QA write error frame {t}: {e}")

            if scene.get('qa_radsat_filename'):
                try:
                    with get_raster_handle(scene['source_type'], scene['source_path'], scene['qa_radsat_filename']) as src:
                        # Initialize with 0 (Background)
                        qa_uint = np.zeros((h_30, w_30), dtype='uint16')
                        # Important: Nearest neighbor for bitmask
                        reproject(rasterio.band(src, 1), qa_uint, src_transform=src.transform, src_crs=src.crs,
                                  dst_transform=tf_30, dst_crs=dst_crs, 
                                  resampling=Resampling.nearest)
                        dset_radsat[t, :, :] = qa_uint
                except Exception as e: print(f"QA write error frame {t}: {e}")

        # Write collected attributes to dataset
        dset_sr.attrs['acquisition_time'] = np.array(acq_times, dtype='float64') # Stored as float64
        dset_sr.attrs['spacecraft_id'] = spacecraft_ids
        dset_sr.attrs['sun_azimuth'] = np.array(sun_azimuths, dtype='float32')
        dset_sr.attrs['sun_elevation'] = np.array(sun_elevations, dtype='float32')
        dset_sr.attrs['wrs_path'] = np.array(wrs_paths, dtype='int8')
        dset_sr.attrs['wrs_row'] = np.array(wrs_rows, dtype='int8')
        dset_sr.attrs['cloud_cover'] = np.array(cloud_covers, dtype='float32')

    print("\nProcessing Complete.")

if __name__ == '__main__':
    
    if SOURCE_DIR:
        out_dir = f"C:/satelliteImagery/LANDSAT/{Location}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = f"{out_dir}/LANDSAT_Stack_{Location}_HDFEOS.h5"
        
        process_landsat_stack(SOURCE_DIR, out_file)