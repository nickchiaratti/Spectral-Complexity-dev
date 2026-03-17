import ee
import geemap
import h5py
import rasterio
import numpy as np
import os
import glob
from datetime import datetime, timezone
import shutil

# ==========================================
# 1. CONFIGURATION
# ==========================================
try:
    ee.Initialize(project="project-ee18dbee-cd7e-4d08-812")
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

Location = "Tait"
if Location == "Rochester":
    # Rochester Bounding Box
    ROI = ee.Geometry.Rectangle([-77.72, 43.0450, -77.4450, 43.28])
    START_DATE = '2015-01-01'
    END_DATE = '2025-12-31'
    OUTPUT_HDF5 = "C:/satelliteImagery/LANDSAT/Rochester/LANDSAT_Stack_Rochester_GEE_2015_2025.h5"
    TEMP_DIR = "C:/satelliteImagery/Rochester_TEMP_GEE_DOWNLOAD"
elif Location == "Tait":
    # Tait Bounding Box
    ROI = ee.Geometry.Rectangle([-77.516127, 43.127698, -77.461968, 43.159168])
    START_DATE = '2015-01-01'
    END_DATE = '2025-12-31'
    OUTPUT_HDF5 = "C:/satelliteImagery/LANDSAT/Tait/LANDSAT_Stack_Tait_GEE_2015_2025.h5"
    TEMP_DIR = "C:/satelliteImagery/Tait_TEMP_GEE_DOWNLOAD"

# Target Coordinate Reference System (UTM Zone 18N for Rochester)
TARGET_CRS = 'EPSG:32618'


# Landsat 8/9 SR Bands to extract, plus all required QA bands
BANDS = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT', 'SR_QA_AEROSOL']
LANDSAT_WAVELENGTHS = [0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201]

# ==========================================
# 2. HDF-EOS5 STRUCTURAL UTILITY
# ==========================================
def generate_struct_metadata(grid_name, x_dim, y_dim, ul_coords, lr_coords, zone_code, num_bands, num_frames):
    """Generates the ODL StructMetadata.0 string required for HDF-EOS5 compliance."""
    proj_params = tuple([0.0] * 13) 
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

# ==========================================
# 3. GEE SERVER-SIDE PROCESSING
# ==========================================
def apply_scale_factors(image):
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(optical_bands, None, True)

def calculate_roi_cloud_cover(image):
    """Calculates the percentage of cloud/shadow specifically within the ROI."""
    qa = image.select('QA_PIXEL')
    
    # Bit 3 = Cloud, Bit 4 = Cloud Shadow
    cloud_shadow_bit_mask = (1 << 4)
    clouds_bit_mask = (1 << 3)
    
    # Create a binary mask where 1 = cloud/shadow, 0 = clear
    is_cloudy = qa.bitwiseAnd(cloud_shadow_bit_mask).neq(0) \
        .Or(qa.bitwiseAnd(clouds_bit_mask).neq(0))
        
    stats = is_cloudy.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ROI,
        scale=40,
        maxPixels=1e9
    )
    
    cloud_fraction = stats.get('QA_PIXEL')
    roi_cc = ee.Algorithms.If(
        cloud_fraction, 
        ee.Number(cloud_fraction).multiply(100), 
        100 
    )
    
    return image.set('roi_cloud_cover', roi_cc)

print("Querying Google Earth Engine for Landsat 8/9 Collection 2 SR...")
collection_l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
collection_l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
merged_collection = collection_l8.merge(collection_l9)

filtered_collection = merged_collection \
    .filterBounds(ROI) \
    .filterDate(START_DATE, END_DATE) \
    .map(calculate_roi_cloud_cover) \
    .filter(ee.Filter.lt('roi_cloud_cover', 30)) \
    .map(apply_scale_factors) \
    .select(BANDS) \
    .sort('system:time_start')

num_images = filtered_collection.size().getInfo()
print(f"Found {num_images} images matching criteria. Fetching metadata dictionary...")

info = filtered_collection.getInfo()
gee_metadata = {}
for feature in info['features']:
    img_id = feature['id'].split('/')[-1] 
    props = feature['properties']
    
    # --- TEMPORAL TRUTH DERIVATION ---
    # Convert universally guaranteed system:time_start (ms) to mathematically exact date/time strings
    time_start_ms = props.get('system:time_start')
    unix_time_sec = time_start_ms / 1000.0
    
    gee_metadata[img_id] = {
        'acquisition_time': unix_time_sec, # Stored as float for unix timestamp
        'spacecraft_id': props.get('SPACECRAFT_ID'),
        'sun_azimuth': props.get('SUN_AZIMUTH'),
        'sun_elevation': props.get('SUN_ELEVATION'),
        'wrs_path': props.get('WRS_PATH'),
        'wrs_row': props.get('WRS_ROW'),
        'cloud_cover': props.get('roi_cloud_cover', props.get('CLOUD_COVER'))
    }

# ==========================================
# 4. DOWNLOAD & TRANSLATE TO HDF5
# ==========================================
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

print(f"Downloading GeoTIFFs (Forced to {TARGET_CRS} Euclidean Grid)...")

for feature in info['features']:
    img_id = feature['id']
    short_name = img_id.split('/')[-1]
    out_tif = os.path.join(TEMP_DIR, f"{short_name}.tif")
    
    if os.path.exists(out_tif) and os.path.getsize(out_tif) > 0:
        print(f"Skipping {short_name} - already downloaded.")
        continue
        
    print(f"Downloading {short_name}...")
    try:
        image = ee.Image(img_id)
        processed_image = apply_scale_factors(image).select(BANDS)
        geemap.ee_export_image(processed_image, filename=out_tif, scale=30, region=ROI, crs=TARGET_CRS)
    except Exception as e:
        print(f"Failed to download {short_name}: {e}")

print(f"\nTranslating downloaded GeoTIFFs to HDF5: {OUTPUT_HDF5}")
tif_files = sorted(glob.glob(os.path.join(TEMP_DIR, "*.tif")))

if len(tif_files) == 0:
    print("Error: No files downloaded.")
    exit()

with rasterio.open(tif_files[0]) as src:
    height, width = src.height, src.width
    num_bands = src.count
    dst_crs_wkt = src.crs.to_wkt()
    dst_transform = src.transform
    bounds = src.bounds
    ul_coords = (bounds.left, bounds.top)
    lr_coords = (bounds.right, bounds.bottom)
    utm_zone = int(TARGET_CRS.split(':')[-1]) - 32600 

stacked_data = np.zeros((len(tif_files), num_bands, height, width), dtype=np.float32)

acq_times, spacecraft_ids, sun_azimuths = [], [], []
sun_elevations, wrs_paths, wrs_rows, cloud_covers = [], [], [], []

for idx, tif_path in enumerate(tif_files):
    filename = os.path.basename(tif_path).replace('.tif', '')

    # STRICT METADATA VALIDATION
    meta = gee_metadata.get(filename)
    if meta is None:
        raise ValueError(f"CRITICAL ERROR: Metadata for downloaded file '{filename}' not found in GEE response. Halting to preserve dataset integrity.")
    
    for key, value in meta.items():
        if value is None:
             raise ValueError(f"CRITICAL ERROR: Metadata attribute '{key}' for '{filename}' is missing/null from GEE. Halting pipeline.")

    acq_times.append(meta['acquisition_time'])
    spacecraft_ids.append(meta['spacecraft_id'])
    sun_azimuths.append(meta['sun_azimuth'])
    sun_elevations.append(meta['sun_elevation'])
    wrs_paths.append(meta['wrs_path'])
    wrs_rows.append(meta['wrs_row'])
    cloud_covers.append(meta['cloud_cover'])
    
    with rasterio.open(tif_path) as src:
        stacked_data[idx] = src.read()

with h5py.File(OUTPUT_HDF5, 'w') as h5f:
    
    # --- ROOT LEVEL METADATA ---
    info_grp = h5f.create_group("HDFEOS INFORMATION")
    info_grp.attrs["HDFEOSVersion"] = "HDFEOS_5.1.16"
    
    struct_meta = generate_struct_metadata("LANDSAT", width, height, ul_coords, lr_coords, utm_zone, 7, len(tif_files))
    dt = h5py.string_dtype(encoding='ascii')
    info_grp.create_dataset("StructMetadata.0", shape=(1,), dtype=dt, data=struct_meta)

    # --- DATA ARRAYS ---
    data_grp = h5f.create_group('/HDFEOS/GRIDS/LANDSAT/Data Fields')
    
    sr_data = stacked_data[:, 0:7, :, :] 
    qa_pixel_data = stacked_data[:, 7, :, :].astype(np.uint16)
    qa_radsat_data = stacked_data[:, 8, :, :].astype(np.uint16)
    qa_aerosol_data = stacked_data[:, 9, :, :].astype(np.uint8)
    
    sr_ds = data_grp.create_dataset('surface_reflectance', data=sr_data, compression='gzip', compression_opts=4)
    data_grp.create_dataset('QUALITY_L1_PIXEL', data=qa_pixel_data, compression='gzip', compression_opts=4)
    data_grp.create_dataset('RADIOMETRIC_SATURATION', data=qa_radsat_data, compression='gzip', compression_opts=4)
    data_grp.create_dataset('QUALITY_L2_AEROSOL', data=qa_aerosol_data, compression='gzip', compression_opts=4)

    # --- PER-FRAME METADATA ATTRIBUTES ---
    sr_ds.attrs['units'] = "Reflectance"
    sr_ds.attrs['_FillValue'] = np.nan
    sr_ds.attrs['wavelengths'] = LANDSAT_WAVELENGTHS
    sr_ds.attrs['spatial_ref'] = dst_crs_wkt
    
    gdal_transform = [dst_transform.a, dst_transform.b, dst_transform.c, dst_transform.d, dst_transform.e, dst_transform.f]
    sr_ds.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

    sr_ds.attrs['acquisition_time'] = np.array(acq_times, dtype='float64') 
    sr_ds.attrs['spacecraft_id'] = spacecraft_ids 
    sr_ds.attrs['sun_azimuth'] = np.array(sun_azimuths, dtype='float32')
    sr_ds.attrs['sun_elevation'] = np.array(sun_elevations, dtype='float32')
    sr_ds.attrs['wrs_path'] = np.array(wrs_paths, dtype='int8')
    sr_ds.attrs['wrs_row'] = np.array(wrs_rows, dtype='int8')
    sr_ds.attrs['cloud_cover'] = np.array(cloud_covers, dtype='float32')

print("\nHDF5 compilation complete. Validated HDF-EOS5 compliant metadata attached.")