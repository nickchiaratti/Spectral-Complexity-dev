import ee
import geemap
import h5py
import rasterio
import numpy as np
import os
import glob
from datetime import datetime, timezone
import shutil
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import Transformer, CRS
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as scipy_shift
import SpecComplex as sc

# ==========================================
# 1. CONFIGURATION
# ==========================================
cloud_threshold = 60
try:
    ee.Initialize(project="project-ee18dbee-cd7e-4d08-812")
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

Location = "Rochester"
# Set SOURCE_CACHE to a larger parent region (e.g., "Rochester") to bypass GEE downloads
# and perform fast, local spatial subsetting from existing parent .tif files.
# Set to None to use the dedicated Location folder.
SOURCE_CACHE = "Rochester" 

if Location == "Rochester":
    # Rochester Bounding Box
    ROI_LON_MIN = -77.72; ROI_LON_MAX = -77.4450
    ROI_LAT_MIN = 43.0450; ROI_LAT_MAX = 43.28
    START_DATE = '2015-01-01'
    END_DATE = '2025-12-31'
elif Location == "Tait":
    # Tait Bounding Box
    ROI_LON_MIN = -77.516127; ROI_LON_MAX = -77.461968
    ROI_LAT_MIN = 43.127698; ROI_LAT_MAX = 43.159168
    START_DATE = '2015-01-01'
    END_DATE = '2025-12-31' 

if SOURCE_CACHE:
    TEMP_DIR = f"C:/satelliteImagery/LANDSAT/SourceData/{SOURCE_CACHE}_TEMP_GEE_DOWNLOAD"
else:
    TEMP_DIR = f"C:/satelliteImagery/LANDSAT/SourceData/{Location}_TEMP_GEE_DOWNLOAD"

# Reconstruct GEE Geometry for searching the catalog
ROI = ee.Geometry.Rectangle([ROI_LON_MIN, ROI_LAT_MIN, ROI_LON_MAX, ROI_LAT_MAX])

# Target Resolution (Matches Tanager)
TARGET_RESOLUTION = 30.0

# --- Artifact Correction & Alignment ---
# Enforce a single orbital path to prevent 1-pixel stereoscopic parallax shifting.
TARGET_WRS_PATH = None#16  # Set to None to include all paths

# Attempt sub-pixel phase correlation registration against a Master Anchor.
AUTO_CO_REGISTER = True 
# Strict failure threshold: If calculated shift exceeds this many pixels, the script will crash.
MAX_ALLOWED_SHIFT = 1.0 

suffix = ""

if TARGET_WRS_PATH is not None:
    suffix += f"_WRS{TARGET_WRS_PATH}"

if AUTO_CO_REGISTER:
    suffix += "_CoReg"

OUTPUT_HDF5 = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025{suffix}.h5"

# Landsat 8/9 SR Bands to extract, plus all required QA bands
BANDS = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT', 'SR_QA_AEROSOL']
LANDSAT_WAVELENGTHS = [0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201]

# ==========================================
# 2. COMMON GEOGRAPHIC FRAMEWORK (MASTER GRID)
# ==========================================
def calculate_target_grid(lon_min, lon_max, lat_min, lat_max, resolution):
    """
    Calculates the exact Euclidean target grid, perfectly mirroring the local 
    Tanager processing logic to establish a Common Geographic Framework.
    """
    central_lon = (lon_min + lon_max) / 2
    utm_zone = int((central_lon + 180) / 6) + 1
    epsg_code = 32600 + utm_zone
    dst_crs = CRS.from_epsg(epsg_code)
    
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs, ys = transformer.transform([lon_min, lon_max, lon_max, lon_min], 
                                   [lat_max, lat_max, lat_min, lat_min])
    
    dst_min_x, dst_max_x = min(xs), max(xs)
    dst_min_y, dst_max_y = min(ys), max(ys)
    
    width = int(np.ceil((dst_max_x - dst_min_x) / resolution))
    height = int(np.ceil((dst_max_y - dst_min_y) / resolution))
    
    target_transform = from_bounds(dst_min_x, dst_min_y, dst_max_x, dst_max_y, width, height)
    
    return dst_crs, target_transform, width, height, (dst_min_x, dst_max_y), (dst_max_x, dst_min_y), utm_zone

# Generate the Master Grid parameters before querying GEE
print("Establishing Common Geographic Framework (Master Grid)...")
dst_crs, tf_target, grid_width, grid_height, ul_coords, lr_coords, utm_zone = calculate_target_grid(
    ROI_LON_MIN, ROI_LON_MAX, ROI_LAT_MIN, ROI_LAT_MAX, TARGET_RESOLUTION
)

TARGET_CRS_STR = f"EPSG:{32600 + utm_zone}"

# Create a strictly projected Geometry bounding box for our baseline
exact_bounds_ee = ee.Geometry.Rectangle(
    coords=[ul_coords[0], lr_coords[1], lr_coords[0], ul_coords[1]], 
    proj=TARGET_CRS_STR, 
    geodesic=False
)

# Apply a 500-meter buffer to the boundary. 
# This prevents interpolation ringing artifacts at the edges during local reprojection.
buffered_bounds_ee = exact_bounds_ee.buffer(500).bounds()

print(f"Master Grid dimensions locked to {grid_width}x{grid_height} at {TARGET_CRS_STR}")

# ==========================================
# 3. HDF-EOS5 STRUCTURAL UTILITY
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
            OBJECT=Dimension_5
                DimensionName="VisBand"
                Size=4
            END_OBJECT=Dimension_5
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
            OBJECT=DataField_3
                DataFieldName="ortho_visual"
                DataType=HDF5T_NATIVE_UINT8
                DimList=("Time","VisBand","YDim","XDim")
            END_OBJECT=DataField_3
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
# 4. GEE SERVER-SIDE PROCESSING
# ==========================================
def prepare_export_types(image):
    """
    Enforces strict native integer data types to minimize GEE payload size.
    Radiometric scaling (float conversion) is deferred to local client-side processing.
    """
    # Keep surface reflectance as native uint16 to halve download size
    optical_bands = image.select('SR_B.').toUint16()
    
    # Enforce minimum viable integer footprints for QA bitmasks
    qa_bands = image.select(['QA_PIXEL', 'QA_RADSAT']).toUint16()
    qa_aerosol = image.select('SR_QA_AEROSOL').toUint8()
    
    # Replace bands in the image
    img = image.addBands(optical_bands, None, True)
    img = img.addBands(qa_bands, None, True)
    img = img.addBands(qa_aerosol, None, True)
    
    return img

print("Querying Google Earth Engine for Landsat 8/9 Collection 2 SR...")
collection_l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
collection_l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
merged_collection = collection_l8.merge(collection_l9)

# Fast metadata-based filtering (Coarse Filter)
filtered_collection = merged_collection \
    .filterBounds(ROI) \
    .filterDate(START_DATE, END_DATE) \
    .filter(ee.Filter.lt('CLOUD_COVER', cloud_threshold))

# Apply Orbital Path Filter to resolve stereoscopic parallax misregistration
if TARGET_WRS_PATH is not None:
    print(f"Applying strict orbital filter: WRS_PATH == {TARGET_WRS_PATH}")
    filtered_collection = filtered_collection.filter(ee.Filter.eq('WRS_PATH', TARGET_WRS_PATH))

filtered_collection = filtered_collection.map(prepare_export_types) \
    .select(BANDS) \
    .sort('system:time_start')

num_images = filtered_collection.size().getInfo()
print(f"Found {num_images} images matching criteria. Fetching metadata dictionary...")

info = filtered_collection.getInfo()
gee_metadata = {}
for feature in info['features']:
    img_id = feature['id'].split('/')[-1] 
    props = feature['properties']
    time_start_ms = props.get('system:time_start')
    unix_time_sec = time_start_ms / 1000.0
    
    gee_metadata[img_id] = {
        'acquisition_time': unix_time_sec, # Stored as float for unix timestamp
        'spacecraft_id': props.get('SPACECRAFT_ID'),
        'sun_azimuth': props.get('SUN_AZIMUTH'),
        'sun_elevation': props.get('SUN_ELEVATION'),
        'wrs_path': props.get('WRS_PATH'),
        'wrs_row': props.get('WRS_ROW'),
        'cloud_cover': props.get('CLOUD_COVER')
    }

# ==========================================
# 5. DOWNLOAD & TRANSLATE TO HDF5
# ==========================================
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

print(f"Downloading GeoTIFFs (Buffered region for local reprojection)...")

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
        processed_image = prepare_export_types(image).select(BANDS)
        
        # Download loose, buffered tiles via standard scale, preventing server compute charges
        geemap.ee_export_image(
            processed_image, 
            filename=out_tif, 
            region=buffered_bounds_ee, 
            scale=TARGET_RESOLUTION,
            crs=TARGET_CRS_STR
        )
    except Exception as e:
        print(f"Failed to download {short_name}: {e}")

print(f"\nTranslating and Harmonizing downloaded GeoTIFFs to HDF5: {OUTPUT_HDF5}")

# Manifest-Driven File Resolution
tif_files = []
missing_files = 0
sorted_img_ids = sorted(gee_metadata.keys(), key=lambda k: gee_metadata[k]['acquisition_time'])

for img_id in sorted_img_ids:
    expected_path = os.path.join(TEMP_DIR, f"{img_id}.tif")
    if os.path.exists(expected_path) and os.path.getsize(expected_path) > 0:
        tif_files.append(expected_path)
    else:
        missing_files += 1
        print(f"WARNING: Expected file {expected_path} is missing or corrupted.")

if len(tif_files) == 0:
    # Fail fast if data ingestion is broken
    raise RuntimeError("CRITICAL ERROR: No valid files found matching the current GEE query.")

print(f"Successfully resolved and chronologically sorted {len(tif_files)} files.")
if missing_files > 0:
    print(f"({missing_files} files failed to download properly).")

num_frames = len(tif_files)
num_bands = len(BANDS)

# Master Data Arrays
stacked_data = np.zeros((num_frames, num_bands, grid_height, grid_width), dtype=np.float32)
vis_data = np.zeros((num_frames, 4, grid_height, grid_width), dtype=np.uint8)

acq_times, spacecraft_ids, sun_azimuths = [], [], []
sun_elevations, wrs_paths, wrs_rows, cloud_covers = [], [], [] , []

# Variables for Image Co-Registration
anchor_nir = None
anchor_mask = None
valid_frame_idx = 0

for idx, tif_path in enumerate(tif_files):
    filename = os.path.basename(tif_path).replace('.tif', '')

    # STRICT METADATA VALIDATION
    meta = gee_metadata.get(filename)
    if meta is None:
        raise ValueError(f"CRITICAL ERROR: Metadata for downloaded file '{filename}' not found in GEE response. Halting to preserve dataset integrity.")
    
    for key, value in meta.items():
        if value is None:
             raise ValueError(f"CRITICAL ERROR: Metadata attribute '{key}' for '{filename}' is missing/null from GEE. Halting pipeline.")

    # --- Local Client-Side Reprojection ---
    with rasterio.open(tif_path) as src:
        # Standard fill value for native raw Landsat SR is 0
        nodata_val = src.nodata if src.nodata is not None else 0
        
        # Temporary arrays constrained to Master Grid
        temp_sr_raw = np.zeros((7, grid_height, grid_width), dtype=np.uint16)
        temp_sr_scaled = np.full((7, grid_height, grid_width), np.nan, dtype=np.float32)
        temp_qa = np.zeros((3, grid_height, grid_width), dtype=np.float32)
        
        # 1. Reproject Continuous SR Bands (Cubic Spline on raw DNs to preserve gradients)
        reproject(
            source=rasterio.band(src, list(range(1, 8))), # Bands 1-7 in rasterio
            destination=temp_sr_raw,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tf_target,
            dst_crs=dst_crs,
            resampling=Resampling.cubic_spline,
            src_nodata=nodata_val,
            dst_nodata=0
        )
        
        # --- 1b. Local Radiometric Scaling ---
        valid_sr_mask = (temp_sr_raw != 0)
        if np.any(valid_sr_mask):
            # Scale factor: value * 0.0000275 - 0.2
            scaled_sr = (temp_sr_raw[valid_sr_mask].astype(np.float32) * 0.0000275) - 0.2
            temp_sr_scaled[valid_sr_mask] = np.clip(scaled_sr, 0.0, 1.0)
        
        # 2. Reproject Categorical QA Bands (Nearest Neighbor to preserve binary bitmasks)
        reproject(
            source=rasterio.band(src, [8, 9, 10]),
            destination=temp_qa,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=tf_target,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
            src_nodata=nodata_val,
            dst_nodata=np.nan
        )
        
        # --- 1c. Optional Phase Correlation Co-Registration ---
        if AUTO_CO_REGISTER:
            # Reconstruct the valid data mask (Reject clouds, shadow, snow, fill)
            # QA_REJECT_MASK = 0b111111 (Bits 0-5)
            qa_pixel_int = temp_qa[0].astype(np.uint16)
            structural_valid_mask = (qa_pixel_int & 0b111111) == 0
            
            # Utilize Band 5 (NIR) for structural registration as it cuts through atmospheric haze best
            current_nir = temp_sr_scaled[4, :, :]
            
            if anchor_nir is None:
                # Require the anchor to be at least 90% structurally clear
                if np.sum(structural_valid_mask) / structural_valid_mask.size >= 0.90:
                    anchor_nir = current_nir.copy()
                    anchor_mask = structural_valid_mask.copy()
                    print(f"  -> Frame {filename} initialized as Phase Correlation Master Anchor.")
            else:
                # Only attempt registration if the moving image has enough clear data to match features (>15%)
                if np.sum(structural_valid_mask) / structural_valid_mask.size >= 0.15:
                    shift, error, diffphase = phase_cross_correlation(
                        anchor_nir, current_nir, 
                        reference_mask=anchor_mask, moving_mask=structural_valid_mask
                    )
                    
                    # Strict Failure Handling: Prevent bad warps from polluting the dataset
                    if abs(shift[0]) > MAX_ALLOWED_SHIFT or abs(shift[1]) > MAX_ALLOWED_SHIFT:
                        print(f"WARNING: Calculated phase shift (dy={shift[0]:.2f}, dx={shift[1]:.2f}) for {filename} exceeds MAX_ALLOWED_SHIFT. Excluding frame from stack.")
                        continue # Skip this frame entirely
                    
                    if shift[0] != 0.0 or shift[1] != 0.0:
                        # Apply Cubic Spline shift to continuous variables
                        for b in range(7):
                            temp_sr_scaled[b] = scipy_shift(temp_sr_scaled[b], shift, cval=np.nan, order=3)
                        # Apply Nearest Neighbor shift to categorical variables (QA bands)
                        for b in range(3):
                            temp_qa[b] = scipy_shift(temp_qa[b], shift, cval=1 if b==0 else 0, order=0)

        # If the frame passes co-registration (or if it's skipped), append metadata and slot the data
        acq_times.append(meta['acquisition_time'])
        spacecraft_ids.append(meta['spacecraft_id'])
        sun_azimuths.append(meta['sun_azimuth'])
        sun_elevations.append(meta['sun_elevation'])
        wrs_paths.append(meta['wrs_path'])
        wrs_rows.append(meta['wrs_row'])
        cloud_covers.append(meta['cloud_cover'])

        # Slot the perfectly aligned, scaled, edge-corrected data into the master stack
        stacked_data[valid_frame_idx, 0:7, :, :] = temp_sr_scaled
        stacked_data[valid_frame_idx, 7:10, :, :] = temp_qa
        valid_frame_idx += 1

# Trim the master arrays to remove any excluded frames
stacked_data = stacked_data[:valid_frame_idx]
vis_data = vis_data[:valid_frame_idx]

# Slice the master stack into respective HDF5 dataset categories
sr_data = stacked_data[:, 0:7, :, :] 

# Defensively convert edge-padding NaNs to compliant integer bitmask fallbacks before casting
qa_pixel_data = np.nan_to_num(stacked_data[:, 7, :, :], nan=1).astype(np.uint16)  # 1 = Fill Bit Active
qa_radsat_data = np.nan_to_num(stacked_data[:, 8, :, :], nan=0).astype(np.uint16)
qa_aerosol_data = np.nan_to_num(stacked_data[:, 9, :, :], nan=0).astype(np.uint8)

print("Generating ortho_visual RGBA representations...")
for t in range(valid_frame_idx):
    rgba_img = sc.generate_rgba_image(sr_data[t, ...])
    vis_data[t, ...] = np.transpose(rgba_img, (2, 0, 1))

# Ensure output directory exists before saving
os.makedirs(os.path.dirname(OUTPUT_HDF5), exist_ok=True)

with h5py.File(OUTPUT_HDF5, 'w') as h5f:
    
    # --- ROOT LEVEL METADATA ---
    info_grp = h5f.create_group("HDFEOS INFORMATION")
    info_grp.attrs["HDFEOSVersion"] = "HDFEOS_5.1.16"
    
    struct_meta = generate_struct_metadata("LANDSAT", grid_width, grid_height, ul_coords, lr_coords, utm_zone, num_bands, valid_frame_idx)
    dt = h5py.string_dtype(encoding='ascii')
    info_grp.create_dataset("StructMetadata.0", shape=(1,), dtype=dt, data=struct_meta)

    # --- DATA ARRAYS ---
    data_grp = h5f.create_group('/HDFEOS/GRIDS/LANDSAT/Data Fields')
    
    sr_ds = data_grp.create_dataset('surface_reflectance', data=sr_data, compression='gzip', compression_opts=6)
    data_grp.create_dataset('QUALITY_L1_PIXEL', data=qa_pixel_data, compression='gzip', compression_opts=6)
    data_grp.create_dataset('RADIOMETRIC_SATURATION', data=qa_radsat_data, compression='gzip', compression_opts=6)
    data_grp.create_dataset('QUALITY_L2_AEROSOL', data=qa_aerosol_data, compression='gzip', compression_opts=6)
    ds_ortho_visual = data_grp.create_dataset('ortho_visual', data=vis_data, dtype='uint8', compression='gzip', compression_opts=6)
    
    # --- PER-FRAME METADATA ATTRIBUTES ---
    sr_ds.attrs['units'] = "Reflectance"
    sr_ds.attrs['_FillValue'] = np.nan
    sr_ds.attrs['wavelengths'] = LANDSAT_WAVELENGTHS
    sr_ds.attrs['spatial_ref'] = dst_crs.to_wkt()
    
    # --- EXPLICIT GEOTRANSFORM ---
    # Affine natively stores: a(Pixel Width), b(Skew X), c(Origin X), d(Skew Y), e(Pixel Height), f(Origin Y)
    # GDAL standard requires: c(Origin X), a(Pixel Width), b(Skew X), f(Origin Y), d(Skew Y), e(Pixel Height)
    gdal_transform = [tf_target.c, tf_target.a, tf_target.b, tf_target.f, tf_target.d, tf_target.e]
    sr_ds.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')
    
    # Attach standardized spatial metadata to the visual layer as well for GIS compatibility
    ds_ortho_visual.attrs['spatial_ref'] = dst_crs.to_wkt()
    ds_ortho_visual.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

    sr_ds.attrs['acquisition_time'] = np.array(acq_times, dtype='float64') 
    sr_ds.attrs['spacecraft_id'] = spacecraft_ids 
    sr_ds.attrs['sun_azimuth'] = np.array(sun_azimuths, dtype='float32')
    sr_ds.attrs['sun_elevation'] = np.array(sun_elevations, dtype='float32')
    sr_ds.attrs['wrs_path'] = np.array(wrs_paths, dtype='int8')
    sr_ds.attrs['wrs_row'] = np.array(wrs_rows, dtype='int8')
    sr_ds.attrs['cloud_cover'] = np.array(cloud_covers, dtype='float32')

print("\nHDF5 compilation complete.")