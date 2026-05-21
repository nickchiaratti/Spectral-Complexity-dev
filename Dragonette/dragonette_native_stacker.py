import os
import json
import h5py
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# --- Configuration ---
TIME_THRESHOLD_SECONDS = 120  # Group acquisitions within 2 minutes into the same temporal pass
SOURCE_DIR = "C:/satelliteImagery/dragonette/ROCX_SourceData"
OUTPUT_DIR = SOURCE_DIR
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Wyvern_Native_Stack_ROCX.h5")

# RGB Wavelength Targets for Ortho Visual generation (matching Wyvern's preview bands)
TARGET_RED_NM = 680.0
TARGET_GREEN_NM = 534.0
TARGET_BLUE_NM = 480.0

def generate_struct_metadata(grid_name, width, height, ul_coords, lr_coords, datasets_info, n_times, n_bands):
    """
    Generates standard HDF-EOS5 Object Definition Language (ODL) metadata.
    Configured strictly for Geographic WGS84 (EPSG:4326) mapping.
    Reference: HDF-EOS5 Data Model, Volume 1 (NASA).
    """
    data_fields_blocks = []
    for i, (name, dtype, rank, dim_names) in enumerate(datasets_info):
        eos_type = "H5T_NATIVE_FLOAT"
        if "uint8" in str(dtype): eos_type = "H5T_NATIVE_UINT8"
        elif "uint16" in str(dtype): eos_type = "H5T_NATIVE_UINT16"
        elif "uint" in str(dtype): eos_type = "H5T_NATIVE_UINT"
        elif "int" in str(dtype): eos_type = "H5T_NATIVE_INT"
        elif "float64" in str(dtype) or "double" in str(dtype): eos_type = "H5T_NATIVE_DOUBLE"
        
        dims_list = ",".join([f"\"{d}\"" for d in dim_names])
        block = f"""            OBJECT=DataField_{i+1}
                DataFieldName="{name}"
                DataType={eos_type}
                DimList=({dims_list})
                MaxdimList=({dims_list})
                CompressionType=HE5_HDFE_COMP_DEFLATE
                DeflateLevel=4
            END_OBJECT=DataField_{i+1}"""
        data_fields_blocks.append(block)
    
    # Projection HE5_GCTP_GEO is required for EPSG:4326 data in HDF-EOS
    odl = f"""GROUP=SwathStructure
END_GROUP=SwathStructure
GROUP=GridStructure
    GROUP=GRID_1
        GridName="{grid_name}"
        XDim={width}
        YDim={height}
        UpperLeftPointMtrs=({ul_coords[0]:.9f},{ul_coords[1]:.9f})
        LowerRightMtrs=({lr_coords[0]:.9f},{lr_coords[1]:.9f})
        Projection=HE5_GCTP_GEO
        SphereCode=12
        CompressionType=HE5_HDFE_COMP_DEFLATE
        DeflateLevel=4
        PixelRegistration=HE5_HDFE_CORNER
        GridOrigin=HE5_HDFE_GD_UL

        GROUP=Dimension
            OBJECT=Dimension_1
                DimensionName="Time"
                Size={n_times}
            END_OBJECT=Dimension_1
            OBJECT=Dimension_2
                DimensionName="Band"
                Size={n_bands}
            END_OBJECT=Dimension_2
            OBJECT=Dimension_3
                DimensionName="YDim"
                Size={height}
            END_OBJECT=Dimension_3
            OBJECT=Dimension_4
                DimensionName="XDim"
                Size={width}
            END_OBJECT=Dimension_4
            OBJECT=Dimension_5
                DimensionName="RGBBand"
                Size=3
            END_OBJECT=Dimension_5
        END_GROUP=Dimension

        GROUP=DataField
{"\n".join(data_fields_blocks)}
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

def parse_wyvern_stac(json_path):
    """Extracts metrology and file paths from the Wyvern STAC JSON."""
    with open(json_path, 'r') as f:
        stac = json.load(f)
    
    dt_str = stac['properties']['datetime']
    acq_time = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
    
    epsg = stac['properties']['proj:epsg']
    
    # Strictly enforce Level-1B (Radiance/EPSG:4326) processing per requirements
    if stac['properties']['processing:level'] != 'L1B' or epsg != 4326:
        raise ValueError(f"CRITICAL: {json_path} is not L1B/EPSG:4326. Will not interpolate or warp incorrectly formatted data.")
    
    assets = stac['assets']
    base_dir = os.path.dirname(json_path)
    
    def resolve_path(href):
        return os.path.join(base_dir, os.path.basename(href))
    
    # Extract band metadata
    eo_bands = assets['Cloud optimized GeoTiff']['eo:bands']
    wavelengths = [b['center_wavelength'] * 1000 for b in eo_bands] # Convert µm to nm
    fwhms = [b['full_width_half_max'] * 1000 for b in eo_bands] # Convert µm to nm
    
    try:
        esun = [b['solar_illumination'] for b in eo_bands]
    except KeyError:
        raise KeyError(f"CRITICAL: Missing 'solar_illumination' in eo:bands for STAC item {stac['id']}.")

    try:
        sun_elev = stac['properties']['view:sun_elevation']
        sun_azim = stac['properties']['view:sun_azimuth']
        off_nadir = stac['properties']['view:off_nadir']
        inc_angle = stac['properties']['view:incidence_angle']
        view_azim = stac['properties']['view:azimuth']
    except KeyError as e:
        raise KeyError(f"CRITICAL: Missing essential geometric metadata {e} in STAC item {stac['id']}.")
    
    # Extract Fill Value
    nodata_val = assets['Cloud optimized GeoTiff']['raster:bands'][0].get('nodata', -9999)

    return {
        'id': stac['id'],
        'json_path': json_path,
        'time': acq_time,
        'epsg': epsg,
        'platform': stac['properties']['platform'],
        'radiance_tif': resolve_path(assets['Cloud optimized GeoTiff']['href']),
        'data_mask_tif': resolve_path(assets['Data Mask']['href']),
        'pixel_quality_tif': resolve_path(assets['Pixel Quality Mask']['href']),
        'wavelengths': np.array(wavelengths, dtype=np.float32),
        'fwhm': np.array(fwhms, dtype=np.float32),
        'solar_illumination': np.array(esun, dtype=np.float32),
        'sun_elevation': sun_elev,
        'sun_azimuth': sun_azim,
        'view_off_nadir': off_nadir,
        'view_incidence_angle': inc_angle,
        'view_azimuth': view_azim,
        'nodata': nodata_val,
        'stac_dict': stac
    }

def calculate_global_geographic_grid(scenes):
    """
    Computes a strict bounding box union aligning to the native angular resolution 
    (Lon/Lat) of the source data. Explicitly avoids affine phase shifts.
    """
    global_min_x, global_min_y = float('inf'), float('inf')
    global_max_x, global_max_y = float('-inf'), float('-inf')
    
    # Utilize the first scene to establish the strict angular grid size
    with rasterio.open(scenes[0]['radiance_tif']) as src:
        ref_transform = src.transform
        x_res = abs(ref_transform.a)
        y_res = abs(ref_transform.e)

    for scene in scenes:
        with rasterio.open(scene['radiance_tif']) as src:
            bounds = src.bounds
            global_min_x = min(global_min_x, bounds.left)
            global_min_y = min(global_min_y, bounds.bottom)
            global_max_x = max(global_max_x, bounds.right)
            global_max_y = max(global_max_y, bounds.top)

    width = int(np.ceil((global_max_x - global_min_x) / x_res))
    height = int(np.ceil((global_max_y - global_min_y) / y_res))

    target_transform = Affine.translation(global_min_x, global_max_y) * Affine.scale(x_res, -y_res)
    
    return target_transform, width, height, (global_min_x, global_max_y), (global_max_x, global_min_y)

def process_native_stack():
    print("Discovering Wyvern STAC collections...")
    
    # 1. Recursive Data Discovery
    root_path = Path(SOURCE_DIR)
    json_files = list(root_path.rglob("*.json"))
    json_files = [f for f in json_files if "catalog" not in f.name.lower()]
    
    # Filter out top-level downloaded JSONs by ensuring the json is adjacent to TIFF data
    valid_json_files = [f for f in json_files if any(f.parent.glob("*.tiff")) or any(f.parent.glob("*.tif"))]
    
    if not valid_json_files:
        raise FileNotFoundError(f"CRITICAL: No valid Wyvern STAC JSON files with adjacent TIFFs found in {SOURCE_DIR}")

    raw_scenes = []
    for j_path in valid_json_files:
        try:
            scene_data = parse_wyvern_stac(str(j_path))
            # Validate required TIFF files exist
            if not all(os.path.exists(scene_data[k]) for k in ['radiance_tif', 'data_mask_tif', 'pixel_quality_tif']):
                raise FileNotFoundError(f"Missing required TIFF assets for item {scene_data['id']}.")
            raw_scenes.append(scene_data)
        except Exception as e:
            # Doctoral Context Constraint: We fail-fast instead of skipping corrupt data.
            raise RuntimeError(f"Data integrity error during discovery of {j_path.name}: {str(e)}")

    raw_scenes.sort(key=lambda x: x['time'])

    # 2. Temporal Grouping
    grouped_scenes = []
    current_group = [raw_scenes[0]]
    for i in range(1, len(raw_scenes)):
        delta = (raw_scenes[i]['time'] - current_group[-1]['time']).total_seconds()
        if delta <= TIME_THRESHOLD_SECONDS:
            current_group.append(raw_scenes[i])
        else:
            grouped_scenes.append(current_group)
            current_group = [raw_scenes[i]]
    grouped_scenes.append(current_group)

    print(f"Aggregated {len(raw_scenes)} scenes into {len(grouped_scenes)} temporal passes.")

    # 3. Establish Universal Grid & Spectral Alignment
    tf_target, width, height, ul_coords, lr_coords = calculate_global_geographic_grid(raw_scenes)
    print(f"Global Geographic Grid: {width}x{height} angular pixels")

    # Enforce Spectral Integrity: All scenes must share identical wavelength arrays.
    base_wv = raw_scenes[0]['wavelengths']
    for scn in raw_scenes:
        if len(scn['wavelengths']) != len(base_wv) or not np.allclose(scn['wavelengths'], base_wv, atol=1.0):
            raise ValueError(f"CRITICAL SPECTRAL MISMATCH: Dataset {scn['id']} wavelength array does not match stack standard. Cannot construct pure 4D tensor.")

    n_times = len(grouped_scenes)
    n_bands = len(base_wv)

    # 4. HDF5 Tensor Construction
    with h5py.File(OUTPUT_FILE, 'w') as out_h5:
        # Establish structural hierarchy
        grp_wyvern = out_h5.create_group("HDFEOS/GRIDS/WYVERN/Data Fields")
        meta_grp = out_h5.create_group("METADATA")
        info_grp = out_h5.create_group("HDFEOS INFORMATION")
        
        # Dataset Registration (Shape/Type/Rank/Dimensions)
        datasets_info = [
            ("radiance", np.dtype('float32'), 4, ["Time", "Band", "YDim", "XDim"]),
            ("data_mask", np.dtype('uint8'), 4, ["Time", "Band", "YDim", "XDim"]), # Mask has 4 feature bands
            ("pixel_quality_mask", np.dtype('uint8'), 4, ["Time", "Band", "YDim", "XDim"]), # Quality mask matches radiance bands
            ("ortho_visual", np.dtype('float32'), 4, ["Time", "RGBBand", "YDim", "XDim"]),
            ("wavelength", np.dtype('float32'), 2, ["Time", "Band"]),
            ("fwhm", np.dtype('float32'), 2, ["Time", "Band"]),
            ("solar_illumination", np.dtype('float32'), 2, ["Time", "Band"])
        ]
        
        # Allocate Datasets in BSQ Layout
        rad_nodata = raw_scenes[0]['nodata']
        ds_rad = grp_wyvern.create_dataset("radiance", shape=(n_times, n_bands, height, width), dtype='float32', compression="gzip", fillvalue=rad_nodata)
        ds_dmask = grp_wyvern.create_dataset("data_mask", shape=(n_times, 4, height, width), dtype='uint8', compression="gzip", fillvalue=255)
        ds_qmask = grp_wyvern.create_dataset("pixel_quality_mask", shape=(n_times, n_bands, height, width), dtype='uint8', compression="gzip", fillvalue=255)
        ds_vis = grp_wyvern.create_dataset("ortho_visual", shape=(n_times, 3, height, width), dtype='float32', compression="gzip", fillvalue=rad_nodata)
        ds_wv = grp_wyvern.create_dataset("wavelength", shape=(n_times, n_bands), dtype='float32', compression="gzip")
        ds_fwhm = grp_wyvern.create_dataset("fwhm", shape=(n_times, n_bands), dtype='float32', compression="gzip")
        ds_esun = grp_wyvern.create_dataset("solar_illumination", shape=(n_times, n_bands), dtype='float32', compression="gzip")
        
        acq_time_array = np.zeros(n_times, dtype='float64')
        platform_array = []
        sun_elev_array = np.zeros(n_times, dtype='float64')
        sun_azim_array = np.zeros(n_times, dtype='float64')
        off_nadir_array = np.zeros(n_times, dtype='float64')
        inc_angle_array = np.zeros(n_times, dtype='float64')
        view_azim_array = np.zeros(n_times, dtype='float64')

        gdal_transform = [tf_target.c, tf_target.a, tf_target.b, tf_target.f, tf_target.d, tf_target.e]
        crs_wkt = "EPSG:4326"

        for t_idx, group in enumerate(grouped_scenes):
            print(f"  Processing Temporal Pass {t_idx+1}/{n_times} ({group[0]['time'].isoformat()})...")
            
            # Temporary canvases for spatial union
            canvas_rad = np.full((n_bands, height, width), rad_nodata, dtype='float32')
            canvas_dmask = np.full((4, height, width), 255, dtype='uint8')
            canvas_qmask = np.full((n_bands, height, width), 255, dtype='uint8')

            # Spatial Mosaic via Nearest Neighbor Reprojection
            for scene in group:
                with rasterio.open(scene['radiance_tif']) as src:
                    # Mosaicking constraint: Nearest neighbor prevents synthesis of false spectral signatures
                    reproject(
                        source=rasterio.band(src, list(range(1, src.count + 1))),
                        destination=canvas_rad,
                        src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target, dst_crs=crs_wkt,
                        resampling=Resampling.nearest,
                        src_nodata=rad_nodata, dst_nodata=rad_nodata
                    )
                
                with rasterio.open(scene['data_mask_tif']) as src:
                    reproject(
                        source=rasterio.band(src, list(range(1, src.count + 1))),
                        destination=canvas_dmask,
                        src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target, dst_crs=crs_wkt,
                        resampling=Resampling.nearest,
                        src_nodata=255, dst_nodata=255
                    )

                with rasterio.open(scene['pixel_quality_tif']) as src:
                    reproject(
                        source=rasterio.band(src, list(range(1, src.count + 1))),
                        destination=canvas_qmask,
                        src_transform=src.transform, src_crs=src.crs, dst_transform=tf_target, dst_crs=crs_wkt,
                        resampling=Resampling.nearest,
                        src_nodata=255, dst_nodata=255
                    )
            
            # Temporal Frame Validation Rule
            if np.all(canvas_rad == rad_nodata):
                raise ValueError(f"CRITICAL: Data assimilation failed for temporal pass {t_idx}. Spatial frame is completely NoData. Aborting to preserve tensor integrity.")

            # Write constructed frame to HDF5 Tensor
            ds_rad[t_idx, ...] = canvas_rad
            ds_dmask[t_idx, ...] = canvas_dmask
            ds_qmask[t_idx, ...] = canvas_qmask
            
            # Store metrology
            acq_time_array[t_idx] = group[0]['time'].timestamp()
            platform_array.append(group[0]['platform'])
            sun_elev_array[t_idx] = group[0]['sun_elevation']
            sun_azim_array[t_idx] = group[0]['sun_azimuth']
            off_nadir_array[t_idx] = group[0]['view_off_nadir']
            inc_angle_array[t_idx] = group[0]['view_incidence_angle']
            view_azim_array[t_idx] = group[0]['view_azimuth']
            
            # Store 2D Spectral/Illumination properties explicitly mapping Time -> Band
            ds_wv[t_idx, :] = group[0]['wavelengths']
            ds_fwhm[t_idx, :] = group[0]['fwhm']
            ds_esun[t_idx, :] = group[0]['solar_illumination']

            meta_grp.attrs[f"frame_{t_idx}_json"] = json.dumps(group[0]['stac_dict'])

        # Generate True Color 'ortho_visual' analytically matching Tanager script behavior
        print("  Extracting 'ortho_visual' specific bands...")
        r_idx = int(np.argmin(np.abs(base_wv - TARGET_RED_NM)))
        g_idx = int(np.argmin(np.abs(base_wv - TARGET_GREEN_NM)))
        b_idx = int(np.argmin(np.abs(base_wv - TARGET_BLUE_NM)))
        
        for t_idx in tqdm(range(n_times), desc="  Writing visual channels"):
            ds_vis[t_idx, 0, :, :] = ds_rad[t_idx, r_idx, :, :]
            ds_vis[t_idx, 1, :, :] = ds_rad[t_idx, g_idx, :, :]
            ds_vis[t_idx, 2, :, :] = ds_rad[t_idx, b_idx, :, :]

        # Metrology Annotation (CF & HDF-EOS Compliance)
        platform_utf8 = np.array(platform_array, dtype='S20')
        
        for dataset in [ds_rad, ds_dmask, ds_qmask, ds_vis]:
            dataset.attrs['spatial_ref'] = crs_wkt
            dataset.attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

        # Add physical units to the explicit 2D spectral datasets
        ds_wv.attrs['unit'] = 'nm'
        ds_fwhm.attrs['unit'] = 'nm'
        ds_esun.attrs['unit'] = 'W*m-2*um-1'

        ds_rad.attrs["acquisition_time"] = acq_time_array
        ds_rad.attrs["sun_elevation"] = sun_elev_array
        ds_rad.attrs["sun_azimuth"] = sun_azim_array
        ds_rad.attrs["view_off_nadir"] = off_nadir_array
        ds_rad.attrs["view_incidence_angle"] = inc_angle_array
        ds_rad.attrs["view_azimuth"] = view_azim_array
        ds_rad.attrs["spacecraft_id"] = platform_utf8
        ds_rad.attrs["unit"] = "W*sr-1*m-2*um-1"

        ds_dmask.attrs["description"] = "Band 1: Clear, Band 2: Cloud, Band 3: Haze, Band 4: Cloud-Shadow"

        # Finalize HDF-EOS StructMetadata definition
        struct_meta = generate_struct_metadata("WYVERN", width, height, ul_coords, lr_coords, datasets_info, n_times, n_bands)
        dt_str = h5py.string_dtype(encoding='ascii')
        info_grp.create_dataset("StructMetadata.0", (1,), dtype=dt_str, data=struct_meta)

    print(f"\nTensor Synthesis Complete. Stored analytically rigorous payload at: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_native_stack()