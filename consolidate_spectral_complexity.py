import h5py
import numpy as np
import os
import SpecComplex as sc

# ==========================================
# --- INPUT FILES ---
# ==========================================
landsat_stack = r"C:\satelliteImagery\LANDSAT\Rochester\LANDSAT_Stack_Rochester_GEE_2015_2025_WRS16.h5"
tanager_stack = r"C:\satelliteImagery\Tanager\Rochester\Tanager_Stack_Rochester_HDFEOS.h5"

input_files = [
    ("LANDSAT", landsat_stack),
    ("TANAGER", tanager_stack)
]


# ==========================================
# --- CONFIGURATION (Globals) ---
# ==========================================
TILE_SIZE = 3          # Size of the window (NxN pixels) for volume calc
SLIDING_STRIDE = 1     # Stride for sliding window
NUM_ENDMEMBERS = 7
GRAM_TYPE = 'minEndmember'
NORM_PARAM = 'bandCount'

# --- Masking Parameters ---
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 2

# LANDSAT Specific Configuration
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'medium' #'low' 'medium' 'high'

# TANAGER Specific Configuration
TANAGER_CLOUD_MASK = True
TANAGER_UNCERTAINTY_THRESHOLD = 0.1
TANAGER_AEROSOL_THRESHOLD = 0.3

# ==========================================
# --- FUNCTIONS ---
# ==========================================

def validate_spatial_alignment(valid_inputs):
    """
    Validates that the height, width, and GeoTransform are exactly identical
    across all input files.
    """
    ref_shape = None
    ref_gt = None
    ref_file = None
    
    for sensor, fpath in valid_inputs:
        with h5py.File(fpath, 'r') as h5:
            base_path = f"/HDFEOS/GRIDS/{sensor}/Data Fields"
            if base_path not in h5:
                raise ValueError(f"Base path {base_path} not found in {fpath}")
            if "surface_reflectance" not in h5[base_path]:
                raise ValueError(f"'surface_reflectance' not found under {base_path} in {fpath}")
                
            sr_ds = h5[base_path]["surface_reflectance"]
            num_frames, num_bands, height, width = sr_ds.shape
            gt = sr_ds.attrs.get("GeoTransform")
            
            if ref_shape is None:
                ref_shape = (height, width)
                ref_gt = gt
                ref_file = fpath
            else:
                if ref_shape != (height, width):
                    raise ValueError(f"Strict Spatial Validation Failed! "
                                     f"{ref_file} has {(ref_shape)}, "
                                     f"but {fpath} has {(height, width)}.")
                                     
                if ref_gt is not None and gt is not None:
                    # using numpy allclose for floating point transformations
                    if not np.allclose(ref_gt, gt, atol=1e-8):
                        raise ValueError(f"GeoTransform mismatch between {ref_file} and {fpath}.")

def process():
    # Only try to process files that are actually present on disk
    valid_inputs = [(sensor, fpath) for sensor, fpath in input_files if os.path.exists(fpath)]
    
    if not valid_inputs:
        print("No valid input files found from the provided globals.")
        return
        
    print("Validating strict spatial alignment across inputs...")
    validate_spatial_alignment(valid_inputs)
    print("Validation passed. Dimensions and GeoTransform Match.")
    
    # Dynamically build output directory from sensors 
    sensors_processed = [sensor.title() for sensor, _ in valid_inputs]
    folder_name = "-".join(sensors_processed)
    out_dir = os.path.join(r"C:\satelliteImagery", folder_name)
    os.makedirs(out_dir, exist_ok=True)
    
    consolidated_out = os.path.join(out_dir, "Consolidated_Spectral_Complexity_Out.h5")
    
    # Process inputs chronologically and stream into a single consolidated output file
    with h5py.File(consolidated_out, 'w') as out_h5:
        
        for sensor, fpath in valid_inputs:
            print(f"\n--- Processing Sensor: {sensor} ---")
            
            with h5py.File(fpath, 'r') as in_h5:
                in_base = f"/HDFEOS/GRIDS/{sensor}/Data Fields"
                sr_ds = in_h5[in_base]["surface_reflectance"]
                num_frames, num_bands, height, width = sr_ds.shape
                
                # Fetch Tanager valid band mask early
                if sensor == "TANAGER":
                    gw_mask = sr_ds.attrs.get("all_good_wavelengths").astype(bool)

                # Use require_group to maintain the HDF-EOS5 expected path natively
                out_grp = out_h5.require_group(in_base)
                
                # Pre-initialize output memory streams for memory efficiency
                # The _FillValue is implicitly created internally by HDF5 via the fillvalue arg
                ds_endmembers = out_grp.create_dataset(
                    'frame_endmembers', shape=(num_frames, num_bands, NUM_ENDMEMBERS), 
                    dtype='float32', compression="gzip", fillvalue=np.nan)
                    
                ds_endmember_indices = out_grp.create_dataset(
                    'frame_endmember_indices', shape=(num_frames, NUM_ENDMEMBERS), 
                    dtype='int32', compression="gzip")
                    
                ds_vol_curve = out_grp.create_dataset(
                    'frame_endmember_volumes', shape=(num_frames, NUM_ENDMEMBERS), 
                    dtype='float32', compression="gzip", fillvalue=np.nan)
                    
                ds_slide = out_grp.create_dataset(
                    'sliding_volume_map', shape=(num_frames, height, width), 
                    dtype='float32', compression="gzip", fillvalue=np.nan)
                    
                ds_msd = out_grp.create_dataset(
                    'msd_map', shape=(num_frames, height, width), 
                    dtype='float32', compression="gzip", fillvalue=np.nan)
                    
                ds_slideZ_masked = out_grp.create_dataset(
                    'sliding_volume_z_score_masked', shape=(num_frames, height, width), 
                    dtype='float32', compression="gzip", fillvalue=np.nan)

                # Iterate Memory-Efficiently Frame-by-Frame
                for t in range(num_frames):
                    print(f"[{sensor}] Frame {t+1}/{num_frames}")
                    
                    frame_sr = sr_ds[t, ...]
                    
                    # Dynamically strip Tanager bad domains 
                    if sensor == "TANAGER":
                        frame_active = np.delete(frame_sr, np.where(~gw_mask[t]), axis=0)
                    else:
                        frame_active = frame_sr
                        
                    # Generating Data Validation Mask Space 
                    if sensor == "LANDSAT":
                        valid_mask = sc.get_landsat_mask(
                            in_h5[in_base], t, (height, width),
                            sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                            cloud_dilation=CLOUD_DILATION,
                            qa_reject_mask=QA_REJECT_MASK,
                            radsat_accept_value=RADSAT_ACCEPT_VALUE,
                            aerosol_accept_level=AEROSOL_ACCEPT_LEVEL
                        )
                    elif sensor == "TANAGER":
                        valid_mask = sc.get_tanager_mask(
                            in_h5[in_base], t, (height, width),
                            sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                            cloud_dilation=CLOUD_DILATION,
                            apply_cloud_mask=TANAGER_CLOUD_MASK,
                            uncertainty_threshold=TANAGER_UNCERTAINTY_THRESHOLD,
                            aerosol_depth_threshold=TANAGER_AEROSOL_THRESHOLD
                        )

                    # 1. Full Frame Volumes
                    endmembers, endmember_idx, vol_curve = sc.process_volume_frame(
                        frame_active, NUM_ENDMEMBERS, GRAM_TYPE, NORM_PARAM)
                        
                    # Restore stripped bad hyper-dimensions with explicit NaNs if TANAGER
                    if sensor == "TANAGER":
                        em_full = np.full((num_bands, NUM_ENDMEMBERS), np.nan, dtype=np.float32)
                        em_full[gw_mask[t]==1, :] = endmembers
                        ds_endmembers[t, ...] = em_full
                    else:
                        ds_endmembers[t, ...] = endmembers
                        
                    ds_endmember_indices[t, ...] = endmember_idx
                    ds_vol_curve[t, ...] = vol_curve
                    
                    # 2. Sliding Volume Tile
                    slide_map = sc.process_volume_sliding_tile(
                        frame_active, TILE_SIZE, SLIDING_STRIDE, NUM_ENDMEMBERS, GRAM_TYPE, NORM_PARAM)
                    ds_slide[t, ...] = slide_map
                    
                    # 3. Spectral Distance
                    ds_msd[t, ...] = sc.process_msd_sliding_tile(
                        frame_active, TILE_SIZE, SLIDING_STRIDE)
                    
                    # 4. Masked Z-Score 
                    z_masked = sc.calculate_global_z_score(slide_map, valid_mask)
                    ds_slideZ_masked[t, ...] = z_masked
                    
                # Bind Metadata & Execution Properties (HDFEOS5 Format)
                ds_endmembers.attrs['description'] = "Endmembers for each pixel"
                ds_endmembers.attrs['NUM_ENDMEMBERS'] = NUM_ENDMEMBERS
                ds_endmembers.attrs['NORM_PARAM'] = NORM_PARAM
                
                ds_endmember_indices.attrs['description'] = "Endmember indices for each pixel"
                
                ds_vol_curve.attrs['description'] = "Full volume curve (Volume vs Endmember Count) for entire frame"
                ds_vol_curve.attrs['GRAM_TYPE'] = GRAM_TYPE
                ds_vol_curve.attrs['NUM_ENDMEMBERS'] = NUM_ENDMEMBERS
                ds_vol_curve.attrs['NORM_PARAM'] = NORM_PARAM
                
                ds_slide.attrs['description'] = "Volume of convex hull of spectral data within each sliding NxN tile"
                ds_slide.attrs['TILE_SIZE'] = TILE_SIZE
                ds_slide.attrs['SLIDING_STRIDE'] = SLIDING_STRIDE
                ds_slide.attrs['NUM_ENDMEMBERS'] = NUM_ENDMEMBERS
                ds_slide.attrs['GRAM_TYPE'] = GRAM_TYPE
                ds_slide.attrs['NORM_PARAM'] = NORM_PARAM
                
                ds_msd.attrs['description'] = "MSD for each pixel"
                ds_msd.attrs['TILE_SIZE'] = TILE_SIZE
                ds_msd.attrs['SLIDING_STRIDE'] = SLIDING_STRIDE
                
                ds_slideZ_masked.attrs['description'] = "Global Spectral Complexity Z-score. Sensor-masked pixels excluded from background stats."
                ds_slideZ_masked.attrs['SUN_ELEVATION_THRESHOLD'] = SUN_ELEVATION_THRESHOLD
                ds_slideZ_masked.attrs['CLOUD_DILATION'] = CLOUD_DILATION
                
                if sensor == "LANDSAT":
                    ds_slideZ_masked.attrs['QA_REJECT_MASK'] = QA_REJECT_MASK
                    ds_slideZ_masked.attrs['RADSAT_ACCEPT_VALUE'] = RADSAT_ACCEPT_VALUE
                    ds_slideZ_masked.attrs['AEROSOL_ACCEPT_LEVEL'] = AEROSOL_ACCEPT_LEVEL
                elif sensor == "TANAGER":
                    ds_slideZ_masked.attrs['TANAGER_CLOUD_MASK'] = TANAGER_CLOUD_MASK
                    ds_slideZ_masked.attrs['TANAGER_UNCERTAINTY_THRESHOLD'] = TANAGER_UNCERTAINTY_THRESHOLD
                    ds_slideZ_masked.attrs['TANAGER_AEROSOL_THRESHOLD'] = TANAGER_AEROSOL_THRESHOLD

    print(f"\nProcessing Complete. Consolidated file saved to '{consolidated_out}'")

if __name__ == "__main__":
    process()
