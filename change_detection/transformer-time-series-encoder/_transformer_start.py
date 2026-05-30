import h5py
import numpy as np
from datetime import datetime, timezone
import rasterio.transform
from pyproj import Transformer, CRS
import SpecComplex as sc
from transformer_pipeline import run_transformer_pipeline, DirectMultiHorizonTransformer
import torch


# ==========================================
# 1. CONFIGURATION
# ==========================================
Location = "Tait"

landsat_path = f"C:/satelliteImagery/LANDSAT/{Location}/LANDSAT_Stack_{Location}_GEE_2015_2025_WRS16_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"
tanager_path = f"C:/satelliteImagery/Tanager/{Location}/Tanager_Stack_{Location}_HDFEOS_SC_EM-7_Gram-minEndmember_Norm-bandCount.h5"

# ARD Mask Configuration Used for mask generation
SUN_ELEVATION_THRESHOLD = 30
CLOUD_DILATION = 2
TANAGER_AEROSOL_DEPTH_THRESHOLD = 0.35
TANAGER_SR_UNCERTAINTY_THRESHOLD = 0.10
QA_REJECT_MASK = 0b111111
RADSAT_ACCEPT_VALUE = 0
AEROSOL_ACCEPT_LEVEL = 'medium'

# ==========================================
# 2. DATA EXTRACTION FRAMEWORK
# ==========================================

def load_merged_datacube(path_l, path_t):
    """
    Ingests Landsat and Tanager HDF5 stacks, calculates spatial QA masks, 
    and returns a strictly temporally-sorted list of dictionary records.
    """
    datacube = []
    
    h5_l = h5py.File(path_l, 'r')
    h5_t = h5py.File(path_t, 'r')
    
    # Process Landsat
    print("Ingesting LANDSAT Stack...")
    grp_l = h5_l['/HDFEOS/GRIDS/LANDSAT/Data Fields']
    sr_l = grp_l['surface_reflectance']
    times_l = sr_l.attrs['acquisition_time']
    spacecraft_l = sr_l.attrs['spacecraft_id']
    geo_tf_l = sr_l.attrs['GeoTransform']
    crs_wkt_l = sr_l.attrs['spatial_ref']
    if isinstance(crs_wkt_l, bytes): crs_wkt_l = crs_wkt_l.decode('utf-8')
    
    for i in range(len(times_l)):
        # 1. Extract Metadata
        dt = datetime.fromtimestamp(times_l[i], tz=timezone.utc)
        sensor_id = spacecraft_l[i].decode('ascii') if isinstance(spacecraft_l[i], bytes) else spacecraft_l[i]
        
        # 2. Extract and format Ortho Visual
        raw_vis = grp_l['ortho_visual'][i, ...]
        if raw_vis.shape[0] in [3, 4]:
            bip_vis = np.transpose(raw_vis, (1, 2, 0))
        else:
            bip_vis = raw_vis
            
        rgba = bip_vis.astype(np.float32) / 255.0
        if rgba.shape[-1] == 4:
            rgba[..., 3] = np.where(bip_vis[..., 3] > 0, 1.0, 0.0)
            
        # 3. Extract Z-Score Complexity
        z_score = grp_l['sliding_volume_z_score_masked'][i, ...].copy()
        
        # 4. Generate QA Mask (DO NOT APPLY to the z_score array)
        qa_mask = sc.get_landsat_mask(
            data_grp=grp_l, f_idx=i, shape=z_score.shape, 
            sun_elevation_threshold=SUN_ELEVATION_THRESHOLD, 
            cloud_dilation=CLOUD_DILATION, qa_reject_mask=QA_REJECT_MASK, 
            radsat_accept_value=RADSAT_ACCEPT_VALUE, aerosol_accept_level=AEROSOL_ACCEPT_LEVEL
        )
        
        datacube.append({
            'source': 'LANDSAT',
            'spacecraft': sensor_id,
            'timestamp': times_l[i],
            'datetime_utc': dt,
            'original_index': i,
            'crs_wkt': crs_wkt_l,
            'geo_transform': geo_tf_l,
            'ortho_visual': rgba,
            'sliding_volume_z_score_masked': z_score,
            'qa_mask': qa_mask
        })

    # Process Tanager
    print("Ingesting TANAGER Stack...")
    grp_t = h5_t['/HDFEOS/GRIDS/TANAGER/Data Fields']
    sr_t = grp_t['surface_reflectance']
    times_t = sr_t.attrs['acquisition_time']
    spacecraft_t = sr_t.attrs['spacecraft_id']
    geo_tf_t = sr_t.attrs['GeoTransform']
    crs_wkt_t = sr_t.attrs['spatial_ref']
    if isinstance(crs_wkt_t, bytes): crs_wkt_t = crs_wkt_t.decode('utf-8')
    
    for i in range(len(times_t)):
        dt = datetime.fromtimestamp(times_t[i], tz=timezone.utc)
        sensor_id = spacecraft_t[i].decode('ascii') if isinstance(spacecraft_t[i], bytes) else spacecraft_t[i]
        
        raw_vis = grp_t['ortho_visual'][i, ...]
        if raw_vis.shape[0] in [3, 4]:
            bip_vis = np.transpose(raw_vis, (1, 2, 0))
        else:
            bip_vis = raw_vis
            
        rgba = bip_vis.astype(np.float32) / 255.0
        if rgba.shape[-1] == 4:
            rgba[..., 3] = np.where(bip_vis[..., 3] > 0, 1.0, 0.0)
            
        z_score = grp_t['sliding_volume_z_score_masked'][i, ...].copy()
        
        qa_mask = sc.get_tanager_mask(
            data_grp=grp_t, f_idx=i, shape=z_score.shape, 
            sun_elevation_threshold=SUN_ELEVATION_THRESHOLD, 
            cloud_dilation=CLOUD_DILATION, apply_cloud_mask=True, 
            uncertainty_threshold=TANAGER_SR_UNCERTAINTY_THRESHOLD, 
            aerosol_depth_threshold=TANAGER_AEROSOL_DEPTH_THRESHOLD
        )
        
        datacube.append({
            'source': 'TANAGER',
            'spacecraft': sensor_id,
            'timestamp': times_t[i],
            'datetime_utc': dt,
            'original_index': i,
            'crs_wkt': crs_wkt_t,
            'geo_transform': geo_tf_t,
            'ortho_visual': rgba,
            'sliding_volume_z_score_masked': z_score,
            'qa_mask': qa_mask
        })

    # Sort the unified datacube chronologically to enable time-series analysis
    print("Sorting unified datacube temporally...")
    datacube.sort(key=lambda x: x['timestamp'])   
    return datacube, h5_l, h5_t

# ==========================================
# 3. RESEARCHER UTILITIES
# ==========================================

def map_pixel_to_latlon(row_y, col_x, geo_transform, crs_wkt):
    """
    Converts a matrix index (Row, Col) into rigorous real-world coordinates (Lat, Lon).
    Adheres strictly to GDAL Affine Transformation metrology.
    """
    # Create the GDAL-compliant affine transformation matrix
    affine = rasterio.transform.Affine.from_gdal(*geo_transform)
    crs = CRS.from_wkt(crs_wkt)
    
    # Mathematical projection: transform back to WGS84 (EPSG:4326)
    transformer_to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    
    # Add 0.5 to target the physical geometric centroid of the pixel
    proj_x, proj_y = affine * (col_x + 0.5, row_y + 0.5)
    lon, lat = transformer_to_ll.transform(proj_x, proj_y)
    
    return lat, lon

# ==========================================
# 4. EXAMPLE USAGE / SKELETON EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Initializing Harmonized Multi-Sensor Data Framework...")
    
    cube, h5_landsat, h5_tanager = load_merged_datacube(landsat_path, tanager_path)
    
    print(f"\nSuccessfully loaded {len(cube)} temporally ordered frames.")
    
    # Extract and stack arrays from the temporally sorted list of dictionaries
    local_z_score = np.stack([frame['sliding_volume_z_score_masked'] for frame in cube])
    valid_mask = np.stack([frame['qa_mask'] for frame in cube])
    
    # Ensure any intrinsic NaNs in the data are manually masked out before passing to the model
    valid_mask[np.isnan(local_z_score)] = 0
    
    # Convert POSIX timestamps to numpy datetime64 array for Pandas DatetimeIndex compatibility
    acq_times = np.array([frame['timestamp'] for frame in cube], dtype='datetime64[s]')
    
    # Cleanup open file handles before model execution
    h5_landsat.close()
    h5_tanager.close()

    # Note on strict failure handling:
    # Do not explicitly set local_z_score[~valid_mask] = np.nan here. 
    # In PyTorch, (NaN * 0) evaluates to NaN. If explicit NaNs are injected into the 
    # target tensor, the valid_mask applied inside `calculate_masked_gaussian_nll` 
    # will still yield a NaN loss, crashing the gradient. The masking logic inherently 
    # ignores these invalid pixels. However, if the source data natively contains NaNs 
    # (e.g. processing failures), the pipeline will correctly hard-fail as requested.
    
    # Initialize computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate expanded feature channels matching spatiotemporal_encoders.py
    # 1 (raw image) + 40 (Fourier freq=10) + 16 (Time2Vec dim=16) + 2 (DOY sin/cos) + 1 (QA Mask) = 60
    num_freqs = 10
    t2v_dim = 16
    encoded_channels = 1 + (4 * num_freqs) + t2v_dim + 2 + 1
    
    # Initialize the Transformer model
    N, H, W = local_z_score.shape
    model = DirectMultiHorizonTransformer(
        input_channels=encoded_channels, 
        patch_pixels=H * W, # Assuming monolithic global patch for this implementation
        d_model=128
    ).to(device)

    print("Executing Transformer Pipeline...")
    mu_pred, logvar_pred, loss = run_transformer_pipeline(
        local_z_score=local_z_score, 
        valid_mask=valid_mask, 
        acq_times=acq_times,
        model=model,
        device=device
    )
    
    print(f"Pipeline executed successfully. Reconstruction Loss (NLL): {loss.item():.4f}")