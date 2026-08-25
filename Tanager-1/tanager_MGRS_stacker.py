import os
import sys
import glob
import math
import numpy as np
import h5py
from pathlib import Path
from pyproj import CRS, Transformer
from rasterio.warp import reproject, Resampling
from rasterio.transform import Affine
from rasterio.control import GroundControlPoint
import rasterio
from PIL import Image

# Add parent folder to sys.path to find SpecComplex
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc

SOURCE_DIR = r"C:\satelliteImagery\Tanager"
OUTPUT_DIR = r"C:\satelliteImagery\Tanager"
LOCATION='Tait'

MIN_ROI_COVERAGE_PERCENT = 25.0 
SUN_ELEVATION_THRESHOLD = 30
TANAGER_CLOUD_DILATION = 4
TANAGER_UNCERTAINTY_THRESHOLD = 0.1
TANAGER_AEROSOL_THRESHOLD = 0.35

def get_utm_epsg_from_lonlat(lon, lat):
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return int(epsg_code)

def intersects(bbox1, bbox2):
    """Evaluates whether two [min_lon, min_lat, max_lon, max_lat] bounding boxes intersect."""
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

def calculate_mgrs_aligned_grid(chunks, roi_bbox=None):
    """
    Computes a collective 30-meter UTM grid mathematically matching the MGRS format used by HLS.
    If roi_bbox is provided, forces the grid to strictly encapsulate the ROI instead of the chunks.
    """
    # 1. Aggregate all geographic coordinates to find the centroid and UTM zone
    all_lons = []
    all_lats = []
    
    if roi_bbox:
        min_lon, min_lat, max_lon, max_lat = roi_bbox
        all_lons = [min_lon, max_lon]
        all_lats = [min_lat, max_lat]
        corners = [
            (min_lon, max_lat),
            (max_lon, max_lat),
            (max_lon, min_lat),
            (min_lon, min_lat)
        ]
    else:
        for chunk in chunks:
            for lon, lat in chunk['bounds_lonlat']:
                all_lons.append(lon)
                all_lats.append(lat)
            
    center_lon = np.mean(all_lons)
    center_lat = np.mean(all_lats)
    
    epsg_code = get_utm_epsg_from_lonlat(center_lon, center_lat)
    target_crs = f"EPSG:{epsg_code}"
    
    # 2. Transform all corner points into the target UTM CRS
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    all_xs = []
    all_ys = []
    if roi_bbox:
        xs, ys = zip(*corners)
        proj_xs, proj_ys = transformer.transform(xs, ys)
        all_xs.extend(proj_xs)
        all_ys.extend(proj_ys)
    else:
        for chunk in chunks:
            xs, ys = zip(*chunk['bounds_lonlat'])
            proj_xs, proj_ys = transformer.transform(xs, ys)
            all_xs.extend(proj_xs)
            all_ys.extend(proj_ys)
        
    min_x, max_x = min(all_xs), max(all_xs)
    min_y, max_y = min(all_ys), max(all_ys)
    
    # 3. Snap boundaries to exact 30-meter multiples (MGRS HLS standard)
    ul_x = math.floor(min_x / 30.0) * 30.0
    ul_y = math.ceil(max_y / 30.0) * 30.0
    lr_x = math.ceil(max_x / 30.0) * 30.0
    lr_y = math.floor(min_y / 30.0) * 30.0
    
    width = int((lr_x - ul_x) / 30.0)
    height = int((ul_y - lr_y) / 30.0)
    
    target_transform = Affine.translation(ul_x, ul_y) * Affine.scale(30.0, -30.0)
    
    return target_transform, width, height, target_crs, (ul_x, ul_y), (lr_x, lr_y)


def process_tanager_mgrs_stack(target_location):
    print(f"Discovering Tanager collections for location: {target_location}...")
    import yaml
    
    config_path = os.path.join(script_dir, "locations_config.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(script_dir.parent, "locations_config.yaml")
        
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
        
    loc_data = config_data.get("locations", {}).get(target_location)
    if not loc_data:
        raise ValueError(f"CRITICAL: Location {target_location} not found in locations_config.yaml")
        
    roi_bbox = [
        loc_data["ROI_LON_MIN"],
        loc_data["ROI_LAT_MIN"],
        loc_data["ROI_LON_MAX"],
        loc_data["ROI_LAT_MAX"]
    ]
    
    source_cache = loc_data.get("SOURCE_CACHE") or target_location
    location_dir = os.path.join(SOURCE_DIR, f"{source_cache}_SourceData")
    if not os.path.exists(location_dir):
        raise FileNotFoundError(f"CRITICAL: Directory {location_dir} does not exist.")
        
    basic_files = glob.glob(os.path.join(location_dir, "**", "*_basic_sr_hdf5.h5"), recursive=True)
    if not basic_files:
        raise FileNotFoundError(f"CRITICAL: No basic_sr_hdf5 files found in {location_dir}.")

    # 1. Parse all chunks and find bounds
    chunks = []
    for f in basic_files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) >= 1:
            pass_ts = parts[0]
            
            with h5py.File(f, 'r') as h5f:
                lat = h5f['HDFEOS/SWATHS/HYP/Geolocation Fields/Latitude'][:]
                lon = h5f['HDFEOS/SWATHS/HYP/Geolocation Fields/Longitude'][:]
                bounds = [
                    (lon[0, 0], lat[0, 0]),
                    (lon[0, -1], lat[0, -1]),
                    (lon[-1, -1], lat[-1, -1]),
                    (lon[-1, 0], lat[-1, 0])
                ]
            
            chunk_bbox = [
                min(l[0] for l in bounds),
                min(l[1] for l in bounds),
                max(l[0] for l in bounds),
                max(l[1] for l in bounds)
            ]
            
            if intersects(chunk_bbox, roi_bbox):
                chunks.append({
                    'file': f,
                    'pass_ts': pass_ts,
                    'bounds_lonlat': bounds
                })
            
    # Group chunks into temporal passes
    passes = {}
    for chunk in chunks:
        if chunk['pass_ts'] not in passes:
            passes[chunk['pass_ts']] = []
        passes[chunk['pass_ts']].append(chunk)

    pass_keys = sorted(list(passes.keys()))
    n_times = len(pass_keys)
    print(f"Aggregated {len(chunks)} chunks into {n_times} temporal passes.")

    if not chunks:
        raise ValueError(f"CRITICAL: No chunks intersect the bounding box for {target_location}.")

    # 2. Establish Universal MGRS-Aligned Grid
    tf_target, width, height, crs_wkt, ul_coords, lr_coords = calculate_mgrs_aligned_grid(chunks, roi_bbox=roi_bbox)
    print(f"MGRS Target Grid: {width}x{height} 30m pixels (CRS: {crs_wkt})")
    print(f"Origin (UL): {ul_coords}, Lower Right: {lr_coords}")

    output_file = os.path.join(OUTPUT_DIR, f"Tanager_MGRS_Stack_{target_location}.h5")
    
    # 3. HDF5 Tensor Construction
    with h5py.File(output_file, 'w') as out_h5:
        grp_tanager = out_h5.create_group("HDFEOS/GRIDS/TANAGER/Data Fields")
        meta_lists = {'acq_time': [], 'space_id': [], 'good_wavelengths': []}
        
        chunk_h, chunk_w = min(height, 256), min(width, 256)
        
        with h5py.File(chunks[0]['file'], 'r') as f_meta:
            src_df = f_meta['HDFEOS/SWATHS/HYP/Data Fields']
            for name in src_df.keys():
                src_dset = src_df[name]
                dtype = src_dset.dtype
                is_3d = len(src_dset.shape) == 3
                bands = src_dset.shape[0] if is_3d else None
                out_shape = (n_times, bands, height, width) if is_3d else (n_times, height, width)
                chunks_dim = (1, bands, chunk_h, chunk_w) if is_3d else (1, chunk_h, chunk_w)
                
                fill_val = src_dset.attrs.get("_FillValue", 0 if dtype.kind in ['i', 'u'] else -9999.0)
                out_dset = grp_tanager.create_dataset(name, shape=out_shape, dtype=dtype, compression="gzip", compression_opts=5, fillvalue=fill_val, chunks=chunks_dim)
                
                # Copy attributes from source datasets
                for attr_name, attr_val in src_dset.attrs.items():
                    out_dset.attrs[attr_name] = attr_val

        ds_vis = grp_tanager.create_dataset("ortho_visual", shape=(n_times, 4, height, width), dtype='uint8', compression="gzip", fillvalue=0)
        
        gdal_transform = [tf_target.c, tf_target.a, tf_target.b, tf_target.f, tf_target.d, tf_target.e]
        
        # 4. Iterative Data Assimilation
        for t_idx, pass_ts in enumerate(pass_keys):
            print(f"  [Pass {t_idx+1}/{n_times}] Assimilating Swath: {pass_ts}...")
            chunk_dicts = passes[pass_ts]
            
            pass_canvases = {}
            pass_times = []
            
            for name in grp_tanager.keys():
                if name == "ortho_visual": continue
                dtype = grp_tanager[name].dtype
                is_3d = len(grp_tanager[name].shape) == 4
                bands = grp_tanager[name].shape[1] if is_3d else None
                canvas_shape = (bands, height, width) if is_3d else (height, width)
                fill_val = grp_tanager[name].fillvalue                    
                pass_canvases[name] = np.full(canvas_shape, fill_val, dtype=dtype)
                
            gw_found = False
            for c_idx, chunk_info in enumerate(chunk_dicts):
                chunk_file = chunk_info['file']
                with h5py.File(chunk_file, 'r') as f_chunk:
                    df_grp = f_chunk['HDFEOS/SWATHS/HYP/Data Fields']
                    geo_grp = f_chunk['HDFEOS/SWATHS/HYP/Geolocation Fields']
                    lat = geo_grp['Latitude'][:]
                    lon = geo_grp['Longitude'][:]
                    pass_times.extend(geo_grp['Time'][:].tolist())
                    
                    if c_idx == 0:
                        gw = df_grp['surface_reflectance'].attrs.get('good_wavelengths')
                        if gw is not None:
                            meta_lists['good_wavelengths'].append(gw)
                            gw_found = True
                            
                    # Construct GCPs (sampled every 10 rows/cols)
                    gcps = []
                    step = 10
                    rows = list(range(0, lat.shape[0], step))
                    if rows[-1] != lat.shape[0] - 1:
                        rows.append(lat.shape[0] - 1)
                    cols = list(range(0, lat.shape[1], step))
                    if cols[-1] != lat.shape[1] - 1:
                        cols.append(lat.shape[1] - 1)
                        
                    for r in rows:
                        for c in cols:
                            gcps.append(GroundControlPoint(row=r, col=c, x=lon[r, c], y=lat[r, c]))

                    for name in pass_canvases.keys():
                        is_3d = len(grp_tanager[name].shape) == 4
                        bands = grp_tanager[name].shape[1] if is_3d else None
                        dtype = df_grp[name].dtype
                        fill_val = grp_tanager[name].fillvalue
                        resample_algo = Resampling.nearest
                        if dtype.kind in ['i', 'u', 'b']:
                            resample_algo = Resampling.nearest
                        
                        src_data = df_grp[name][:]
                        if not is_3d:
                            src_data = src_data[np.newaxis, ...]
                            incoming = np.full((1, height, width), fill_val, dtype=dtype)
                        else:
                            incoming = np.full((bands, height, width), fill_val, dtype=dtype)
                            
                        reproject(
                            source=src_data,
                            destination=incoming,
                            src_transform=None,
                            gcps=gcps,
                            src_crs="EPSG:4326",
                            dst_transform=tf_target,
                            dst_crs=crs_wkt,
                            resampling=resample_algo,
                            src_nodata=fill_val,
                            dst_nodata=fill_val,
                            tps=True
                        )
                        
                        # Apply boolean valid mask exactly like EnMAP fixing logic
                        if dtype.kind in ['f', 'c'] and np.isnan(fill_val):
                            valid_mask = ~np.isnan(incoming)
                        else:
                            valid_mask = ~np.isclose(incoming, fill_val, equal_nan=True)
                            
                        if not is_3d:
                            pass_canvases[name][valid_mask[0]] = incoming[0][valid_mask[0]]
                        else:
                            pass_canvases[name][valid_mask] = incoming[valid_mask]
                            
            if len(pass_times) > 0:
                meta_lists['acq_time'].append(np.mean(pass_times))
            else:
                meta_lists['acq_time'].append(0.0)
            meta_lists['space_id'].append('Tanager-1')
            
            if not gw_found:
                bands = grp_tanager['surface_reflectance'].shape[1]
                meta_lists['good_wavelengths'].append(np.zeros(bands, dtype=bool))
                
            for name in pass_canvases.keys():
                grp_tanager[name][t_idx, ...] = pass_canvases[name]
                
        # Generate Common Mask
        print("  Generating Common Mask for Tanager on MGRS Grid...")
        mask_ds = grp_tanager.create_dataset('common_mask', shape=(n_times, height, width), dtype=bool, compression="gzip", compression_opts=5, chunks=(1, chunk_h, chunk_w), fillvalue=False)
        mask_ds.attrs['description'] = "True = Invalid/Masked, False = Valid. Generated from SpecComplex ARD rules."
        mask_ds.attrs['cloud_dilation'] = TANAGER_CLOUD_DILATION
        mask_ds.attrs['uncertainty_threshold'] = TANAGER_UNCERTAINTY_THRESHOLD
        mask_ds.attrs['aerosol_depth_threshold'] = TANAGER_AEROSOL_THRESHOLD
        mask_ds.attrs['sun_elevation_threshold'] = SUN_ELEVATION_THRESHOLD
        for out_idx in range(n_times):
            valid_mask = sc.get_tanager_mask(grp_tanager, out_idx, (height, width),
                                             sun_elevation_threshold=SUN_ELEVATION_THRESHOLD,
                                             cloud_dilation=TANAGER_CLOUD_DILATION,
                                             apply_cloud_mask=True,
                                             uncertainty_threshold=TANAGER_UNCERTAINTY_THRESHOLD,
                                             aerosol_depth_threshold=TANAGER_AEROSOL_THRESHOLD)
            mask_ds[out_idx, ...] = valid_mask
            
        print("  Generating strict 'ortho_visual' RGB composite from SR...")
        wavelengths = grp_tanager['surface_reflectance'].attrs.get('wavelengths', np.zeros(grp_tanager['surface_reflectance'].shape[1]))
        r_idx = int(np.argmin(np.abs(wavelengths - 650)))
        g_idx = int(np.argmin(np.abs(wavelengths - 550)))
        b_idx = int(np.argmin(np.abs(wavelengths - 450)))
        
        sr_dset_ref = grp_tanager["surface_reflectance"]
        for out_idx in range(n_times):
            r_band = sr_dset_ref[out_idx, r_idx, :, :]
            g_band = sr_dset_ref[out_idx, g_idx, :, :]
            b_band = sr_dset_ref[out_idx, b_idx, :, :]
            
            # Mask out values < -1 or == fillvalue as nan for PNG creation
            r_input = np.where(r_band < -1, np.nan, r_band)
            g_input = np.where(g_band < -1, np.nan, g_band)
            b_input = np.where(b_band < -1, np.nan, b_band)
            
            rgba_img = sc.generate_rgba_image(r_input, g_input, b_input)
            ds_vis[out_idx, ...] = np.transpose(rgba_img, (2, 0, 1))

        # Attributes
        dt_str = h5py.string_dtype(encoding='ascii')
        grp_tanager['surface_reflectance'].attrs['acquisition_time'] = np.array(meta_lists['acq_time'], dtype='float64')
        grp_tanager['surface_reflectance'].attrs.create('spacecraft_id', data=np.array(meta_lists['space_id'], dtype=dt_str))
        if len(meta_lists['good_wavelengths']) == n_times:
            grp_tanager['surface_reflectance'].attrs['all_good_wavelengths'] = np.array(meta_lists['good_wavelengths'], dtype=bool)

        for name in grp_tanager.keys():
            grp_tanager[name].attrs['spatial_ref'] = crs_wkt
            grp_tanager[name].attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

    # Save PNGs
    print("  Exporting visual frames as PNGs...")
    with h5py.File(output_file, 'r') as out_h5:
        vis = out_h5["HDFEOS/GRIDS/TANAGER/Data Fields/ortho_visual"]
        mask = out_h5["HDFEOS/GRIDS/TANAGER/Data Fields/common_mask"]
        for t_idx in range(n_times):
            pass_ts = pass_keys[t_idx]
            frame_rgba = vis[t_idx, ...]
            # (4, H, W) -> (H, W, 4)
            frame_rgba = np.transpose(frame_rgba, (1, 2, 0))
            img = Image.fromarray(frame_rgba, 'RGBA')
            png_name = f"Tanager_{target_location}_{pass_ts}_RGB.png"
            png_path = os.path.join(location_dir, png_name)
            img.save(png_path)
            print(f"    Saved: {png_name}")
            
            frame_mask = mask[t_idx, ...]
            overlay = Image.new('RGBA', img.size, (255, 166, 0, 0))
            overlay_data = np.array(overlay)
            overlay_data[frame_mask == True] = [255, 166, 0, 128]
            overlay_img = Image.fromarray(overlay_data, 'RGBA')
            masked_img = Image.alpha_composite(img, overlay_img)
            
            masked_png_name = f"Tanager_{target_location}_{pass_ts}_RGB_masked.png"
            masked_png_path = os.path.join(location_dir, masked_png_name)
            masked_img.save(masked_png_path)
            print(f"    Saved: {masked_png_name}")

    print(f"\nTensor Synthesis Complete. MGRS Stack Stored at: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default=LOCATION, help="Target location prefix")
    args = parser.parse_args()
    process_tanager_mgrs_stack(args.location)
