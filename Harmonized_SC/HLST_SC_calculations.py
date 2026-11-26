"""
Calculates derivative spectral complexity metrics for HDFEOS compliant grids.
Stores calculated values in-place directly inside the input HDF5 file.
"""

import os
import sys
import time
import warnings
from pathlib import Path
import h5py
import numpy as np
# Add parent folder to sys.path to find SpecComplexTorch / SpecComplex
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))

import SpecComplexTorch as sc

# Spatial smoothing configuration:
# When False (default for HLST pipeline), uses direct 3x3 center-pixel neighborhood_map without
# secondary uniform_filter averaging. This prevents 5x5 spatial blurring and volume inflation
# across wetland/forest boundaries.
APPLY_SPATIAL_SMOOTHING = False


def scale_to_int16(raw_floats, scale_factor=10000, nodata_val=-9999, nodata_value=None):
    """Quantizes float32 array to int16, mapping NaNs to nodata value."""
    if nodata_value is not None:
        nodata_val = nodata_value
    raw_floats = np.asarray(raw_floats)
    out = np.full(raw_floats.shape, nodata_val, dtype=np.int16)
    valid = ~np.isnan(raw_floats)
    scaled = np.round(raw_floats[valid] * float(scale_factor))
    clipped = np.clip(scaled, -32768, 32767)
    out[valid] = clipped.astype(np.int16)
    return out


def read_scaled_int16(dataset, scale_factor=10000, nodata_val=-9999, slice_obj=np.s_[...]):
    """Invert int16 quantized array or HDF5 dataset back to float32 reflectance."""
    if hasattr(dataset, 'attrs'):
        sf = dataset.attrs.get("scale_factor")
        if sf is not None:
            scale_factor = float(sf)
        fv = dataset.attrs.get("_FillValue", dataset.attrs.get("fill_value", dataset.attrs.get("fillvalue")))
        if fv is not None:
            nodata_val = fv
        data = dataset[slice_obj] if hasattr(dataset, '__getitem__') else dataset
    else:
        data = np.asarray(dataset)

    out = np.full(data.shape, np.nan, dtype=np.float32)
    valid = (data != nodata_val)
    out[valid] = data[valid].astype(np.float32) / float(scale_factor)
    return out


def load_enmap_sr(dataset, frame_idx, spatial_slice=np.s_[:, :]):
    """
    Consistent method to load EnMAP surface reflectance from source .h5 files.
    Correctly handles int to float conversion, converts EnMAP specific bad pixels (-32767) 
    to NaN, and strips water absorption/defective bands.
    """
    if isinstance(spatial_slice, tuple):
        raw_slice = (frame_idx, slice(None)) + spatial_slice
    else:
        raw_slice = (frame_idx, slice(None), spatial_slice)
        
    float_data = load_scaled_reflectance(dataset, raw_slice)
    scale = dataset.attrs.get("scale_to_float", 1.0)
    enmap_bad_pixel_scaled = -32767 * scale
    float_data[np.isclose(float_data, enmap_bad_pixel_scaled, atol=1e-6)] = np.nan
    
    gw_attr = dataset.attrs.get("all_good_wavelengths")
    wavelengths = dataset.attrs.get("wavelengths")[:]
    
    if gw_attr is not None:
        gw_mask = gw_attr[frame_idx].astype(bool)
        float_data_pruned = float_data[gw_mask, ...]
        valid_wavelengths = wavelengths[gw_mask]
    else:
        float_data_pruned = float_data
        valid_wavelengths = wavelengths
        
    return float_data_pruned, valid_wavelengths


# ==============================================================================
# INLINED DATA LOADERS, QUANTIZATION, AND PROCESSING HELPERS
# ==============================================================================

def load_scaled_reflectance(dataset, slice_obj=np.s_[...]):
    """
    Safely load a surface reflectance dataset from HDF5, detect if it is scaled integer,
    and convert it to float32 reflectance. Converts nodata values into np.nan.
    Fails explicitly if an integer dataset lacks a fillvalue or scale factor attribute.
    """
    data = dataset[slice_obj] if hasattr(dataset, '__getitem__') and hasattr(dataset, 'attrs') else dataset

    if np.issubdtype(data.dtype, np.integer):
        nodata = None
        if hasattr(dataset, "attrs"):
            if "_FillValue" in dataset.attrs:
                nodata = dataset.attrs["_FillValue"]
            elif "fillvalue" in dataset.attrs:
                nodata = dataset.attrs["fillvalue"]
            elif "fill_value" in dataset.attrs:
                nodata = dataset.attrs["fill_value"]

        if nodata is None and hasattr(dataset, "fillvalue") and dataset.fillvalue is not None:
            if dataset.fillvalue != 0:
                nodata = dataset.fillvalue

        if nodata is None:
            raise KeyError(
                f"Dataset {getattr(dataset, 'name', 'dataset')} has integer datatype ({data.dtype}) "
                "but has no '_FillValue' attribute or non-zero fillvalue defined."
            )

        scale = None
        if hasattr(dataset, "attrs"):
            if "scale_to_float" in dataset.attrs:
                scale = float(dataset.attrs["scale_to_float"])
            elif "scale_factor" in dataset.attrs:
                sf = float(dataset.attrs["scale_factor"])
                scale = (1.0 / sf) if sf > 1.0 else sf

        if scale is None:
            raise KeyError(
                f"Dataset {getattr(dataset, 'name', 'dataset')} has integer datatype ({data.dtype}) "
                "but has no 'scale_to_float' or 'scale_factor' attribute defined."
            )

        float_data = data.astype(np.float32)
        float_data[data == nodata] = np.nan
        float_data *= scale
        return float_data

    if np.issubdtype(data.dtype, np.floating):
        nodata = None
        if hasattr(dataset, "attrs"):
            if "_FillValue" in dataset.attrs:
                nodata = dataset.attrs["_FillValue"]
            elif "fillvalue" in dataset.attrs:
                nodata = dataset.attrs["fillvalue"]
            elif "fill_value" in dataset.attrs:
                nodata = dataset.attrs["fill_value"]

        if nodata is None and hasattr(dataset, "fillvalue") and dataset.fillvalue is not None:
            nodata = dataset.fillvalue

        if nodata is not None and not np.isnan(nodata):
            float_data = data.copy()
            float_data[np.isclose(data, nodata, equal_nan=True)] = np.nan
            return float_data

    return data


def load_tanager_sr(dataset, frame_idx, spatial_slice=np.s_[:, :]):
    """
    Load Tanager surface reflectance from source H5 dataset.
    Only invokes load_scaled_reflectance if the dataset has an integer datatype.
    Strips defective or water absorption bands according to good wavelengths metadata.
    """
    if isinstance(spatial_slice, tuple):
        raw_slice = (frame_idx, slice(None)) + spatial_slice
    else:
        raw_slice = (frame_idx, slice(None), spatial_slice)

    dataset_dtype = getattr(dataset, "dtype", np.asarray(dataset).dtype)
    if np.issubdtype(dataset_dtype, np.integer):
        float_data = load_scaled_reflectance(dataset, raw_slice)
    else:
        float_data = dataset[raw_slice]

    gw_attr = dataset.attrs.get("all_good_wavelengths") if hasattr(dataset, 'attrs') else None
    wavelengths = dataset.attrs.get("wavelengths") if hasattr(dataset, 'attrs') else None
    if wavelengths is not None:
        wavelengths = np.asarray(wavelengths)

    if gw_attr is not None:
        gw_mask = gw_attr[frame_idx].astype(bool)
        float_data_pruned = float_data[gw_mask, ...]
        valid_wavelengths = wavelengths[gw_mask] if wavelengths is not None else None
    else:
        float_data_pruned = float_data
        valid_wavelengths = wavelengths

    return float_data_pruned, valid_wavelengths



def process_volume_frame(frame_data, num_endmembers=7, gram_type='minEndmember', norm_param=None):
    """
    Process image frame to extract global endmembers, indices, and local Gram volume curve.
    Invokes sc.maximumDistance and sc.calcGramLocalVolumes.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    image2D = np.reshape(img, (height * width, bands), order="F").copy()

    image2D = np.clip(image2D, 0.0, 1.0)

    valid_mask = ~np.isnan(image2D).any(axis=1)
    if np.sum(valid_mask) < num_endmembers:
        return (
            np.full((bands, num_endmembers), np.nan, dtype=np.float32),
            np.full(num_endmembers, -1, dtype=np.int32),
            np.full(num_endmembers, np.nan, dtype=np.float32)
        )

    valid_indices = np.where(valid_mask)[0]
    valid_data = image2D[valid_mask].astype(np.float32).T

    endmembers = sc.maximumDistance(valid_data, num_endmembers)

    endmember_indices = np.zeros(num_endmembers, dtype=np.int32)
    for k in range(num_endmembers):
        diffs = np.sum(np.abs(valid_data - endmembers[:, k:k + 1]), axis=0)
        match_k = np.argmin(diffs)
        endmember_indices[k] = valid_indices[match_k]

    mean_vector = np.nanmean(img, axis=(0, 1))
    localization_vec = endmembers[:, 1]

    remaining_endmembers = np.delete(endmembers, 1, axis=1)
    volume = sc.calcGramLocalVolumes(remaining_endmembers, localization_vector=localization_vec, divide_by_factorial=False, axis=1)
    volume = np.insert(volume, 0, 0.0)

    return endmembers.astype(np.float32), endmember_indices.astype(np.int32), volume.astype(np.float32)


# ==============================================================================
# DATASET CREATION AND MANAGEMENT HELPERS
# ==============================================================================

def overwrite_dset(data_grp, name, shape, dtype, fillvalue=None, spatial_ref=None, geo_transform=None, chunks=None, **kwargs):
    """Creates or overwrites a dataset in an HDF5 group with gzip level 6 compression."""
    if name in data_grp:
        del data_grp[name]

    create_kwargs = {
        "shuffle": True,
        "compression": "gzip",
        "compression_opts": 6,
        "chunks": chunks,
        **kwargs
    }
    if fillvalue is not None:
        create_kwargs["fillvalue"] = fillvalue

    ds = data_grp.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        **create_kwargs
    )

    if spatial_ref is not None:
        ds.attrs['spatial_ref'] = spatial_ref
    if geo_transform is not None:
        ds.attrs['GeoTransform'] = geo_transform
    if fillvalue is not None:
        ds.attrs['_FillValue'] = fillvalue

    return ds


# ==============================================================================
# PIPELINE EXECUTION ENGINE
# ==============================================================================

def process_ard_cube(filepath, tile_size=3, num_endmembers=7, norm_param=None, metric="all", apply_spatial_smoothing=APPLY_SPATIAL_SMOOTHING):
    """
    Processes the multi-sensor ARD cube in-place, appending the required metrics:
      1. CALC_GLOBAL_ENDMEMBERS
      2. CALC_SLIDING_VOLUME (baseline sliding_volume_map / sliding_volume_z_score)
      3. CALC_3D_VOLUME (OSDprodnormalized 3D-equivalent volume_map_3d / volume_map_3d_zscore)
    directly back into the source HDF5 file.
    """
    calc_baseline = metric.lower() in ["all", "both", "baseline"]
    calc_3d = metric.lower() in ["all", "both", "3d"]
    smoothing_str = "spatially averaged" if apply_spatial_smoothing else "direct neighborhood_map (no spatial averaging)"
    print(f"\nProcessing in-place spectral complexity calculations (metric={metric}, smoothing={smoothing_str}) for: {filepath}")

    with h5py.File(filepath, 'r+') as h5_file:
        grids = [g for g in h5_file['/HDFEOS/GRIDS'].keys() if g != 'HARMONIZED']
        if not grids:
            raise ValueError("No sensor grids found to process in input HDF5 file.")

        timeline = []
        for grid in grids:
            base_path = f"/HDFEOS/GRIDS/{grid}/Data Fields"
            if base_path not in h5_file:
                raise ValueError(f"Data Fields missing for {grid}")

            data_grp = h5_file[base_path]
            for req_ds in ["surface_reflectance", "common_mask"]:
                if req_ds not in data_grp:
                    raise ValueError(f"'{req_ds}' missing in {grid}.")

            acq_times = data_grp["surface_reflectance"].attrs['acquisition_time']
            spacecraft_ids = data_grp["surface_reflectance"].attrs['spacecraft_id']

            for i, ts in enumerate(acq_times):
                sp_id = spacecraft_ids[i]
                sp_str = sp_id.decode('utf-8') if isinstance(sp_id, bytes) else str(sp_id)
                timeline.append({'time': ts, 'grid': grid, 'local_idx': i, 'spacecraft': sp_str})

        timeline.sort(key=lambda x: x['time'])
        total_frames = len(timeline)
        print(f"Global Timeline Established: {total_frames} frames across {len(grids)} sensors.")

        ref_sr = h5_file[f"/HDFEOS/GRIDS/{grids[0]}/Data Fields/surface_reflectance"]
        ref_grid = h5_file[f"/HDFEOS/GRIDS/{grids[0]}/Data Fields/common_mask"]
        height, width = ref_grid.shape[-2:]
        spatial_ref = ref_grid.attrs.get('spatial_ref') or ref_sr.attrs.get('spatial_ref')
        geo_transform = ref_grid.attrs.get('GeoTransform') if 'GeoTransform' in ref_grid.attrs else ref_sr.attrs.get('GeoTransform')

        print("Pre-calculating global persistent water mask from Fmask history...")
        water_sum = np.zeros((height, width), dtype=np.int32)
        valid_sum = np.zeros((height, width), dtype=np.int32)

        for meta in timeline:
            g_name = meta['grid']
            l_idx = meta['local_idx']
            fmask_path = f"/HDFEOS/GRIDS/{g_name}/Data Fields/Fmask"
            if fmask_path in h5_file:
                fmask_frame = h5_file[fmask_path][l_idx, :, :]
                valid_pixels = (fmask_frame != 255)
                water_pixels = ((fmask_frame & 32) > 0)
                water_sum[valid_pixels] += water_pixels[valid_pixels]
                valid_sum[valid_pixels] += 1

        if np.max(water_sum) > 0:
            water_sum_masked = np.ma.masked_where(valid_sum == 0, water_sum)
            binary_mask_masked = water_sum_masked >= (1.0 / 3.0 * np.max(water_sum))
            persistent_water_mask = binary_mask_masked.filled(False)
        else:
            persistent_water_mask = np.zeros((height, width), dtype=bool)

        print(f"Persistent water mask computed. Marked {np.sum(persistent_water_mask)} pixels as persistent water.")

        chunk_h, chunk_w = min(height, 256), min(width, 256)
        chunks_3d = (1, chunk_h, chunk_w)

        harm_base = '/HDFEOS/GRIDS/HARMONIZED'
        harm_path = f"{harm_base}/Data Fields"
        if harm_path in h5_file:
            harm_grp = h5_file[harm_path]
        else:
            if harm_base not in h5_file:
                h5_file.create_group(harm_base)
            harm_grp = h5_file.create_group(harm_path)

        if calc_baseline or "common_mask" not in harm_grp:
            ds_harm_mask = overwrite_dset(harm_grp, 'common_mask', (total_frames, height, width), dtype='uint8', fillvalue=255, spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
            ds_harm_ortho = overwrite_dset(harm_grp, 'ortho_visual', (total_frames, 3, height, width), dtype='uint8', fillvalue=0, spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=(1, 3, chunk_h, chunk_w))
            ds_harm_ndvi = overwrite_dset(harm_grp, 'ndvi', (total_frames, height, width), dtype='float32', fillvalue=np.nan, spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
        else:
            ds_harm_mask = harm_grp['common_mask']
            ds_harm_ortho = harm_grp['ortho_visual']
            ds_harm_ndvi = harm_grp['ndvi']

        ds_harm_slide = None
        ds_harm_z = None
        if calc_baseline:
            ds_harm_slide = overwrite_dset(harm_grp, 'sliding_volume_map', (total_frames, height, width), dtype='float32', fillvalue=np.nan, spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
            ds_harm_z = overwrite_dset(harm_grp, 'sliding_volume_z_score', (total_frames, height, width), dtype='float32', fillvalue=np.nan, spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
        elif 'sliding_volume_map' in harm_grp:
            ds_harm_slide = harm_grp['sliding_volume_map']
            ds_harm_z = harm_grp['sliding_volume_z_score']

        ds_harm_3d = None
        ds_harm_3d_z = None
        if calc_3d:
            ds_harm_3d = overwrite_dset(harm_grp, 'volume_map_3d', (total_frames, height, width), dtype='float32', fillvalue=np.nan, spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)
            ds_harm_3d_z = overwrite_dset(harm_grp, 'volume_map_3d_zscore', (total_frames, height, width), dtype='float32', fillvalue=np.nan, spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=chunks_3d)

        sensor_dsets = {}
        need_sensor_recompute = calc_baseline or ("frame_endmembers" not in h5_file[f"/HDFEOS/GRIDS/{grids[0]}/Data Fields"])
        if need_sensor_recompute:
            for grid in grids:
                data_grp = h5_file[f"/HDFEOS/GRIDS/{grid}/Data Fields"]
                sr_shape = data_grp["surface_reflectance"].shape
                n_frames, n_bands = sr_shape[0], sr_shape[1]

                em_ds = overwrite_dset(data_grp, 'frame_endmembers', (n_frames, n_bands, num_endmembers), dtype='float32', fillvalue=np.nan, spatial_ref=spatial_ref, geo_transform=geo_transform, chunks=(1, n_bands, num_endmembers))
                if 'wavelengths' in data_grp["surface_reflectance"].attrs:
                    em_ds.attrs['wavelengths'] = data_grp["surface_reflectance"].attrs['wavelengths']

                idx_ds = overwrite_dset(data_grp, 'frame_endmember_indices', (n_frames, num_endmembers), dtype='int32',fillvalue=-9999, chunks=(1, num_endmembers))
                vol_ds = overwrite_dset(data_grp, 'frame_endmember_volumes', (n_frames, num_endmembers), dtype='float32',fillvalue=np.nan, chunks=(1, num_endmembers))

                sensor_dsets[grid] = {'em': em_ds, 'idx': idx_ds, 'vol': vol_ds, 'num_bands': n_bands}
        else:
            for grid in grids:
                data_grp = h5_file[f"/HDFEOS/GRIDS/{grid}/Data Fields"]
                sr_shape = data_grp["surface_reflectance"].shape
                sensor_dsets[grid] = {'em': data_grp.get('frame_endmembers'), 'idx': data_grp.get('frame_endmember_indices'), 'vol': data_grp.get('frame_endmember_volumes'), 'num_bands': sr_shape[1]}

        global_means_list = []
        global_stds_list = []
        global_means_3d_list = []
        global_stds_3d_list = []

        print(f"Spooling {total_frames} frames into SpecComplexTorch Compute Engine...")
        t_start_pipeline = time.perf_counter()

        for global_idx, meta in enumerate(timeline):
            t_start_frame = time.perf_counter()
            grid_name = meta['grid']
            t_local = meta['local_idx']

            data_grp = h5_file[f"/HDFEOS/GRIDS/{grid_name}/Data Fields"]
            sr_dataset = data_grp["surface_reflectance"]
            frame_sr = load_scaled_reflectance(sr_dataset, np.s_[t_local, ...])
            frame_mask = data_grp["common_mask"][t_local, ...].copy()
            raw_frame_ortho = data_grp["ortho_visual"][t_local, ...]

            if persistent_water_mask is not None:
                frame_mask[persistent_water_mask] = 1

            valid_mask = (frame_mask == 0)

            gw_attr = data_grp["surface_reflectance"].attrs.get("all_good_wavelengths")
            if gw_attr is not None:
                gw_mask = gw_attr[t_local].astype(bool)
                eval_sr = frame_sr[gw_mask]
            else:
                gw_mask = None
                eval_sr = frame_sr

            if calc_baseline or "common_mask" not in harm_grp:
                frame_ortho = raw_frame_ortho
                if frame_ortho.shape[0] in [3, 4]:
                    frame_ortho = frame_ortho[:3, :, :]
                else:
                    frame_ortho = np.transpose(frame_ortho[..., :3], (2, 0, 1))

                if frame_ortho.dtype != np.uint8:
                    fo = frame_ortho.astype(np.float32)
                    ortho_ds = data_grp["ortho_visual"]
                    ortho_fill = ortho_ds.attrs.get("_FillValue", ortho_ds.attrs.get("fillvalue", getattr(ortho_ds, "fillvalue", None)))
                    if ortho_fill is not None and not np.isnan(ortho_fill):
                        valid_ortho = ~np.isclose(fo, ortho_fill, equal_nan=True) & ~np.isnan(fo)
                    else:
                        valid_ortho = ~np.isnan(fo)
                    if np.any(valid_ortho):
                        p1, p99 = np.percentile(fo[valid_ortho], (1, 99))
                        if p99 > p1:
                            fo[valid_ortho] = (fo[valid_ortho] - p1) / (p99 - p1)
                    fo[~valid_ortho] = 0.0
                    frame_ortho = np.clip(fo * 255, 0, 255).astype(np.uint8)

                ds_harm_mask[global_idx, ...] = frame_mask
                ds_harm_ortho[global_idx, ...] = frame_ortho

                wavelengths = data_grp["surface_reflectance"].attrs.get("wavelengths")
                if wavelengths is not None:
                    wl = np.asarray(wavelengths)
                    # Account for nanometers vs micrometers
                    scale_wl = 1000.0 if np.nanmax(wl) < 10.0 else 1.0
                    red_target, nir_target = 665.0, 865.0
                    red_idx = int(np.nanargmin(np.abs(wl * scale_wl - red_target)))
                    nir_idx = int(np.nanargmin(np.abs(wl * scale_wl - nir_target)))
                else:
                    if "HLSS30" in grid_name:
                        red_idx, nir_idx = 3, 7
                    elif "HLSL30" in grid_name:
                        red_idx, nir_idx = 3, 4
                    elif "TANAGER" in grid_name:
                        red_idx, nir_idx = 59, 97
                    else:
                        red_idx = min(3, frame_sr.shape[0] - 1)
                        nir_idx = min(4, frame_sr.shape[0] - 1)

                ndvi = sc.calc_ndvi_frame(frame_sr, red_idx, nir_idx)
                ds_harm_ndvi[global_idx, ...] = ndvi

            if calc_baseline:
                em, em_idx, vol_curve = process_volume_frame(eval_sr, num_endmembers)

                if gw_mask is not None:
                    em_full = np.full((sensor_dsets[grid_name]['num_bands'], num_endmembers), np.nan, dtype=np.float32)
                    em_full[gw_mask, :] = em
                    em_out = em_full
                else:
                    em_out = em

                sensor_dsets[grid_name]['em'][t_local, ...] = em_out
                sensor_dsets[grid_name]['idx'][t_local, ...] = em_idx
                sensor_dsets[grid_name]['vol'][t_local, ...] = vol_curve

                slide_map = sc.process_volume_sliding_tile(
                    eval_sr, tile_size=tile_size, stride=1, num_endmembers=num_endmembers,
                    apply_spatial_smoothing=apply_spatial_smoothing
                )
                ds_harm_slide[global_idx, ...] = slide_map

                z_map, global_mean, global_std = sc.calculate_global_z_score(slide_map, valid_mask)
                ds_harm_z[global_idx, ...] = z_map

                global_means_list.append(global_mean)
                global_stds_list.append(global_std)

            if calc_3d:
                # 3D equivalent volume (OSDprodnormalized extracting 20 endmembers with 1% delta reflectance stopping rule)
                slide_map_3d = sc.process_volume_3d_sliding_tile(
                    eval_sr, tile_size=tile_size, stride=1, num_endmembers=20, separability_threshold=0.01,
                    apply_spatial_smoothing=apply_spatial_smoothing
                )
                ds_harm_3d[global_idx, ...] = slide_map_3d

                z_map_3d, global_mean_3d, global_std_3d = sc.calculate_global_z_score(slide_map_3d, valid_mask)
                ds_harm_3d_z[global_idx, ...] = z_map_3d

                global_means_3d_list.append(global_mean_3d)
                global_stds_3d_list.append(global_std_3d)

            duration = time.perf_counter() - t_start_frame
            print(f"  [{global_idx + 1}/{total_frames}] {grid_name} (Global Index {global_idx}) processed in {duration:.2f}s")

        print("\nApplying Data Provenance Attributes...")
        dt_str = h5py.string_dtype(encoding='ascii')
        prov_grid = np.array([m['grid'] for m in timeline], dtype=dt_str)
        prov_space = np.array([m['spacecraft'] for m in timeline], dtype=dt_str)
        prov_time = np.array([m['time'] for m in timeline], dtype='float64')
        prov_idx = np.array([m['local_idx'] for m in timeline], dtype='int32')

        created_harm_dsets = []
        if calc_baseline or "common_mask" not in harm_grp:
            created_harm_dsets.extend([ds_harm_mask, ds_harm_ortho, ds_harm_ndvi])
        if calc_baseline and ds_harm_slide is not None:
            created_harm_dsets.extend([ds_harm_slide, ds_harm_z])
        if calc_3d and ds_harm_3d is not None:
            created_harm_dsets.extend([ds_harm_3d, ds_harm_3d_z])

        for ds in created_harm_dsets:
            ds.attrs.create('source_grid', data=prov_grid)
            ds.attrs.create('source_spacecraft', data=prov_space)
            ds.attrs['acquisition_time'] = prov_time
            ds.attrs['source_frame_index'] = prov_idx

            if ds.name.endswith('sliding_volume_map'):
                ds.attrs['description'] = f"Volume of convex hull within sliding {tile_size}x{tile_size} tile"
                ds.attrs['tile_size'] = tile_size
                ds.attrs['sliding_stride'] = 1
                ds.attrs['gram_type'] = 'minEndmember'
                ds.attrs['num_endmembers'] = num_endmembers
                ds.attrs['Normalization'] = norm_param if norm_param else "None"
                ds.attrs['spatial_smoothing'] = apply_spatial_smoothing
                ds.attrs['metric_type'] = 'spatially_averaged' if apply_spatial_smoothing else 'neighborhood_map_center'
            elif ds.name.endswith('sliding_volume_z_score'):
                ds.attrs['description'] = "Global Spectral Complexity Z-score. ARD Masked pixels excluded from background stats."
                ds.attrs['MASKING_APPLIED'] = True
                ds.attrs['MASK_SOURCE'] = "HARMONIZED_common_mask"
                ds.attrs['spatial_smoothing'] = apply_spatial_smoothing
                ds.attrs['metric_type'] = 'spatially_averaged' if apply_spatial_smoothing else 'neighborhood_map_center'
                ds.attrs['frame_global_means'] = np.array(global_means_list, dtype=np.float32)
                ds.attrs['frame_global_stds'] = np.array(global_stds_list, dtype=np.float32)
            elif ds.name.endswith('volume_map_3d'):
                ds.attrs['description'] = f"3D-normalized equivalent volume (OSDprodnormalized) within sliding {tile_size}x{tile_size} tile"
                ds.attrs['tile_size'] = tile_size
                ds.attrs['sliding_stride'] = 1
                ds.attrs['num_endmembers'] = 20
                ds.attrs['stopping_rule'] = "1% delta reflectance (h_crit = 0.01 * sqrt(B))"
                ds.attrs['separability_threshold'] = 0.01
                ds.attrs['spatial_smoothing'] = apply_spatial_smoothing
                ds.attrs['metric_type'] = 'spatially_averaged' if apply_spatial_smoothing else 'neighborhood_map_center'
            elif ds.name.endswith('volume_map_3d_zscore'):
                ds.attrs['description'] = "Global Z-score of 3D-normalized volume map. ARD Masked pixels excluded from background stats."
                ds.attrs['MASKING_APPLIED'] = True
                ds.attrs['MASK_SOURCE'] = "HARMONIZED_common_mask"
                ds.attrs['spatial_smoothing'] = apply_spatial_smoothing
                ds.attrs['metric_type'] = 'spatially_averaged' if apply_spatial_smoothing else 'neighborhood_map_center'
                ds.attrs['frame_global_means'] = np.array(global_means_3d_list, dtype=np.float32)
                ds.attrs['frame_global_stds'] = np.array(global_stds_3d_list, dtype=np.float32)
            elif ds.name.endswith('ndvi'):
                ds.attrs['description'] = "Harmonized Normalized Difference Vegetation Index (NDVI)" 

        if calc_baseline:
            for grid in grids:
                dsets = sensor_dsets[grid]
                dsets['vol'].attrs['description'] = "Full volume curve (Volume vs Endmember Count) for entire frame"
                dsets['vol'].attrs['gram_type'] = 'minEndmember'
                dsets['vol'].attrs['num_endmembers'] = num_endmembers
                dsets['vol'].attrs['Normalization'] = norm_param if norm_param else "None"

                dsets['em'].attrs['description'] = "Endmembers extracted for frame"
                dsets['em'].attrs['num_endmembers'] = num_endmembers
                dsets['em'].attrs['Normalization'] = norm_param if norm_param else "None"
                dsets['idx'].attrs['description'] = "Spatial 1D indices for extracted endmembers"

        t_end_pipeline = time.perf_counter()
        print(f"\nSpectral complexity in-place processing complete in {(t_end_pipeline - t_start_pipeline)/60:.2f} minutes.")


def main(target_location=None, tile_size=3, num_endmembers=7, norm_param=None, metric="all", apply_spatial_smoothing=APPLY_SPATIAL_SMOOTHING):
    """CLI and top-level entry point."""
    import argparse
    import yaml

    file_path = None
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Calculate Spectral Complexity Metrics")
        parser.add_argument("--file", type=str, help="Path to HLST ARD Master Grid HDF5 Cube")
        parser.add_argument("--location", type=str, help="Target location identifier (e.g. Rochesterv2)")
        parser.add_argument("--tile-size", type=int, default=3, help="Sliding window tile size")
        parser.add_argument("--num-endmembers", type=int, default=7, help="Number of endmembers to extract")
        parser.add_argument("--norm-param", type=str, default=None, help="Normalization parameter (default None)")
        parser.add_argument("--metric", type=str, default="all", choices=["all", "3d", "baseline"], help="Metric to calculate: all, 3d, or baseline")
        parser.add_argument("--spatial-smoothing", dest="spatial_smoothing", action="store_true", default=None, help="Enable legacy 3x3 uniform filter spatial averaging")
        parser.add_argument("--no-spatial-smoothing", dest="spatial_smoothing", action="store_false", help="Bypass spatial averaging and use direct neighborhood_map (default)")
        args, _ = parser.parse_known_args()
        file_path = args.file
        if args.location:
            target_location = args.location
        tile_size = args.tile_size
        num_endmembers = args.num_endmembers
        norm_param = args.norm_param
        if args.metric:
            metric = args.metric
        if args.spatial_smoothing is not None:
            apply_spatial_smoothing = args.spatial_smoothing

    if not file_path:
        if target_location is not None:
            location = target_location
        else:
            script_dir = Path(__file__).resolve().parent
            config_path = os.path.join(script_dir, "locations_config.yaml")
            if not os.path.exists(config_path):
                config_path = os.path.join(script_dir.parent, "locations_config.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                location = config_data.get("current_run", {}).get("location")
            else:
                location = None

        if location:
            file_path = f"C:/satelliteImagery/HLST30/HLST_{location}.h5"

    if not file_path or not os.path.exists(file_path):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            print("Please select the HLST ARD Master Grid HDF5 Cube...")
            file_path = filedialog.askopenfilename(
                title="Select HLST ARD Master Grid HDF5 Cube",
                filetypes=[("HDF5 files", "*.h5")]
            )
            root.destroy()
        except Exception:
            pass

    if file_path and os.path.exists(file_path):
        process_ard_cube(file_path, tile_size=tile_size, num_endmembers=num_endmembers, norm_param=norm_param, metric=metric, apply_spatial_smoothing=apply_spatial_smoothing)
    else:
        print(f"Target file not found or no file selected: {file_path}. Exiting.")


if __name__ == '__main__':
    main()
