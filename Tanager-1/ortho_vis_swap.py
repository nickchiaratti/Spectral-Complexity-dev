import h5py
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

# Target wavelengths for True Color composite (in nanometers)
TARGET_RED_NM = 670.0
TARGET_GREEN_NM = 550.0
TARGET_BLUE_NM = 480.0

def percentile_stretch(band_data, fill_value=-9999.0, lower_pct=1, upper_pct=99):
    """
    Applies a robust linear contrast stretch based on data percentiles.
    Reference: Richards, J. A. (2013). Remote Sensing Digital Image Analysis.
    Ignores background fill values and negative atmospheric correction artifacts.
    """
    valid_mask = (band_data != fill_value) & (band_data >= 0)
    
    if not np.any(valid_mask):
        return np.zeros_like(band_data, dtype=np.uint8)
    
    valid_data = band_data[valid_mask]
    p_low, p_high = np.percentile(valid_data, (lower_pct, upper_pct))
    
    if p_high == p_low:
        return np.zeros_like(band_data, dtype=np.uint8)
        
    stretched = (band_data.astype(np.float32) - p_low) / (p_high - p_low)
    stretched = np.clip(stretched, 0, 1) * 255
    
    result = np.zeros_like(band_data, dtype=np.uint8)
    result[valid_mask] = stretched[valid_mask].astype(np.uint8)
    return result

def parse_odl_grid_params(odl_str):
    """Parses existing physical grid dimensions from the HDF-EOS ODL string."""
    ul_match = re.search(r'UpperLeftPointMtrs=\(\s*(-?[\d\.]+)\s*,\s*(-?[\d\.]+)\s*\)', odl_str)
    lr_match = re.search(r'LowerRightMtrs=\(\s*(-?[\d\.]+)\s*,\s*(-?[\d\.]+)\s*\)', odl_str)
    x_match = re.search(r'XDim=(\d+)', odl_str)
    y_match = re.search(r'YDim=(\d+)', odl_str)
    zone_match = re.search(r'ZoneCode=(\d+)', odl_str)
    
    ul_mtrs = (float(ul_match.group(1)), float(ul_match.group(2)))
    lr_mtrs = (float(lr_match.group(1)), float(lr_match.group(2)))
    width = int(x_match.group(1))
    height = int(y_match.group(1))
    zone = int(zone_match.group(1)) if zone_match else 18
    
    return width, height, ul_mtrs, lr_mtrs, zone

def generate_struct_metadata(grid_name, width, height, ul_mtrs, lr_mtrs, datasets_info, n_times, n_bands, utm_zone):
    """Regenerates HDF-EOS5 StructMetadata.0 ODL string ensuring standard compliance."""
    data_fields_blocks = []
    for i, (name, dtype, rank, dim_names) in enumerate(datasets_info):
        eos_type = "H5T_NATIVE_FLOAT"
        dtype_str = str(dtype)
        if "int8" in dtype_str: eos_type = "H5T_NATIVE_INT8"
        elif "uint8" in dtype_str: eos_type = "H5T_NATIVE_UINT8"
        elif "int16" in dtype_str: eos_type = "H5T_NATIVE_INT16"
        elif "uint16" in dtype_str: eos_type = "H5T_NATIVE_UINT16"
        elif "int32" in dtype_str: eos_type = "H5T_NATIVE_INT32"
        elif "uint32" in dtype_str: eos_type = "H5T_NATIVE_UINT32"
        elif "float64" in dtype_str or "double" in dtype_str: eos_type = "H5T_NATIVE_DOUBLE"
        
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
    
    odl = f"""GROUP=SwathStructure
END_GROUP=SwathStructure
GROUP=GridStructure
    GROUP=GRID_1
        GridName="{grid_name}"
        XDim={width}
        YDim={height}
        UpperLeftPointMtrs=({ul_mtrs[0]:.6f},{ul_mtrs[1]:.6f})
        LowerRightMtrs=({lr_mtrs[0]:.6f},{lr_mtrs[1]:.6f})
        Projection=HE5_GCTP_UTM
        ZoneCode={utm_zone}
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
                DimensionName="VisBand"
                Size=4
            END_OBJECT=Dimension_5
            OBJECT=Dimension_6
                DimensionName="RGBBand"
                Size=3
            END_OBJECT=Dimension_6
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

def retrofit_file(filepath):
    print(f"Opening HDF5 file for retrofitting: {filepath}")
    
    with h5py.File(filepath, 'r+') as h5:
        # Check standard grid names (TANAGER or HYP)
        if "HDFEOS/GRIDS/TANAGER/Data Fields" in h5:
            grid_name = "TANAGER"
        elif "HDFEOS/GRIDS/HYP/Data Fields" in h5:
            grid_name = "HYP"
        else:
            raise KeyError("Standard HDFEOS grid group (TANAGER or HYP) not found.")
            
        base_path = f"HDFEOS/GRIDS/{grid_name}/Data Fields"
        data_grp = h5[base_path]
        
        if "surface_reflectance" not in data_grp:
            raise KeyError("'surface_reflectance' dataset is required but missing.")
            
        sr_dset = data_grp["surface_reflectance"]
        n_times, n_bands, height, width = sr_dset.shape
        
        # 1. Rename existing 'ortho_visual' to 'rgb_vis'
        if "ortho_visual" in data_grp:
            current_vis = data_grp["ortho_visual"]
            if current_vis.shape[1] == 4:
                print("  Found existing 4-band 'ortho_visual'. Renaming to 'rgb_vis' via hardlink...")
                if "rgb_vis" in data_grp:
                    del data_grp["rgb_vis"]
                # Create hard link, then delete old reference (Zero data copy)
                data_grp["rgb_vis"] = current_vis
                del data_grp["ortho_visual"]
            elif current_vis.shape[1] == 3:
                print("  'ortho_visual' is already a 3-band composite. Re-calculating to ensure accuracy...")
                del data_grp["ortho_visual"]
        
        # 2. Extract SR Fill Value
        sr_fill = -9999.0
        if "_FillValue" in sr_dset.attrs:
            fv = sr_dset.attrs["_FillValue"]
            sr_fill = fv[0] if isinstance(fv, (np.ndarray, list, tuple)) else fv

        # 3. Determine precise RGB band indices
        if "wavelengths" not in sr_dset.attrs:
            raise AttributeError("Missing 'wavelengths' attribute in surface_reflectance.")
            
        wavelengths = sr_dset.attrs['wavelengths']
        r_idx = int(np.argmin(np.abs(wavelengths - TARGET_RED_NM)))
        g_idx = int(np.argmin(np.abs(wavelengths - TARGET_GREEN_NM)))
        b_idx = int(np.argmin(np.abs(wavelengths - TARGET_BLUE_NM)))
        print(f"  Mapped Target Wavelengths to Indices -> R:{r_idx}, G:{g_idx}, B:{b_idx}")

        # 4. Generate the new ortho_visual RGB dataset
        print("  Calculating new 3-band 'ortho_visual' from surface_reflectance...")
        ortho_vis_dset = data_grp.create_dataset(
            "ortho_visual", shape=(n_times, 3, height, width), 
            dtype='uint8', compression="gzip", fillvalue=0
        )
        
        # Copy spatial geotransform attributes if available
        if 'spatial_ref' in sr_dset.attrs: ortho_vis_dset.attrs['spatial_ref'] = sr_dset.attrs['spatial_ref']
        if 'GeoTransform' in sr_dset.attrs: ortho_vis_dset.attrs['GeoTransform'] = sr_dset.attrs['GeoTransform']
        
        for t_idx in tqdm(range(n_times), desc="  Processing Frames"):
            # Load specific bands to memory
            r_band = sr_dset[t_idx, r_idx, :, :]
            g_band = sr_dset[t_idx, g_idx, :, :]
            b_band = sr_dset[t_idx, b_idx, :, :]
            
            ortho_vis_dset[t_idx, 0, :, :] = percentile_stretch(r_band, sr_fill)
            ortho_vis_dset[t_idx, 1, :, :] = percentile_stretch(g_band, sr_fill)
            ortho_vis_dset[t_idx, 2, :, :] = percentile_stretch(b_band, sr_fill)
            
        # 5. Regenerate the HDFEOS StructMetadata.0 ODL
        print("  Re-building HDF-EOS5 StructMetadata.0 ODL string...")
        info_grp = h5["HDFEOS INFORMATION"]
        old_odl_bytes = info_grp["StructMetadata.0"][()]
        old_odl = old_odl_bytes[0].decode('ascii') if isinstance(old_odl_bytes, np.ndarray) else old_odl_bytes.decode('ascii')
        
        # Parse physical dimensions from old ODL
        try:
            width, height, ul_mtrs, lr_mtrs, utm_zone = parse_odl_grid_params(old_odl)
        except Exception as e:
            raise RuntimeError(f"Failed to parse geospatial grid parameters from existing ODL: {e}")

        # Build comprehensive dataset list mapping Dimensions based on shape
        datasets_info = []
        for name, dset in data_grp.items():
            shape = dset.shape
            if len(shape) == 4:
                if name == "rgb_vis" or (name == "ortho_visual" and shape[1] == 4):
                    dim_names = ["Time", "VisBand", "YDim", "XDim"]
                elif name == "ortho_visual" and shape[1] == 3:
                    dim_names = ["Time", "RGBBand", "YDim", "XDim"]
                else:
                    dim_names = ["Time", "Band", "YDim", "XDim"]
            elif len(shape) == 3:
                dim_names = ["Time", "YDim", "XDim"]
            else:
                dim_names = [f"Dim_{i}" for i in range(len(shape))]
            datasets_info.append((name, dset.dtype, len(shape), dim_names))

        # Re-write the updated ODL to the file
        new_odl = generate_struct_metadata(
            grid_name, width, height, ul_mtrs, lr_mtrs, 
            datasets_info, n_times, n_bands, utm_zone
        )
        
        # Update dataset payload
        dt_str = h5py.string_dtype(encoding='ascii')
        del info_grp["StructMetadata.0"]
        info_grp.create_dataset("StructMetadata.0", (1,), dtype=dt_str, data=new_odl)
        
        print("Retrofit completed successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    print("Select the processed Tanager HDF5 file(s) to retrofit...")
    file_paths = filedialog.askopenfilenames(
        title="Select Tanager HDF5 Stacks",
        filetypes=[("HDF5 files", "*.h5")]
    )
    
    for path in file_paths:
        retrofit_file(path)
        
    root.destroy()