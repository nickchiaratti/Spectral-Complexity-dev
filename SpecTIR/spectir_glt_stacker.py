import os
import sys
import numpy as np
import h5py
from pathlib import Path
from PIL import Image

# Add parent folder to sys.path to find SpecComplex
script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
import SpecComplex as sc
import spectral

OUTPUT_DIR = r"C:\satelliteImagery\share2012\SpecTIR"

def process_spectir_glt(run_id):
    REF_DAT = rf"C:\satelliteImagery\share2012\SpecTIR\{run_id}\{run_id}_pol_ref.dat"
    REF_HDR = rf"C:\satelliteImagery\share2012\SpecTIR\{run_id}\{run_id}_pol_ref.hdr"
    LOG_FILE = rf"C:\satelliteImagery\share2012\SpecTIR\{run_id}\{run_id}_pol_ref.log"

    GLT_DAT = rf"C:\satelliteImagery\share2012\SpecTIR\IGM_GLTS_AVON_AM\IGM_GLTS\{run_id}_rad_glt.dat"
    GLT_HDR = rf"C:\satelliteImagery\share2012\SpecTIR\IGM_GLTS_AVON_AM\IGM_GLTS\{run_id}_rad_glt.hdr"

    OUTPUT_H5 = os.path.join(OUTPUT_DIR, f"SpecTIR_Ortho_Stack_{run_id}.h5")

    if not os.path.exists(REF_DAT) or not os.path.exists(GLT_DAT):
        print(f"Skipping {run_id}: Missing REF or GLT data files.")
        return

    print(f"\n--- Processing run: {run_id} ---")
    print(f"Loading metadata from ENVI headers...")
    ref_img = spectral.envi.open(REF_HDR, REF_DAT)
    glt_img = spectral.envi.open(GLT_HDR, GLT_DAT)
    
    lines, samples, bands = ref_img.shape
    map_lines, map_samples, glt_bands = glt_img.shape
    
    print(f"Ref Image: {lines}L x {samples}S x {bands}B (dtype: {ref_img.dtype})")
    print(f"GLT Image: {map_lines}L x {map_samples}S x {glt_bands}B")
    
    # Parse GeoTransform and Spatial Ref from GLT Map Info
    map_info = glt_img.metadata.get('map info', [])
    gdal_transform = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    crs_wkt = ""
    if len(map_info) >= 11:
        # e.g. ['UTM', '1.000', '1.000', '273299.802', '4758275.533', '1.000', '1.000', '18', 'North', 'WGS-84', 'units=Meters']
        x_tie = float(map_info[3])
        y_tie = float(map_info[4])
        pixel_x = float(map_info[5])
        pixel_y = float(map_info[6])
        # GLT Map Info assumes tie point is at (1.000, 1.000)
        # We need to construct a GDAL GeoTransform (0-based)
        gdal_transform = [x_tie - pixel_x/2.0, pixel_x, 0.0, y_tie + pixel_y/2.0, 0.0, -pixel_y]
        
        # Build a basic CRS WKT if possible
        zone = map_info[7]
        hemisphere = map_info[8]
        crs_wkt = f"PROJCS[\"WGS 84 / UTM zone {zone}{'N' if hemisphere.lower() == 'north' else 'S'}\"]"
    
    wavelengths = np.zeros(bands, dtype='float32')
    if 'wavelength' in ref_img.metadata:
        wavelengths = np.array([float(w) for w in ref_img.metadata['wavelength']], dtype='float32')
        # Convert to nanometers if in micrometers
        if np.max(wavelengths) < 10.0:
            wavelengths = wavelengths * 1000.0
        
    fill_value = 0 # Default fill value if not found
    if 'data ignore value' in ref_img.metadata:
        fill_value = float(ref_img.metadata['data ignore value'])
        if ref_img.dtype.char in ['h', 'i', 'b', 'H', 'I', 'B']:
            fill_value = int(fill_value)
            
    print(f"Using Fill Value: {fill_value}")
    
    acq_time = 0.0
    # Attempt to extract time from log
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            for line in f:
                if "Time:" in line:
                    # Simple heuristic, full parsing would require more logic
                    pass
    
    # Create HDF5
    print(f"Creating HDF5 Output at {OUTPUT_H5}...")
    with h5py.File(OUTPUT_H5, 'w') as out_h5:
        grp = out_h5.create_group("HDFEOS/GRIDS/SPECTIR/Data Fields")
        
        chunk_h, chunk_w = min(map_lines, 256), min(map_samples, 256)
        
        out_shape = (1, bands, map_lines, map_samples)
        chunks_dim = (1, bands, chunk_h, chunk_w)
        
        # Create dataset
        ds_ref = grp.create_dataset(
            "surface_reflectance", 
            shape=out_shape, 
            dtype=ref_img.dtype, 
            compression="gzip", 
            compression_opts=5, 
            shuffle=True, 
            fillvalue=fill_value, 
            chunks=chunks_dim
        )
        
        ds_vis = grp.create_dataset(
            "ortho_visual", 
            shape=(1, 4, map_lines, map_samples), 
            dtype='uint8', 
            compression="gzip", 
            shuffle=True, 
            fillvalue=0
        )
        
        ds_mask = grp.create_dataset(
            "nodata_mask",
            shape=(1, 1, map_lines, map_samples),
            dtype='bool',
            compression="gzip",
            compression_opts=5,
            shuffle=True,
            fillvalue=True,
            chunks=(1, 1, chunk_h, chunk_w)
        )
        
        print(f"Loading entire RAW image into memory for fast lookup...")
        raw_data = ref_img.load()
        
        print("Iterating over GLT map space in chunks to orthorectify...")
        
        # For ortho_visual composite array accumulation
        # Because storing 360 bands for 4949x4020 takes ~14GB, 
        # instead of a full vis_buffer, we can accumulate just the RGB bands!
        # Let's find RGB bands if wavelengths are present
        vis_buffer = None
        vis_bands = [0, 0, 0] # default indices
        
        if len(wavelengths) > 0:
            # SpecComplex usually finds closest to 650, 550, 450
            # Let's see how sc.generate_rgba_from_hsi works
            # Actually, `generate_rgba_from_hsi` takes the *entire* HSI frame (H, W, bands)
            # This would require 14GB buffer. We should extract just the 3 bands.
            vis_bands[0] = np.argmin(np.abs(wavelengths - 650))
            vis_bands[1] = np.argmin(np.abs(wavelengths - 550))
            vis_bands[2] = np.argmin(np.abs(wavelengths - 450))
        
        print(f"Extracted RGB bands for vis buffer: {vis_bands}")
        vis_buffer_rgb = np.full((3, map_lines, map_samples), fill_value, dtype=ref_img.dtype)
        
        chunk_size = 256
        for start_row in range(0, map_lines, chunk_size):
            end_row = min(start_row + chunk_size, map_lines)
            chunk_h_actual = end_row - start_row
            
            glt_subset = glt_img.read_subregion((start_row, end_row), (0, map_samples))
            b1 = glt_subset[:, :, 0] # Sample / X
            b2 = glt_subset[:, :, 1] # Line / Y
            
            # The dataset is used as is, taking absolute values for negative (nearest neighbor interpolated) coordinates
            # Zero means no data / background in GLT
            valid = (b1 != 0) & (b2 != 0)
            
            # Save the nodata mask
            ds_mask[0, 0, start_row:end_row, :] = ~valid
            
            sample_idx = np.abs(b1[valid]) - 1
            line_idx = np.abs(b2[valid]) - 1
            
            # Enforce bounds just in case GLT is out of range
            sample_idx = np.clip(sample_idx, 0, samples - 1)
            line_idx = np.clip(line_idx, 0, lines - 1)
            
            # Create incoming buffer for this row chunk
            incoming = np.full((bands, chunk_h_actual, map_samples), fill_value, dtype=ref_img.dtype)
            
            if np.any(valid):
                # Lookup data from raw image
                # raw_data[line_idx, sample_idx] has shape (N_valid, bands)
                lookup_vals = raw_data[line_idx, sample_idx, :]
                
                # Transpose to (bands, N_valid)
                lookup_vals = lookup_vals.T
                
                # Assign to incoming array at valid locations
                for b in range(bands):
                    incoming[b, valid] = lookup_vals[b, :]
                    
            # Write this chunk to HDF5 dataset
            ds_ref[0, :, start_row:end_row, :] = incoming
            
            # Save to rgb visual buffer for ortho_visual generation later
            vis_buffer_rgb[0, start_row:end_row, :] = incoming[vis_bands[0], :, :]
            vis_buffer_rgb[1, start_row:end_row, :] = incoming[vis_bands[1], :, :]
            vis_buffer_rgb[2, start_row:end_row, :] = incoming[vis_bands[2], :, :]
            
            print(f"  Processed rows {start_row} to {end_row} ({end_row/map_lines*100:.1f}%)")

        print("Generating ortho_visual RGB composite...")
        # dummy_wavelengths matching the 3 extracted bands
        dummy_wavelengths = np.array([650.0, 550.0, 450.0])
        # frame_data must be (Bands, H, W) as expected by generate_rgba_from_hsi
        rgba_img = sc.generate_rgba_from_hsi(frame_data=vis_buffer_rgb, wavelengths=dummy_wavelengths, nodata=fill_value)
        ds_vis[0, ...] = np.transpose(rgba_img, (2, 0, 1))

        # Attributes
        dt_str = h5py.string_dtype(encoding='ascii')
        ds_ref.attrs['acquisition_time'] = np.array([acq_time], dtype='float64')
        ds_ref.attrs.create('spacecraft_id', data=np.array(['SpecTIR'], dtype=dt_str))
        ds_ref.attrs['wavelengths'] = wavelengths
        
        for name in grp.keys():
            grp[name].attrs['spatial_ref'] = crs_wkt
            grp[name].attrs['GeoTransform'] = np.array(gdal_transform, dtype='float64')

    print("Exporting high-resolution ortho_visual preview (600 DPI)...")
    png_name = f"SpecTIR_{run_id}_ortho_visual_preview.png"
    png_path = os.path.join(OUTPUT_DIR, png_name)
    img = Image.fromarray(rgba_img, 'RGBA')
    img.save(png_path, dpi=(600, 600))
    print(f"Saved: {png_path}")
            
    print(f"Synthesis Complete. Stack Stored at: {OUTPUT_H5}")

if __name__ == "__main__":
    runs_to_process = [
        "0920-1631",
        "0920-1638",
        "0920-1646",
        "0920-1654",
        "0920-1706"
    ]
    for run in runs_to_process:
        process_spectir_glt(run)
