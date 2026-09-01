import os
import sys
import datetime
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

LOCATION = 'Rochesterv2'
# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(script_dir))
import SpecComplex as sc

def reproduce_pngs(location="Rochesterv2"):
    h5_path = rf"C:\satelliteImagery\dragonette\Dragonette_MGRS_Stack_{location}.h5"
    out_dir = rf"C:\satelliteImagery\dragonette\{location}_SourceData"
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Cannot find {h5_path}")
        
    print(f"Opening {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        sr_ds = f['HDFEOS/GRIDS/DRAGONETTE/Data Fields/surface_reflectance']
        mask_ds = f['HDFEOS/GRIDS/DRAGONETTE/Data Fields/common_mask']
        
        n_times, n_bands, height, width = sr_ds.shape
        print(f"Found {n_times} passes on grid ({height}x{width}, {n_bands} bands).")
        
        acq_time_array = sr_ds.attrs.get('acquisition_time')
        wavelengths = sr_ds.attrs.get('wavelengths')
        
        if wavelengths is None or not any(w > 0 for w in wavelengths):
            raise ValueError(f"No valid center wavelengths could be resolved.")
            
        for t_idx in tqdm(range(n_times), desc="Generating RGB PNGs"):
            # 1. Parse timestamps
            if acq_time_array is not None:
                acq_dt = datetime.datetime.fromtimestamp(acq_time_array[t_idx], tz=datetime.timezone.utc)
                pass_ts_safe = acq_dt.strftime("%Y%m%dT%H%M%S")
            else:
                pass_ts_safe = f"pass_{t_idx:03d}"
                
            frame_data = sr_ds[t_idx, ...]
            # Dragonette stacker explicitly sets fillvalue to -9999
            nodata_val = -9999
            
            # 2. Generate RGBA true color composite
            r_idx = np.argmin(np.abs(wavelengths - 650.0))
            g_idx = np.argmin(np.abs(wavelengths - 550.0))
            b_idx = np.argmin(np.abs(wavelengths - 470.0))
            
            rgba_img = sc.generate_rgba_image(
                r_band=frame_data[r_idx, :, :],
                g_band=frame_data[g_idx, :, :],
                b_band=frame_data[b_idx, :, :],
                nodata=nodata_val
            )
            
            # 3. Save clean RGB PNG
            png_filename = f"Dragonette_{location}_{pass_ts_safe}_RGB.png"
            png_path = os.path.join(out_dir, png_filename)
            img = Image.fromarray(rgba_img, 'RGBA')
            img.save(png_path)
            
            # 4. Save Masked RGB PNG
            frame_mask = mask_ds[t_idx, ...]
            overlay = Image.new('RGBA', img.size, (255, 166, 0, 0))
            overlay_data = np.array(overlay)
            overlay_data[frame_mask == True] = [255, 166, 0, 128]
            
            overlay_img = Image.fromarray(overlay_data, 'RGBA')
            masked_img = Image.alpha_composite(img, overlay_img)
            
            masked_png_filename = f"Dragonette_{location}_{pass_ts_safe}_RGB_masked.png"
            masked_png_path = os.path.join(out_dir, masked_png_filename)
            masked_img.save(masked_png_path)
            
    print(f"\nSuccessfully regenerated all {n_times} RGB and RGB_masked PNG previews in {out_dir}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--location", type=str, default=LOCATION, help="Target location prefix")
    args = parser.parse_args()
    reproduce_pngs(args.location)
