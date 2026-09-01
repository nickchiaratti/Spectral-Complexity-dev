import os
import sys
import json
import datetime
import h5py
import numpy as np
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

# Add project root to sys.path
LOCATION='Rochesterv2'
sys.path.insert(0, r"f:\Resilio\IMGS 890 Research\Spectral-Complexity-dev")
import SpecComplex as sc

def reproduce_pngs(location="SantaBarabara"):
    h5_path = rf"C:\satelliteImagery\enmap\EnMAP_MGRS_Stack_{location}.h5"
    out_dir = rf"C:\satelliteImagery\Enmap\{location}_SourceData"
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Cannot find {h5_path}")
        
    print(f"Opening {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        sr_ds = f['HDFEOS/GRIDS/ENMAP/Data Fields/surface_reflectance']
        mask_ds = f['HDFEOS/GRIDS/ENMAP/Data Fields/common_mask']
        meta_grp = f['METADATA']
        
        n_times, n_bands, height, width = sr_ds.shape
        print(f"Found {n_times} passes on grid ({height}x{width}, {n_bands} bands).")
        
        for t_idx in tqdm(range(n_times), desc="Generating RGB PNGs"):
            # 1. Parse JSON metadata for timestamps and exact wavelengths
            stac_dict = json.loads(meta_grp.attrs[f"frame_{t_idx}_json"])
            dt_str = stac_dict['properties']['datetime']
            acq_dt = datetime.datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            pass_ts_safe = acq_dt.strftime("%Y%m%dT%H%M%S")
            scene_id = stac_dict.get('id', '')
            
            # Extract wavelengths from STAC JSON if present
            eo_bands = stac_dict.get('assets', {}).get('image', {}).get('eo:bands', [])
            wavelengths = []
            if eo_bands:
                wavelengths = [b.get('eo:center_wavelength') if 'eo:center_wavelength' in b else b.get('center_wavelength', 0.0) for b in eo_bands]
            
            # If wavelengths are missing from STAC JSON, locate and parse METADATA.XML
            if not wavelengths or not any(w > 0 for w in wavelengths):
                xml_matches = glob.glob(os.path.join(out_dir, f"*{scene_id}*", "*-METADATA.XML")) + \
                              glob.glob(os.path.join(out_dir, f"{scene_id}-METADATA.XML"))
                if xml_matches and os.path.exists(xml_matches[0]):
                    try:
                        import xml.etree.ElementTree as ET
                        tree = ET.parse(xml_matches[0])
                        root = tree.getroot()
                        xml_wvs = [float(e.text) for e in root.iter() if e.tag.lower().endswith('wavelengthcenterofband') and e.text]
                        if len(xml_wvs) == n_bands:
                            wavelengths = xml_wvs
                    except Exception as e:
                        print(f"Warning: Failed to parse XML for frame {t_idx}: {e}")
            
            # Fallback to dataset attribute if valid
            if not wavelengths or not any(w > 0 for w in wavelengths):
                ds_wvs = sr_ds.attrs.get('wavelengths')
                if ds_wvs is not None and any(w > 0 for w in ds_wvs):
                    wavelengths = ds_wvs
                    
            if not wavelengths or not any(w > 0 for w in wavelengths):
                raise ValueError(f"Frame {t_idx} ({scene_id}): No valid center wavelengths could be resolved.")
                
            frame_data = sr_ds[t_idx, ...]
            nodata_val = -32768
            
            # 2. Generate RGBA true color composite
            rgba_img = sc.generate_rgba_from_hsi(
                frame_data=frame_data,
                wavelengths=wavelengths,
                nodata=nodata_val,
                scale=sr_ds.attrs.get('scale_to_float', sr_ds.attrs.get('scale_factor', 0.0001))
            )
            
            # 3. Save clean RGB PNG
            png_filename = f"EnMAP_{location}_{pass_ts_safe}_RGB.png"
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
            
            masked_png_name = f"EnMAP_{location}_{pass_ts_safe}_RGB_masked.png"
            masked_png_path = os.path.join(out_dir, masked_png_name)
            masked_img.save(masked_png_path)
            
    print(f"\nSuccessfully regenerated all {n_times} RGB and RGB_masked PNG previews in {out_dir}.")

if __name__ == "__main__":
    reproduce_pngs(LOCATION)
