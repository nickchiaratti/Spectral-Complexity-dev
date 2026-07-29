"""
export_ortho_visual_ieee.py

Exports high-DPI, publication-ready PNG images of ortho_visual frames from 
HDF5 Harmonized Landsat/Sentinel datasets for IEEE journal formatting.

Specifications:
- Default HDF5: HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5 (or raw Harmonized file)
- Target Single-Column Width: 3.5 inches
- Target DPI: 600 DPI (IEEE High Resolution Standard)
- Frame image only: No axes, no grids, no padding/borders.
"""

import os
import h5py
import numpy as np
from PIL import Image
from datetime import datetime, timezone
import argparse

def export_ieee_ortho_frames(
    sc_h5_path=r"C:\satelliteImagery\HLST30\HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5",
    raw_h5_path=r"C:\satelliteImagery\HLST30\HLST_Rochesterv2_Harmonized.h5",
    output_dir=r"C:\satelliteImagery\HLST30\IEEE_Figures",
    target_date="2025-09-19",
    column_width_inches=3.5,
    target_dpi=600,
    background_mode="transparent" # 'transparent', 'white', or 'black'
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Opening HDF5 dataset: {sc_h5_path}")
    
    with h5py.File(sc_h5_path, 'r') as f_sc, h5py.File(raw_h5_path, 'r') as f_raw:
        ortho_harm = f_sc['HDFEOS/GRIDS/HARMONIZED/Data Fields/ortho_visual']
        times = ortho_harm.attrs['acquisition_time'][:]
        grids = ortho_harm.attrs['source_grid'][:]
        spacecrafts = ortho_harm.attrs['source_spacecraft'][:]
        indices = ortho_harm.attrs['source_frame_index'][:]
        
        grids = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in grids]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in spacecrafts]
        
        target_indices = []
        for idx, (ts, g, sc, src_idx) in enumerate(zip(times, grids, spacecrafts, indices)):
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            if dt.strftime('%Y-%m-%d') == target_date:
                target_indices.append((idx, g, sc, src_idx, dt))
                
        if not target_indices:
            raise ValueError(f"Date {target_date} not found in {sc_h5_path}.")
            
        print(f"Found {len(target_indices)} acquisition(s) on {target_date}:")
        for idx, g, sc, src_idx, dt in target_indices:
            print(f"  - [{g}] {sc} at {dt.strftime('%H:%M:%S UTC')} (Harmonized frame {idx})")

        target_pixel_width = int(round(column_width_inches * target_dpi))

        for idx, g, sc, src_idx, dt in target_indices:
            raw_path = f'HDFEOS/GRIDS/{g}/Data Fields/ortho_visual'
            raw_ortho = f_raw[raw_path][src_idx] # shape (4, H, W)
            
            if raw_ortho.ndim == 3 and raw_ortho.shape[0] in [3, 4]:
                ortho_hwc = np.transpose(raw_ortho, (1, 2, 0))
            else:
                ortho_hwc = raw_ortho.copy()
                
            h, w, c = ortho_hwc.shape
            name = f"{g}_{sc}_{target_date}"
            print(f"\nProcessing {name} (Native resolution: {w}x{h} px)...")

            # Extract RGB + Alpha
            rgb = ortho_hwc[..., :3].astype(np.uint8)
            alpha = ortho_hwc[..., 3].astype(np.uint8) if c == 4 else np.full((h, w), 255, dtype=np.uint8)
            
            # Binary transparency mask (where non-zero alpha)
            alpha = np.where(alpha > 0, 255, 0).astype(np.uint8)

            if background_mode == "transparent":
                rgba = np.dstack([rgb, alpha])
                img = Image.fromarray(rgba, mode='RGBA')
            else:
                bg_color = (255, 255, 255) if background_mode == "white" else (0, 0, 0)
                bg = Image.new("RGB", (w, h), bg_color)
                fg = Image.fromarray(np.dstack([rgb, alpha]), mode='RGBA')
                bg.paste(fg, mask=Image.fromarray(alpha))
                img = bg

            # 1. Native resolution PNG
            native_path = os.path.join(output_dir, f"{name}_native.png")
            img.save(native_path, dpi=(target_dpi, target_dpi), compress_level=6)
            print(f"  [Saved Native] {native_path}")

            # 2. Resampled 600 DPI publication PNG fitting 3.5-inch column
            aspect_ratio = h / w
            target_pixel_height = int(round(target_pixel_width * aspect_ratio))
            
            img_resampled = img.resize((target_pixel_width, target_pixel_height), resample=Image.Resampling.LANCZOS)
            resampled_path = os.path.join(output_dir, f"{name}_3.5in_{target_dpi}dpi.png")
            img_resampled.save(resampled_path, dpi=(target_dpi, target_dpi), compress_level=6)
            
            print(f"  [Saved IEEE 3.5-inch Column Standard] {resampled_path}")
            print(f"    - Dimensions: {target_pixel_width} x {target_pixel_height} px ({column_width_inches} in @ {target_dpi} DPI)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export IEEE-compliant ortho_visual frames from HDF5.")
    parser.add_argument("--sc-h5", type=str, default=r"C:\satelliteImagery\HLST30\HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5", help="Path to SC HDF5 dataset")
    parser.add_argument("--raw-h5", type=str, default=r"C:\satelliteImagery\HLST30\HLST_Rochesterv2_Harmonized.h5", help="Path to Raw HDF5 dataset")
    parser.add_argument("--output-dir", type=str, default=r"C:\satelliteImagery\HLST30\IEEE_Figures", help="Output directory")
    parser.add_argument("--date", type=str, default="2025-09-19", help="Acquisition date (YYYY-MM-DD)")
    parser.add_argument("--column-width", type=float, default=3.5, help="IEEE column width in inches")
    parser.add_argument("--dpi", type=int, default=600, help="Target DPI")
    parser.add_argument("--background", type=str, choices=["transparent", "white", "black"], default="transparent", help="Background handling for nodata pixels")
    
    args = parser.parse_args()
    
    export_ieee_ortho_frames(
        sc_h5_path=args.sc_h5,
        raw_h5_path=args.raw_h5,
        output_dir=args.output_dir,
        target_date=args.date,
        column_width_inches=args.column_width,
        target_dpi=args.dpi,
        background_mode=args.background
    )
