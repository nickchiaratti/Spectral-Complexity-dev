import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from itertools import groupby
from operator import itemgetter
import re
from datetime import datetime


# Add the parent directory to sys.path to import SpecComplex
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from SpecComplex import generate_rgba_image


def find_nearest_band_idx(wavelengths, target_wl):
    return (np.abs(wavelengths - target_wl)).argmin()


def main():
    # Hardcoded paths as requested by the user
    sr_file = r"C:\satelliteImagery\Tanager\Rochesterv2_SourceData\20250704_165208_78_4001\20250704_165208_78_4001_basic_sr_hdf5.h5"
    sr_6s_file = r"C:\satelliteImagery\Tanager\Rochesterv2_SourceData\20250704_165208_78_4001\20250704_165208_78_4001_basic_6Ssr_hdf5.h5"

    if not os.path.exists(sr_file):
        print(f"Error: Original SR file not found: {sr_file}")
        return
    if not os.path.exists(sr_6s_file):
        print(f"Error: 6S SR file not found: {sr_6s_file}")
        return

    # Open both HDF5 files
    try:
        f_orig = h5py.File(sr_file, 'r')
    except OSError as e:
        print(f"Error: Could not read original SR file ({sr_file}). It may be corrupted or not a valid HDF5 file.")
        print(f"Details: {e}")
        return

    try:
        f_6s = h5py.File(sr_6s_file, 'r')
    except OSError as e:
        print(f"Error: Could not read 6S SR file ({sr_6s_file}). It may be corrupted. You might need to re-run tanager_py6s_atmospheric_correction.py.")
        print(f"Details: {e}")
        f_orig.close()
        return

    data_path = 'HDFEOS/SWATHS/HYP/Data Fields/surface_reflectance'

    if data_path not in f_orig or data_path not in f_6s:
        print(f"Error: {data_path} not found in one of the HDF5 files.")
        return

    ds_orig = f_orig[data_path]
    ds_6s = f_6s[data_path]

    # Extract attributes
    wavelengths = ds_orig.attrs['wavelengths']
    wl_units_raw = ds_orig.attrs.get('wavelengths_units', b'nm')
    wl_units = wl_units_raw.decode('utf-8') if isinstance(wl_units_raw, bytes) else str(wl_units_raw)
    
    if 'nm' in wl_units.lower():
        wl_nm = wavelengths
    else:
        wl_nm = wavelengths * 1000.0  # Convert to nm for standard indexing

    good_wavelengths = ds_orig.attrs['good_wavelengths']
    fill_val = ds_orig.attrs.get('_FillValue', -9999.0)

    # Find Red, Green, Blue bands for generating RGBA image
    # Assuming ~640nm (Red), ~550nm (Green), ~470nm (Blue)
    r_idx = find_nearest_band_idx(wl_nm, 640.0)
    g_idx = find_nearest_band_idx(wl_nm, 550.0)
    b_idx = find_nearest_band_idx(wl_nm, 470.0)

    print(f"Extracting RGB bands for visualization:")
    print(f"  Red: Band {r_idx} ({wl_nm[r_idx]:.1f} nm)")
    print(f"  Green: Band {g_idx} ({wl_nm[g_idx]:.1f} nm)")
    print(f"  Blue: Band {b_idx} ({wl_nm[b_idx]:.1f} nm)")

    r_band = ds_orig[r_idx, :, :].astype(np.float32)
    g_band = ds_orig[g_idx, :, :].astype(np.float32)
    b_band = ds_orig[b_idx, :, :].astype(np.float32)

    # Mask fill values with NaN
    r_band[r_band == fill_val] = np.nan
    g_band[g_band == fill_val] = np.nan
    b_band[b_band == fill_val] = np.nan

    print("Generating RGBA image...")
    rgba_img = generate_rgba_image(r_band, g_band, b_band)

    # Setup the plot
    fig, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(14, 6))

    ax_img.imshow(rgba_img)
    
    # Extract date/time from filename for title
    filename_base = os.path.basename(sr_file)
    match = re.search(r'(\d{8})_(\d{6})', filename_base)
    if match:
        dt_str = match.group(1) + match.group(2)
        dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        ax_img.set_title(f"Original Surface Reflectance (RGB)\nAcquired: {time_str}")
    else:
        ax_img.set_title("Original Surface Reflectance (RGB)")
        
    ax_img.set_xlabel("Pixel X")
    ax_img.set_ylabel("Pixel Y")

    ax_spec.set_title("Spectral Profile")
    ax_spec.set_xlabel(f"Wavelength ({wl_units})")
    ax_spec.set_ylabel("Reflectance")
    ax_spec.grid(True)
    ax_spec.text(0.5, 0.5, "Click on the image to select a pixel", 
                 ha='center', va='center', transform=ax_spec.transAxes, 
                 fontsize=12, color='gray')

    selected_point = None
    current_x = None
    current_y = None

    # Handle clicks
    def onclick(event):
        nonlocal selected_point, current_x, current_y
        if event.inaxes != ax_img:
            return

        # Ensure valid pixel coordinates
        x, y = int(round(event.xdata)), int(round(event.ydata))
        
        # Check bounds
        height, width = r_band.shape
        if not (0 <= x < width and 0 <= y < height):
            return

        current_x, current_y = x, y
        print(f"Pixel selected: ({x}, {y})")

        # Read the entire spectrum for the selected pixel
        spec_orig = ds_orig[:, y, x].astype(np.float32)
        spec_6s = ds_6s[:, y, x].astype(np.float32)

        # Apply NaN masking where fill values exist
        spec_orig[spec_orig == fill_val] = np.nan
        spec_6s[spec_6s == fill_val] = np.nan

        ax_spec.clear()

        # Overlay gray bars for bad wavelengths
        # 'good_wavelengths' != 1 signifies bad bands
        bad_idx = np.where(good_wavelengths != 1)[0]
        if len(bad_idx) > 0:
            # Group contiguous bad indices to draw fewer span boxes
            for k, g in groupby(enumerate(bad_idx), lambda ix: ix[0] - ix[1]):
                group = list(map(itemgetter(1), g))
                start_idx = group[0]
                end_idx = group[-1]
                
                # Approximate start and end wavelengths for the bar span
                # If the first band is bad, we just use its wavelength as start
                # Use midway point between neighboring good and bad bands for the span
                start_wl = wavelengths[start_idx]
                end_wl = wavelengths[end_idx]
                
                if start_idx > 0:
                    start_wl = (wavelengths[start_idx] + wavelengths[start_idx - 1]) / 2.0
                if end_idx < len(wavelengths) - 1:
                    end_wl = (wavelengths[end_idx] + wavelengths[end_idx + 1]) / 2.0
                elif start_idx == end_idx:
                    # Single band case at the end
                    start_wl -= 5
                    end_wl += 5

                # Plot the span (legend duplication is handled below)
                ax_spec.axvspan(start_wl, end_wl, color='gray', alpha=0.3, label='Tanager Bad Wavelengths')

        # Plot the spectra
        ax_spec.plot(wavelengths, spec_orig, label='Original SR', color='blue')
        ax_spec.plot(wavelengths, spec_6s, label='Py6S SR', color='red')

        ax_spec.set_title(f"Spectral Profile at ({x}, {y})")
        ax_spec.set_xlabel(f"Wavelength ({wl_units})")
        ax_spec.set_ylabel("Reflectance")
        
        # Deduplicate legend labels
        handles, labels = ax_spec.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_spec.legend(by_label.values(), by_label.keys())
        
        ax_spec.grid(True)

        # Update the crosshair on the image plot
        if selected_point is not None:
            selected_point.remove()

        selected_point = ax_img.scatter(x, y, color='red', marker='+', s=150, linewidths=2)

        fig.canvas.draw_idle()

    # Connect the click event
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    # Add Save Button
    ax_btn = plt.axes([0.85, 0.02, 0.1, 0.06])
    btn_save = Button(ax_btn, 'Save Image')

    def save_image(event):
        if current_x is None or current_y is None:
            print("Please click on a pixel first.")
            return

        save_dir = r"C:\satelliteImagery\Tanager\Rochesterv2_SourceData\20250704_165208_78_4001"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"pixel_{current_x}_{current_y}.png")
        
        # Temporarily hide button during save
        ax_btn.set_visible(False)
        fig.savefig(filename, dpi=300)
        ax_btn.set_visible(True)
        
        print(f"Saved image to: {filename}")

    btn_save.on_clicked(save_image)

    print("Opening interactive plot...")
    plt.show()

if __name__ == "__main__":
    main()
