import os
import sys
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.interpolate import interp1d

def main():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    hdr_path = filedialog.askopenfilename(
        title="Select Raw HDR File",
        filetypes=[("ENVI Header", "*.hdr"), ("All files", "*.*")],
        initialdir=r"F:\GlobusData\hyspex\20250918_GE18\dual"
    )
    if not hdr_path:
        return
        
    target_files = filedialog.askopenfilenames(
        title="Select Ground Truth Panel files (.txt) for Calibration",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        initialdir=r"C:\satelliteImagery\hyspex\calibration_targets"
    )
    if not target_files:
        print("No targets selected. Exiting.")
        return

    run_id = os.path.basename(hdr_path).split('_Mjolnir')[0]
    output_dir = os.path.dirname(target_files[0])
    
    # 1. Open image
    img = envi.open(hdr_path)
    # Determine expected image file based on header file name
    base_name = os.path.splitext(hdr_path)[0]
    expected_img = base_name + ".bsq" # Common for HySpex/PARGE
    
    if not os.path.exists(expected_img):
        for ext in ['', '.img', '.dat', '.raw']:
            if os.path.exists(base_name + ext):
                expected_img = base_name + ext
                break

    img_mm = img.open_memmap(interleave='bip')
    lines, samples, bands = img_mm.shape
    
    scale_val = 1.0
    if 'reflectance scale factor' in img.metadata:
        try:
            val = img.metadata['reflectance scale factor']
            scale_val = float(val[0]) if isinstance(val, list) else float(val)
        except Exception:
            pass

    fill_value = -1.0
    if 'data ignore value' in img.metadata:
        fill_value = float(img.metadata['data ignore value'])
    scaled_fill_val = fill_value / scale_val
    
    wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
    
    # Create valid mask for plotting
    mid_band = bands // 2
    band_data = img_mm[:, :, mid_band]
    valid_mask_2d = (band_data != fill_value)
    
    valid_y, valid_x = np.nonzero(valid_mask_2d)
    if len(valid_y) == 0:
        print("No valid data found in image.")
        return
        
    min_y, max_y = np.min(valid_y), np.max(valid_y)
    min_x, max_x = np.min(valid_x), np.max(valid_x)
    
    # Fast display image for plotting (stride to save RAM/compute)
    plot_stride = max(1, (max_y - min_y) // 800)
    disp_img = img_mm[min_y:max_y+1:plot_stride, min_x:max_x+1:plot_stride, mid_band].astype(np.float32) / scale_val
    valid_disp = valid_mask_2d[min_y:max_y+1:plot_stride, min_x:max_x+1:plot_stride]
    disp_img[~valid_disp] = np.nan
    
    vmin, vmax = None, None
    if np.any(valid_disp):
        vmin = np.nanpercentile(disp_img, 2)
        vmax = np.nanpercentile(disp_img, 98)

    # --- Setup Window 1: Explorer ---
    fig_expl, (ax_img, ax_spec) = plt.subplots(1, 2, figsize=(16, 8))
    fig_expl.canvas.manager.set_window_title('Spectral Explorer')
    ax_img.imshow(disp_img, cmap='gray', vmin=vmin, vmax=vmax, extent=[min_x, max_x, max_y, min_y])
    ax_img.set_title("Click to view spectra")
    
    ax_spec.set_title("Spectra")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Radiance (Pixel)")
    
    ax_twin = ax_spec.twinx()
    ax_twin.set_ylabel("Reflectance (Ground Truth)")
    ax_twin.set_ylim(0, 1.1)
    
    # Plot ground truth targets on twin axis
    target_refls = []
    target_names = []
    colors = ['r', 'b', 'g', 'm', 'c']
    for i, t_file in enumerate(target_files):
        data = np.loadtxt(t_file, skiprows=1)
        ax_twin.plot(data[:, 0], data[:, 1], color=colors[i % len(colors)], linestyle='--', label=os.path.basename(t_file))
        
        # Resample to image wavelengths for ELM
        f_interp = interp1d(data[:, 0], data[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
        resampled_refl = f_interp(wavelengths)
        target_refls.append(resampled_refl)
        target_names.append(os.path.basename(t_file))
        
    ax_twin.legend(loc='upper right')
    
    def on_expl_click(event):
        if event.inaxes == ax_img:
            x, y = int(event.xdata), int(event.ydata)
            spec = img_mm[y, x, :].astype(np.float32) / scale_val
            if img_mm[y, x, mid_band] == fill_value:
                return # Nodata clicked
                
            if hasattr(on_expl_click, "line"):
                on_expl_click.line.remove()
            if hasattr(on_expl_click, "dot"):
                on_expl_click.dot.remove()
                
            line, = ax_spec.plot(wavelengths, spec, color='orange', linewidth=2, label=f"Pixel ({x}, {y})")
            on_expl_click.line = line
            ax_spec.legend(loc='upper left')
            on_expl_click.dot = ax_img.scatter(x, y, c='orange', s=50, marker='o')
            fig_expl.canvas.draw_idle()
            
    fig_expl.canvas.mpl_connect('button_press_event', on_expl_click)
    
    # --- Setup Window 2: Calibration Selection ---
    fig_cal, ax_cal = plt.subplots(figsize=(10, 8))
    fig_cal.canvas.manager.set_window_title('ELM Calibration Tool')
    ax_cal.imshow(disp_img, cmap='gray', vmin=vmin, vmax=vmax, extent=[min_x, max_x, max_y, min_y])
    
    cal_state = {'idx': 0, 'rads': [], 'coords': []}
    
    def update_cal_title():
        if cal_state['idx'] < len(target_names):
            ax_cal.set_title(f"ELM Step {cal_state['idx']+1}/{len(target_names)}: Click center of {target_names[cal_state['idx']]}\n(Use the Explorer window to verify coordinates)")
        else:
            ax_cal.set_title("Computing ELM Calibration... Please wait.")
        fig_cal.canvas.draw_idle()
            
    update_cal_title()
    
    def compute_elm():
        target_rads = np.array(cal_state['rads'])
        gt_refls = np.array(target_refls)
        
        elm_gain = np.ones(bands, dtype=np.float32)
        elm_offset = np.zeros(bands, dtype=np.float32)
        
        # Identify Water Absorption Bands
        is_water_band = ((wavelengths > 1350) & (wavelengths < 1450)) | ((wavelengths > 1800) & (wavelengths < 1950))
        
        if len(target_rads) == 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                elm_gain = gt_refls[0] / target_rads[0]
            elm_gain[np.isnan(elm_gain) | np.isinf(elm_gain)] = 1.0
            elm_offset[:] = 0.0
        else:
            for b in range(bands):
                if is_water_band[b]:
                    elm_gain[b] = 1.0
                    elm_offset[b] = 0.0
                    continue
                    
                x = target_rads[:, b]
                y = gt_refls[:, b]
                
                if np.ptp(x) < 1e-5 or np.mean(x) < 1e-5:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        elm_gain[b] = y[0] / x[0] if x[0] > 0 else 1.0
                    elm_offset[b] = 0.0
                else:
                    A = np.vstack([x, np.ones(len(x))]).T
                    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                    elm_gain[b] = m
                    elm_offset[b] = c
                    
                    
        cal_out = os.path.join(output_dir, f"elm_calibration_{run_id}.npz")
        np.savez(cal_out, gain=elm_gain, offset=elm_offset, wavelengths=wavelengths)
        
        ax_cal.set_title(f"ELM Calibration Saved!\n{cal_out}\nYou may now close these windows.")
        fig_cal.canvas.draw_idle()
        print(f"Calibration saved successfully to: {cal_out}")

    def on_cal_click(event):
        if event.inaxes == ax_cal and cal_state['idx'] < len(target_names):
            cx, cy = int(event.xdata), int(event.ydata)
            
            # Extract 3x3 window radiance
            y_start = max(0, cy - 1)
            y_end = min(lines, cy + 2)
            x_start = max(0, cx - 1)
            x_end = min(samples, cx + 2)
            
            rad_win = img_mm[y_start:y_end, x_start:x_end, :]
            rad_win_float = rad_win.astype(np.float32) / scale_val
            valid_pixels = (rad_win != fill_value)
            
            rad_win_flat = rad_win_float.reshape(-1, bands)
            valid_flat = valid_pixels.reshape(-1, bands)
            
            mean_rad = np.zeros(bands, dtype=np.float32)
            for b in range(bands):
                v = rad_win_flat[valid_flat[:, b], b]
                mean_rad[b] = np.mean(v) if len(v) > 0 else 0.0
                
            cal_state['rads'].append(mean_rad)
            cal_state['coords'].append((cx, cy))
            
            # Mark selected pixel
            ax_cal.scatter(cx, cy, c=colors[cal_state['idx'] % len(colors)], marker='X', s=100, edgecolors='white', linewidths=2)
            
            cal_state['idx'] += 1
            update_cal_title()
            
            if cal_state['idx'] == len(target_names):
                compute_elm()
                
    fig_cal.canvas.mpl_connect('button_press_event', on_cal_click)
    
    plt.show()

if __name__ == "__main__":
    main()
