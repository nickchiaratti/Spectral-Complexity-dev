import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pyproj import Transformer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
LOCATION = 'Rochesterv2'

# Import the core spectral complexity logic and the skeleton loader
sys.path.append(r'F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev')
import SpecComplex as sc
from Harmonized_SC.HLST_skeleton_loader import HLST_ARD_Interface

def main():
    h5_path = f'C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized.h5'
    sc_h5_path = f'C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-None.h5'
    
    if not os.path.exists(h5_path):
        print(f"Error: Could not find raw HDF5 at {h5_path}")
        return
    if not os.path.exists(sc_h5_path):
        print(f"Error: Could not find SC HDF5 at {sc_h5_path}")
        return

    # Use the skeleton loader initialized with the SC cube to properly index timelines and masks
    ard = HLST_ARD_Interface(sc_h5_path)
    
    # We will open the raw cube directly to extract raw surface reflectance
    raw_h5 = h5py.File(h5_path, 'r')
    
    sensors_to_process = ['HLSL30', 'HLSS30', 'TANAGER']
    processed_sensors = []
    
    # We will store the global results
    results = {}
    plotted_failures = set()
    first_failures_info = {}
    
    for t in range(ard.num_frames):
        # We process 1/10th of the entire timeline for HLS, but ALL of Tanager
        sensor_grid = ard.grids[t]
        
        if sensor_grid not in sensors_to_process:
            continue
            
        local_idx = ard.indices[t]
        height, width = ard.height, ard.width
        
        print(f"\nProcessing Frame {t}: Sensor={sensor_grid}, Local Index={local_idx}")
        
        native_ds = raw_h5[f'/HDFEOS/GRIDS/{sensor_grid}/Data Fields/surface_reflectance']
        mask_ds = ard.harm_grp['common_mask']
        
        sr = native_ds[local_idx, ...] # [bands, height, width]
        sr = np.transpose(sr, (1, 2, 0)).astype(np.float64) # [height, width, bands]
        
        # Filter out invalid hyperspectral bands if attribute exists
        good_bands = native_ds.attrs.get('good_wavelengths')
        if good_bands is not None:
            valid_indices = np.where(good_bands == 1)[0]
            sr = sr[:, :, valid_indices]
            
        bands = sr.shape[2]
        mask = mask_ds[t, ...] # [height, width]
        
        if np.all(mask != 1):
            print("Frame is entirely masked. Skipping.")
            continue
            

        
        total_pixels_evaluated = 0
        total_encapsulated = 0
        k_opt_freq = defaultdict(int)
        
        stride = 1 
        
        for r in tqdm(range(1, height - 1, stride), desc=f"Processing {sensor_grid}"):
            for c in range(1, width - 1, stride):
                window_mask = mask[r-1:r+2, c-1:c+2]
                
                # STRICT MASKING: Do not overlap the tile with any areas that are masked.
                # Must be 9 perfectly valid pixels. (common_mask convention: 0 = Valid, 1 = Invalid)
                if np.any(window_mask != 0):
                    continue
                
                window_pixels = sr[r-1:r+2, c-1:c+2, :].reshape(9, bands)
                
                # Reject physically invalid negative reflectance values (atmospheric correction failures over snow/ice)
                if np.any(window_pixels < 0):
                    continue
                
                # Check for completely flat/zero spectra
                if np.all(window_pixels == 0):
                    continue
                
                try:
                    # MaxD requires 3D shape (H, W, Bands)
                    data_3d = window_pixels.reshape(3, 3, bands)
                    # Extract 7 endmembers
                    endmembers, _ = sc.maximumDistance(data_3d, num_endmembers=7, strict_nan=True)
                except Exception as e:
                    raise Exception(f"MaxD failed at pixel ({r}, {c}): {e}")
                    
                k_min = 4
                k_max = 7
                
                # Check if MaxD returned fewer than 7 (rare, but possible)
                actual_ems = endmembers.shape[1]
                if actual_ems < k_min:
                    raise Exception(f"MaxD returned fewer than {k_min} endmembers: {actual_ems} at pixel ({r}, {c})")
                
                # Localize the simplex to the second endmember (minimum norm)
                localization_vector = endmembers[:, 1]
                
                # We must delete the localization vector from the endmembers array before calculating Gramian
                # otherwise the localized vector is 0 and the volume collapses to 0!
                remaining_endmembers = np.delete(endmembers, 1, axis=1)
                
                volumes = sc.calcGramLocalVolumes(remaining_endmembers, localization_vector)
                
                # Insert 0 at index 0 and 1 so that index matches k
                # (e.g. index 4 corresponds to k=4 endmembers)
                volumes = np.insert(volumes, 0, [0.0, 0.0])
                
                # We need to find the k that maximizes the volume, between k_min and k_max
                sub_volumes = volumes[k_min : k_max + 1]
                
                best_idx = np.argmax(sub_volumes)
                k_opt = k_min + best_idx
                
                # The presented k_opt integer is decremented by 1 since v2 is used 
                # as a shift rather than an independent vertex in the mathematical language.
                presented_k_opt = k_opt - 1
                k_opt_freq[presented_k_opt] += 1
                
                # Now evaluate encapsulation using ONLY the optimal number of endmembers
                ems_opt = endmembers[:, :k_opt] # Shape: [bands, k_opt]
                
                lower_bound = np.min(ems_opt, axis=1) # Shape: [bands]
                upper_bound = np.max(ems_opt, axis=1) # Shape: [bands]
                
                # Define band-specific tolerances based on atmospheric correction uncertainty
                tolerances = np.full(bands, 0.03)
                #if bands < 8: #Landsat Case
                #    # Approximating 2xsigma for HLS errors identified in  Claverie et al. (2025) 
                #    tolerances[0] = 0.022 #Ultra Blue
                #    tolerances[1] = 0.025 # Blue
                #    tolerances[2] = 0.02 #Green
                #    tolerances[3] = 0.02 #Red
                #    tolerances[4] = 0.04 #NIR
                #    tolerances[5] = 0.04 #SWIR 1
                #    tolerances[6] = 0.03 #SWIR 2 
                #elif bands == 9: #Sentinel-2 Case
                #    # Approximating 2xsigma for HLS errors identified in  Claverie et al. (2025) 
                #    tolerances[0] = 0.022 #Ultra Blue
                #    tolerances[1] = 0.025 # Blue
                #    tolerances[2] = 0.02 #Green
                #    tolerances[3] = 0.02 #Red
                #    tolerances[7] = 0.04 #NIR
                #    tolerances[8] = 0.04 #SWIR 1
                #    tolerances[9] = 0.03 #SWIR 2 
                
                enclosed = 0
                encapsulated_pixels = []
                failed_pixels = []
                
                for p in window_pixels:
                    # Test if pixel falls within [min, max] envelope across all bands
                    if np.all((p >= lower_bound - tolerances) & (p <= upper_bound + tolerances)):
                        enclosed += 1
                        encapsulated_pixels.append(p)
                    else:
                        failed_pixels.append(p)
                        
                total_encapsulated += enclosed
                total_pixels_evaluated += 9
                
                # Plot the first AABB failure we encounter for each sensor
                if len(failed_pixels) > 0 and len(encapsulated_pixels) > 0 and sensor_grid not in plotted_failures:
                    output_dir = r'F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\Harmonized_SC\encapsulation_results'
                    os.makedirs(output_dir, exist_ok=True)
                    
                    plt.figure(figsize=(10, 6))
                    x_bands = np.arange(bands)
                    
                    # Plot AABB
                    plt.fill_between(x_bands, lower_bound, upper_bound, color='gray', alpha=0.3, label='Spectral Bounds Envelope')

                    # Plot passing pixels
                    for i, p in enumerate(encapsulated_pixels):
                        plt.plot(x_bands, p, 'b-', alpha=0.5, label='Encapsulated Pixel' if i==0 else "")
                    # Plot failing pixels
                    for i, p in enumerate(failed_pixels):
                        plt.plot(x_bands, p, 'r-', linewidth=1.5, label='Failed Pixel' if i==0 else "")
                    
                    # Plot Endmembers (All 7 extracted)
                    em_colors = ['purple', 'green', 'orange', 'cyan', 'brown', 'magenta', 'olive']
                    for i in range(endmembers.shape[1]):
                        color = em_colors[i % len(em_colors)]
                        label_str = f'Endmember {i+1}'
                        if i >= k_opt:
                            label_str += ' (Ignored)'
                        plt.plot(x_bands, endmembers[:, i], color=color, linestyle='--', linewidth=2, label=label_str)

                    plt.title(f"{sensor_grid} - Spectral Bounds Envelope Encapsulation Failure Example (k={presented_k_opt})")
                    plt.xlabel("Band Index")
                    plt.ylabel("Reflectance")
                    plt.legend(loc='upper right')
                    plt.tight_layout()
                    
                    plot_path = os.path.join(output_dir, f"spectral_bounds_envelope_failure_{sensor_grid}.png")
                    plt.savefig(plot_path)
                    plt.close()
                    
                    print(f"\nSaved failure plot for {sensor_grid} at {plot_path}")
                    print(f"Failure occurred at Row={r}, Col={c}")
                    
                    # Extract NxN ortho_visual chip
                    CHIP_DIM = 100
                    vis_ds = raw_h5[f'/HDFEOS/GRIDS/{sensor_grid}/Data Fields/ortho_visual']
                    vis_image = vis_ds[local_idx, ...] # [bands, height, width]
                    vis_image = np.transpose(vis_image, (1, 2, 0)) # [height, width, 3]
                    
                    r_min = max(0, r - int(CHIP_DIM/2))
                    r_max = min(height, r + int(CHIP_DIM/2))
                    c_min = max(0, c - int(CHIP_DIM/2))
                    c_max = min(width, c + int(CHIP_DIM/2))
                    
                    chip = vis_image[r_min:r_max, c_min:c_max, :]
                    
                    if chip.dtype != np.uint8:
                        if np.max(chip) > 255:
                            chip_plot = np.clip(chip / 10000.0, 0, 1)
                        else:
                            chip_plot = np.clip(chip, 0, 1)
                    else:
                        chip_plot = chip
                        
                    plt.figure(figsize=(6, 6))
                    plt.imshow(chip_plot)
                    center_r = r - r_min
                    center_c = c - c_min
                    rect = plt.Rectangle((center_c - 1.5, center_r - 1.5), 3, 3, linewidth=2, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rect)
                    
                    dt = datetime.fromtimestamp(ard.times[t], tz=timezone.utc)
                    date_str = dt.strftime('%Y-%m-%d')
                    
                    proj_x, proj_y = ard.affine * (c, r)
                    transformer_reverse = Transformer.from_crs(ard.crs, "EPSG:4326", always_xy=True)
                    lon, lat = transformer_reverse.transform(proj_x, proj_y)
                    
                    plt.title(f"{sensor_grid} Visual Chip (40x40)\nCenter: {lat:.6f}, {lon:.6f} | Date: {date_str}", fontsize=10)
                    plt.axis('off')
                    
                    chip_path = os.path.join(output_dir, f"aabb_failure_chip_{sensor_grid}.png")
                    plt.savefig(chip_path)
                    plt.close()
                    
                    print(f"Saved visual chip to {chip_path}\n")
                    
                    first_failures_info[sensor_grid] = {
                        'date': date_str,
                        'lat': lat,
                        'lon': lon,
                        'row': r,
                        'col': c,
                        'k_opt': presented_k_opt
                    }
                    plotted_failures.add(sensor_grid)
                
        if sensor_grid not in results:
            results[sensor_grid] = {
                'encapsulated_pixels': 0,
                'total_pixels': 0,
                'k_opt_freq': defaultdict(int)
            }
            
        results[sensor_grid]['encapsulated_pixels'] += total_encapsulated
        results[sensor_grid]['total_pixels'] += total_pixels_evaluated
        for k, v in k_opt_freq.items():
            results[sensor_grid]['k_opt_freq'][k] += v
        
        if total_pixels_evaluated > 0:
            pct = (total_encapsulated / total_pixels_evaluated) * 100.0
            print(f"  Frame Encapsulation: {pct:.2f}% ({total_encapsulated}/{total_pixels_evaluated} pixels)")
            print(f"  Frame k_opt frequency: {dict(k_opt_freq)}")
            
    raw_h5.close()
    
    # Save the results to text to ensure they aren't lost
    output_dir = r'F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\Harmonized_SC\encapsulation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'gram_optimal_encapsulation.txt'), 'w') as f:
        f.write("Gramian Optimal k Spectral Bounds Encapsulation\n")
        f.write("Strict Masking: 9/9 valid pixels per window\n")
        f.write("Tolerance: Band-specific (Ultra-Blue/Blue=0.015, Others=0.005)\n\n")
        
        for sensor, data in results.items():
            tot = data['total_pixels']
            enc = data['encapsulated_pixels']
            pct = (enc / tot) * 100.0 if tot > 0 else 0
            
            f.write(f"Sensor: {sensor}\n")
            f.write(f"  Total Valid Pixels Tested: {tot}\n")
            f.write(f"  Encapsulated Pixels: {enc} ({pct:.2f}%)\n")
            f.write(f"  k_opt Frequencies: {data['k_opt_freq']}\n")
            
            if sensor in first_failures_info:
                info = first_failures_info[sensor]
                f.write(f"  First Failure Context: Date={info['date']}, Lat={info['lat']:.6f}, Lon={info['lon']:.6f}, Array=[{info['row']}, {info['col']}], k_opt={info['k_opt']}\n")
            f.write("\n")
            
            # Generate k_opt bar chart
            freqs = data['k_opt_freq']
            if freqs:
                plt.figure(figsize=(8, 5))
                ks = list(freqs.keys())
                counts = list(freqs.values())
                
                # Ensure x-axis represents k values 3-7 for consistent scale
                all_ks = np.arange(3, 8)
                all_counts = [freqs.get(k, 0) for k in all_ks]
                
                plt.bar(all_ks, all_counts, color='skyblue', edgecolor='black')
                plt.title(f"{sensor} - Optimal Geometric Dimensionality ($k_{{opt}}$) Distribution\nTotal Valid Windows: {tot//9}")
                plt.xlabel("Optimal Number of Endmembers ($k_{opt}$)")
                plt.ylabel("Frequency (Number of $3\\times3$ Windows)")
                plt.xticks(all_ks)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add count labels above bars
                for i, v in enumerate(all_counts):
                    if v > 0:
                        plt.text(all_ks[i], v + (max(all_counts)*0.01), str(v), ha='center', va='bottom')
                        
                plt.tight_layout()
                plot_path = os.path.join(output_dir, f"k_opt_distribution_{sensor}.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"Saved k_opt distribution plot to: {plot_path}")
            
    print("\nResults saved to:", os.path.join(output_dir, 'gram_optimal_encapsulation.txt'))

if __name__ == "__main__":
    main()
