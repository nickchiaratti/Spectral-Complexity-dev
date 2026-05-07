import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import re
import scienceplots

# Set the style to be used for plotting
plt.style.use(['science', 'no-latex'])

def plot_spectral_ranges():
    """
    Generates side-by-side plots for Landsat 8/9 and Tanager spectral ranges.
    Reads Tanager data from JSON and Landsat TIRS/OLI data from files. Idealized 
    rectangles are used for Landsat OLI bands if the RSR file is missing.
    """
    
    # --- File Paths ---
    script_dir = os.path.dirname(os.path.realpath(__file__))
    tanager_json_path = os.path.join(script_dir, 'Tanager_wavelengths.json')
    tirs_b10_path = os.path.join(script_dir, 'L9_TIRS2_RSR.xlsx')
    tirs_b11_path = os.path.join(script_dir, 'L9_TIRS2_RSR.xlsx')
    oli_rsr_path = os.path.join(script_dir, 'L9_OLI2_RSR.xlsx')

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(18, 10))
    fig.suptitle('Spectral Ranges Sampled: Landsat 8/9 vs Tanager vs Sentinel-2A', fontsize=16, fontweight='bold')
    
    ax_landsat = axes[0]
    ax_tanager = axes[1]
    ax_sentinel = axes[2]

    # ==========================================
    # Plot 1: Landsat 8/9 (OLI + TIRS)
    # ==========================================
    
    oli_1_7_color = '#1f77b4'  # Blue for Bands 1-7
    oli_other_color = '#7f7f7f' # Gray for Bands 8-9
    tirs_color = '#d62728'      # Red for TIRS Bands 10-11

    # 1. Plot Real OLI Bands from Attached Excel (Bands 1-9)
    oli_plotted = False
    if os.path.exists(oli_rsr_path):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                xls_oli = pd.read_excel(oli_rsr_path, sheet_name=None)
            
            oli_band_mapping = {
                'CoastalAerosol': 1, 'Blue': 2, 'Green': 3, 'Red': 4,
                'NIR': 5, 'SWIR1': 6, 'SWIR2': 7, 'Pan': 8, 'Cirrus': 9
            }
            
            for sheet_name, df_oli in xls_oli.items():
                if sheet_name in oli_band_mapping:
                    band_num = oli_band_mapping[sheet_name]
                    
                    wl = df_oli.iloc[:, 0].dropna().values
                    rsr = df_oli.iloc[:, 1].dropna().values
                    
                    if len(wl) > 0:
                        if wl[0] > 10: # Convert nm to um
                            wl = wl / 1000.0
                            
                        color = oli_1_7_color if 1 <= band_num <= 7 else oli_other_color
                        ax_landsat.plot(wl, rsr, color=color, linewidth=1.5)
                        ax_landsat.fill_between(wl, rsr, color=color, alpha=0.5)
                        peak_idx = np.argmax(rsr)
                        ax_landsat.text(wl[peak_idx], rsr[peak_idx] + 0.02, f'B{band_num}', ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')
                        oli_plotted = True
                                
        except Exception as e:
            print(f"Error reading OLI Excel file: {e}")

    # Fallback to Idealized OLI Bands if Excel fails or doesn't exist
    if not oli_plotted:
        print("OLI Excel file missing or unreadable. Falling back to idealized rectangular bands.")
        # Wavelengths given in micrometers (um) [Lower bound, Upper bound]
        oli_bands = {
            'B1': (0.435, 0.451), 'B2': (0.452, 0.512), 'B3': (0.533, 0.590),
            'B4': (0.636, 0.673), 'B5': (0.851, 0.879), 'B6': (1.566, 1.651),
            'B7': (2.107, 2.294),
            # Rest of the bands
            #'B8': (0.503, 0.676), 'B9': (1.363, 1.384)
        }

        # Draw OLI bands as rectangular idealized Relative Spectral Responses
        for band, (lower, upper) in oli_bands.items():
            width = upper - lower
            color = oli_1_7_color if band in ['B1','B2','B3','B4','B5','B6','B7'] else oli_other_color
            
            # Add rectangle patch (x, y), width, height
            rect = patches.Rectangle((lower, 0), width, 1.0, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
            ax_landsat.add_patch(rect)
            
            # Add label above the band
            ax_landsat.text(lower + width/2, 1.02, band, ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

    # 2. Plot Real TIRS Bands from Attached Excel files (Bands 10-11)
    if os.path.exists(tirs_b10_path):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df_b10 = pd.read_excel(tirs_b10_path, sheet_name='TIRS Band 10 BA RSR')
                df_b11 = pd.read_excel(tirs_b11_path, sheet_name='TIRS Band 11 BA RSR')
            
            # Extract arrays
            wl_10 = df_b10.iloc[:, 0].dropna().values
            rsr_10 = df_b10.iloc[:, 1].dropna().values
            wl_11 = df_b11.iloc[:, 0].dropna().values
            rsr_11 = df_b11.iloc[:, 1].dropna().values
            
            # Plot the TIRS curves
            ax_landsat.plot(wl_10, rsr_10, color=tirs_color, linewidth=2, label='TIRS (B10/B11)')
            ax_landsat.fill_between(wl_10, rsr_10, color=tirs_color, alpha=0.5)
            ax_landsat.text(np.mean(wl_10), max(rsr_10) + 0.02, 'B10', ha='center', va='bottom', fontsize=9, color=tirs_color, fontweight='bold')
            
            ax_landsat.plot(wl_11, rsr_11, color=tirs_color, linewidth=2)
            ax_landsat.fill_between(wl_11, rsr_11, color=tirs_color, alpha=0.5)
            ax_landsat.text(np.mean(wl_11), max(rsr_11) + 0.02, 'B11', ha='center', va='bottom', fontsize=9, color=tirs_color, fontweight='bold')
        except Exception as e:
            print(f"Error reading TIRS Excel file: {e}")
    else:
        print("TIRS Excel files not found. Please ensure they are in the same directory.")

    # Landsat Subplot Formatting
    ax_landsat.set_title('Landsat 8/9 (OLI & TIRS)', fontsize=14)
    ax_landsat.set_xlabel('Wavelength (µm)', fontsize=12)
    ax_landsat.set_ylabel('Relative Spectral Response (RSR)', fontsize=12)
   # ax_landsat.set_xlim(0, 13)
    ax_landsat.set_xlim(0.3, 3) # Standard VSWIR range
    ax_landsat.set_ylim(0, 1.15)
    ax_landsat.grid(True, linestyle='--', alpha=0.6)
    
    # Custom legend
    import matplotlib.lines as mlines
    l1 = patches.Patch(color=oli_1_7_color, label='OLI Bands 1-7')
    l2 = patches.Patch(color=oli_other_color, label='OLI Bands 8-9')
    l3 = patches.Patch(color=tirs_color, label='TIRS Bands 10-11')
    ax_landsat.legend(handles=[l1, l2, l3], loc='upper right')


    # ==========================================
    # Plot 2: Tanager Hyperspectral
    # ==========================================
    
    if os.path.exists(tanager_json_path):
        try:
            with open(tanager_json_path, 'r') as f:
                tanager_data = json.load(f)
            
            bands = tanager_data.get('assets', {}).get('basic_radiance_hdf5', {}).get('eo:bands', [])
            
            if not bands:
                print("No band data found in Tanager JSON.")
            else:
                tanager_color = '#2ca02c' # Green for Tanager hyperspectral lines
                water_band_color = '#7f7f7f' # Gray for water absorption bands
                
                # Plot each Tanager band as a Gaussian distribution
                # FWHM = 2.355 * sigma  => sigma = FWHM / 2.355
                for band in bands:
                    center = band.get('center_wavelength')
                    fwhm = band.get('full_width_half_max')
                    
                    if center is not None and fwhm is not None:
                        sigma = fwhm / 2.355
                        
                        # Generate x values for the Gaussian curve (+/- 3 standard deviations)
                        x = np.linspace(center - 3*sigma, center + 3*sigma, 100)
                        
                        # Generate Gaussian y values (normalized to peak at 1.0 for RSR comparability)
                        y = np.exp(-0.5 * ((x - center) / sigma) ** 2)
                        
                        # Determine color based on wavelength (water absorption)
                        if (1.35 <= center <= 1.45) or (1.80 <= center <= 1.95):
                            current_color = water_band_color
                        else:
                            current_color = tanager_color
                        
                        # Plot the distribution
                        ax_tanager.plot(x, y, color=current_color, alpha=0.6, linewidth=1)
                        ax_tanager.fill_between(x, y, color=current_color, alpha=0.1)
                
                # Add a custom legend entry for Tanager
                l4 = mlines.Line2D([], [], color=tanager_color, label=f'Tanager Hyperspectral ({len(bands)} bands)')
                l_water = mlines.Line2D([], [], color=water_band_color, label='Water Vapor Absorption')
                ax_tanager.legend(handles=[l4, l_water], loc='upper right')
                
        except Exception as e:
            print(f"Error reading Tanager JSON: {e}")
    else:
        print("Tanager JSON file not found. Please ensure it is in the same directory.")

    # Tanager Subplot Formatting
    ax_tanager.set_title('Tanager-1 (Hyperspectral)', fontsize=14)
    ax_tanager.set_xlabel('Wavelength (µm)', fontsize=12)
    ax_tanager.set_ylabel('Simulated Spectral Response (Scaled via FWHM)', fontsize=12)
   # ax_tanager.set_xlim(0.3, 2.6) # Standard VSWIR range for Tanager
    ax_tanager.set_xlim(0.3, 3) # Standard VSWIR range
    ax_tanager.set_ylim(0, 1.15)
    ax_tanager.grid(True, linestyle='--', alpha=0.6)

    # ==========================================
    # Plot 3: Sentinel-2A
    # ==========================================
    sentinel_path = os.path.join(script_dir, 'Sentinel-2A MSI Spectral Responses.xlsx')
    
    if os.path.exists(sentinel_path):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df_s2a = pd.read_excel(sentinel_path, sheet_name='Spectral Responses (S2A)')
            
            # Wavelength in the file is in nm, we need um
            wl = df_s2a['SR_WL'].values / 1000.0
            sentinel_color = '#ff7f0e' # Orange
            
            # Find all band columns
            band_cols = [c for c in df_s2a.columns if c.startswith('S2A_SR_AV_B')]
            
            for col in band_cols:
                rsr = df_s2a[col].values
                if np.max(rsr) > 0.01: # Only plot bands with actual response
                    ax_sentinel.plot(wl, rsr, color=sentinel_color, linewidth=1.5)
                    ax_sentinel.fill_between(wl, rsr, color=sentinel_color, alpha=0.5)
                    
                    # Add text label
                    peak_idx = np.argmax(rsr)
                    band_name = col.split('_')[-1] # e.g. B1, B8A
                    ax_sentinel.text(wl[peak_idx], rsr[peak_idx] + 0.02, band_name, ha='center', va='bottom', fontsize=9, color=sentinel_color, fontweight='bold')
                    
            import matplotlib.lines as mlines
            l5 = mlines.Line2D([], [], color=sentinel_color, label='Sentinel-2A MSI')
            ax_sentinel.legend(handles=[l5], loc='upper right')
            
        except Exception as e:
            print(f"Error reading Sentinel-2A Excel file: {e}")
    else:
        print("Sentinel-2A Excel file not found. Please ensure it is in the same directory.")

    # Sentinel Subplot Formatting
    ax_sentinel.set_title('Sentinel-2A (MSI)', fontsize=14)
    ax_sentinel.set_xlabel('Wavelength (µm)', fontsize=12)
    ax_sentinel.set_ylabel('Relative Spectral Response (RSR)', fontsize=12)
    ax_sentinel.set_xlim(0.3, 3) # Standard VSWIR range
    ax_sentinel.set_ylim(0, 1.15)
    ax_sentinel.grid(True, linestyle='--', alpha=0.6)

    # Final Adjustments and Show/Save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(script_dir,'spectral_ranges_comparison.png'), dpi=300)
    plt.show()

if __name__ == '__main__':
    plot_spectral_ranges()