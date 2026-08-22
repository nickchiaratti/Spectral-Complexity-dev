import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import re
import scienceplots

# Set the style to be used for plotting
plt.style.use(['science', 'ieee'])

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

    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(7.16, 4))
    
    ax_landsat = axes[0]
    ax_tanager = axes[1]
    ax_enmap = axes[2]
    ax_sentinel = axes[3]

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
                            
                        if band_num <= 7:
                            color = oli_1_7_color
                            ax_landsat.plot(wl, rsr, color=color, linewidth=1.5, linestyle='-')
                            ax_landsat.fill_between(wl, rsr, color=color, alpha=0.5)
                            peak_idx = np.argmax(rsr)
                            text_x = wl[peak_idx]
                            ax_landsat.text(text_x, rsr[peak_idx] + 0.02, f'B{band_num}', ha='center', va='bottom', color=color, fontsize=7)
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
            if band in ['B1','B2','B3','B4','B5','B6','B7']:
                width = upper - lower
                color = oli_1_7_color
                
                # Add rectangle patch (x, y), width, height
                rect = patches.Rectangle((lower, 0), width, 1.0, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
                ax_landsat.add_patch(rect)
                
                # Add label above the band
                text_x = (lower + width/2)
                ax_landsat.text(text_x, 1.02, band, ha='center', va='bottom', color=color, fontsize=7)

    # Landsat Subplot Formatting
   # ax_landsat.set_xlim(0, 13)
    ax_landsat.set_xlim(0.37, 2.51) # Standard VSWIR range
    ax_landsat.set_ylim(0, 1.3)
    ax_landsat.grid(True, linestyle='--', alpha=0.6)
    
    # Add plot label
    ax_landsat.text(0.98, 0.90, 'Landsat', transform=ax_landsat.transAxes, ha='right', va='top', fontsize=9, fontweight='bold')


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
                        
                        # Determine if it's a water absorption band using the metadata attribute
                        is_good = band.get('good_wavelength')
                        if is_good is None:
                            raise ValueError("Missing 'good_wavelength' attribute in Tanager JSON. Cannot verify valid bands.")
                            
                        if is_good:
                            current_color = tanager_color
                            # Plot the distribution
                            ax_tanager.plot(x, y, color=current_color, alpha=0.6, linewidth=1, linestyle='-')
                            ax_tanager.fill_between(x, y, color=current_color, alpha=0.1)
                
        except Exception as e:
            print(f"Error reading Tanager JSON: {e}")
    else:
        print("Tanager JSON file not found. Please ensure it is in the same directory.")

    # Tanager Subplot Formatting
   # ax_tanager.set_xlim(0.3, 2.6) # Standard VSWIR range for Tanager
    ax_tanager.set_xlim(0.37, 2.51) # Standard VSWIR range
    ax_tanager.set_ylim(0, 1.3)
    ax_tanager.grid(True, linestyle='--', alpha=0.6)
    
    # Add plot label
    ax_tanager.text(0.98, 0.90, 'Tanager', transform=ax_tanager.transAxes, ha='right', va='top', fontsize=9, fontweight='bold')

    # ==========================================
    # Plot 3: EnMAP Hyperspectral
    # ==========================================
    enmap_excel_path = os.path.join(script_dir, 'EnMAP_Spectral_Bands_update.xlsx')
    
    if os.path.exists(enmap_excel_path):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                df_vnir = pd.read_excel(enmap_excel_path, sheet_name='VNIR')
                df_swir = pd.read_excel(enmap_excel_path, sheet_name='SWIR')
                df_enmap = pd.concat([df_vnir, df_swir], ignore_index=True)
            
            enmap_color = '#9467bd' # Purple for EnMAP hyperspectral lines
            bands_found = False
            
            for idx, row in df_enmap.iterrows():
                center_nm = row.get('CW (nm)')
                fwhm_nm = row.get('FWHM (nm)')
                
                if pd.notna(center_nm) and pd.notna(fwhm_nm):
                    bands_found = True
                    # Excel has wavelengths in nm, we need um
                    center = center_nm / 1000.0
                    fwhm = fwhm_nm / 1000.0
                    
                    # Exclude major atmospheric water absorption bands
                    if (1.35 <= center <= 1.45) or (1.80 <= center <= 1.95):
                        continue
                    
                    sigma = fwhm / 2.355
                    x = np.linspace(center - 3*sigma, center + 3*sigma, 100)
                    y = np.exp(-0.5 * ((x - center) / sigma) ** 2)
                    
                    ax_enmap.plot(x, y, color=enmap_color, alpha=0.6, linewidth=1, linestyle='-')
                    ax_enmap.fill_between(x, y, color=enmap_color, alpha=0.1)
                        
            if not bands_found:
                print("No band data found in EnMAP Excel.")
        except Exception as e:
            print(f"Error reading EnMAP Excel: {e}")
    else:
        print("EnMAP Excel file not found.")

    # EnMAP Subplot Formatting
    ax_enmap.set_xlim(0.37, 2.51) # Standard VSWIR range
    ax_enmap.set_ylim(0, 1.3)
    ax_enmap.grid(True, linestyle='--', alpha=0.6)
    ax_enmap.text(0.98, 0.90, 'EnMAP', transform=ax_enmap.transAxes, ha='right', va='top', fontsize=9, fontweight='bold')

    # ==========================================
    # Plot 4: Sentinel-2A
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
                    peak_idx = np.argmax(rsr)
                    peak_wl = wl[peak_idx]
                    
                    band_name = col.split('_')[-1] # e.g. B1, B8A
                    
                    if not ((1.35 <= peak_wl <= 1.45) or (1.80 <= peak_wl <= 1.95) or band_name == 'B8'):
                        color_to_use = sentinel_color

                        ax_sentinel.plot(wl, rsr, color=color_to_use, linewidth=1.5, linestyle='-')
                        ax_sentinel.fill_between(wl, rsr, color=color_to_use, alpha=0.5)
                        
                        # Add text label
                        text_x = wl[peak_idx]
                        ax_sentinel.text(text_x, rsr[peak_idx] + 0.023, band_name, ha='center', va='bottom', color=color_to_use, fontsize=7)
                    
        except Exception as e:
            print(f"Error reading Sentinel-2A Excel file: {e}")
    else:
        print("Sentinel-2A Excel file not found. Please ensure it is in the same directory.")

    # Sentinel Subplot Formatting
    ax_sentinel.set_xlabel('Wavelength (µm)')
    ax_sentinel.set_xlim(0.37, 2.51) # Standard VSWIR range
    ax_sentinel.set_ylim(0, 1.3)
    ax_sentinel.grid(True, linestyle='--', alpha=0.6)
    
    # Add plot label
    ax_sentinel.text(0.98, 0.90, 'Sentinel-2', transform=ax_sentinel.transAxes, ha='right', va='top', fontsize=9, fontweight='bold')

    fig.supylabel('Relative Spectral Response (RSR)', fontsize=10, x=0.04)

    # Final Adjustments and Show/Save
    plt.tight_layout(rect=[0.04, 0, 1, 1], h_pad=0.2)
    plt.savefig(os.path.join(script_dir,'spectral_ranges_comparison.png'), dpi=600, bbox_inches='tight')
    #plt.show()

if __name__ == '__main__':
    plot_spectral_ranges()