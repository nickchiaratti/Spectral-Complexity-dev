import numpy as np
import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Default source directory for calibration targets
base_dir = r"C:\satelliteImagery\ground_truth\calibration_targets"
# Optional: reference text path for a single target file (or None to process all targets in base_dir)
ref_txt_path = None

SENSORS = {
    "rsLandsat8": {
        "wav": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsLandsat8\S07LSAT8_Wavelengths_Landsat8_(7_bands)_microns.txt",
        "fwhm": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsLandsat8\S07LSAT8_Bandpass_(FWHM)_Landsat8_(7_bands)_um.txt"
    },
    "rsSentinel2": {
        "wav": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsSentinel2\S07SNTL2_Wavelengths_Sentinel2_(13_bands)_microns.txt",
        "fwhm": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsSentinel2\S07SNTL2_Bandpass_(FWHM)_Sentinel2_(13_bands)_um.txt"
    },
    "rsWorldView3": {
        "wav": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsWorldView3\S07WV3_Wavelengths_WorldView3_(16_bands)_micron.txt",
        "fwhm": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsWorldView3\S07WV3_Bandpass_(FWHM)_WorldView3_(16_bands)_um.txt"
    },
    "cvHYMAP2014": {
        "wav": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvHYMAP2014\s07_HY14_HyMap2_Alaska2014_wavelengths_126ch.txt",
        "fwhm": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvHYMAP2014\s07_HY14_HyMap2_Alaska2014_bandpass_(FWHM)_126ch.txt"
    },
    "cvAVIRISc2014": {
        "wav": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvAVIRISc2014\s07_AV14_Wavelengths_in_microns_224_ch_AVIRIS14a.txt",
        "fwhm": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvAVIRISc2014\s07_AV14_Resolution_in_microns_224_ch_AVIRIS14a.txt"
    }
}

# Standard ASD 2151 wavelength grid (350 nm - 2500 nm, 1 nm spacing)
ASD_WAVELENGTHS = np.arange(350.0, 2500.0 + 1.0, 1.0, dtype=np.float32)

def load_asd_txt(filepath):
    """Loads USGS splib07 ASD ASCII bandpass / wavelength file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[1:]:
        if not line.strip():
            continue
        val = float(line.strip())
        if val < -1.0: # typically -1.23e34 nodata
            data.append(np.nan)
        else:
            data.append(val)
    return np.array(data, dtype=np.float32)

def load_spectrum_txt(filepath):
    """
    Loads a 2-column spectrum text file where column 1 is center wavelength in nm,
    and column 2 is surface reflectance.
    
    Robustly handles:
      - Optional headers (e.g. 'Wavelength_nm Reflectance_0_1') or no header
      - Arbitrary delimiters (tabs, single or multiple spaces)
      - Standard decimal notation and scientific notation (varying significant digits)
      - Blank or commented lines
    """
    wavelengths = []
    reflectances = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                w = float(parts[0])
                r = float(parts[1])
                wavelengths.append(w)
                reflectances.append(r)
            except ValueError:
                # Header row or non-numeric metadata line
                continue
                
    w_arr = np.array(wavelengths, dtype=np.float64)
    r_arr = np.array(reflectances, dtype=np.float32)
    if len(w_arr) == 0:
        raise ValueError(f"No valid 2-column numerical spectral data found in: {filepath}")
    
    # Sort monotonically by wavelength
    sort_indices = np.argsort(w_arr)
    return w_arr[sort_indices], r_arr[sort_indices]

def convert_calibration_spectra(target_dir=base_dir, single_file=ref_txt_path, sensors=SENSORS):
    """
    Converts 2-column spectral text files into sensor-resampled CSV files for use in simulate_inset_squares.py.
    """
    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    if single_file:
        if os.path.isabs(single_file):
            target_files = [single_file]
        else:
            target_files = [os.path.join(target_dir, single_file)]
    else:
        # Find all .txt files in target_dir, excluding any sensor files or other non-target text
        all_txt = glob.glob(os.path.join(target_dir, "*.txt"))
        # Exclude splib sensor definitions or existing exported logs if any
        target_files = [f for f in all_txt if not os.path.basename(f).startswith(("S07", "s07"))]
        target_files.sort()
        
    if not target_files:
        print(f"No spectral text files found to process in: {target_dir}")
        return
        
    print(f"Found {len(target_files)} spectra file(s) in {target_dir}:")
    for f in target_files:
        print(f"  - {os.path.basename(f)}")
        
    plt.figure(figsize=(10, 6))
    
    # Cache sensor band wavelengths and FWHMs
    sensor_bands = {}
    for sensor, paths in sensors.items():
        s_wav = load_asd_txt(paths['wav']) * 1000.0   # Convert microns to nm
        s_fwhm = load_asd_txt(paths['fwhm']) * 1000.0 # Convert microns to nm
        sensor_bands[sensor] = (s_wav, s_fwhm)
        
    for filepath in target_files:
        target_name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\nProcessing target '{target_name}' from {filepath}...")
        
        w_raw, r_raw = load_spectrum_txt(filepath)
        print(f"  Loaded {len(w_raw)} points ({w_raw.min():.1f} - {w_raw.max():.1f} nm)")
        
        # 1. Create dense 1 nm grid for precise Gaussian SRF convolution
        grid_start = int(np.ceil(w_raw.min()))
        grid_end = int(np.floor(w_raw.max()))
        w_nm = np.arange(grid_start, grid_end + 1, 1.0, dtype=np.float64)
        dhr = np.interp(w_nm, w_raw, r_raw).astype(np.float32)
        
        # 2. Save cvASD data matching standard 2151 ASD channels (350 nm to 2500 nm)
        # Used directly by simulate_inset_squares.py when sensor_name == "cvASD"
        asd_dhr = np.interp(ASD_WAVELENGTHS, w_raw, r_raw).astype(np.float32)
        df_asd = pd.DataFrame({
            'wavelength': ASD_WAVELENGTHS,
            'DHR_mean': asd_dhr,
            'DHR_stdev': np.zeros_like(asd_dhr, dtype=np.float32)
        })
        asd_csv_path = os.path.join(target_dir, f"cvASD_{target_name}.csv")
        df_asd.to_csv(asd_csv_path, index=False)
        print(f"  Saved {asd_csv_path} (2151 channels)")
        
        # Plot continuous spectrum
        plt.plot(w_nm, dhr, label=target_name)
        
        # 3. Resample for each simulated sensor using Gaussian Spectral Response Function (SRF)
        for sensor, (s_wav, s_fwhm) in sensor_bands.items():
            resampled_dhr = []
            for cw, fwhm in zip(s_wav, s_fwhm):
                if np.isnan(cw) or np.isnan(fwhm) or fwhm <= 0:
                    resampled_dhr.append(np.nan)
                    continue
                
                # Gaussian Spectral Response Function
                srf = np.exp(-4 * np.log(2) * ((w_nm - cw)**2) / (fwhm**2))
                sum_srf = np.sum(srf)
                if sum_srf > 0:
                    r_val = float(np.sum(dhr * srf) / sum_srf)
                else:
                    r_val = np.nan
                resampled_dhr.append(r_val)
                
            df_resampled = pd.DataFrame({
                'wavelength': s_wav,
                'DHR_mean': resampled_dhr
            })
            r_csv_path = os.path.join(target_dir, f"{sensor}_{target_name}.csv")
            df_resampled.to_csv(r_csv_path, index=False)
            print(f"  Saved {r_csv_path} ({len(s_wav)} bands)")
            
    # Save combined plot of source calibration spectra
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylabel("Reflectance")
    plt.xlabel("Wavelength (nm)")
    plt.title("Calibration Targets Spectral Reflectance")
    plot_path = os.path.join(target_dir, "Calibration_Targets_Spectral_Reflectance.png")
    plt.savefig(plot_path, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.close()
    print(f"\nSaved summary plot: {plot_path}")
    print("Calibration spectra conversion completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert calibration target spectra text files for use with simulate_inset_squares.py.")
    parser.add_argument("--base_dir", type=str, default=base_dir, help="Base directory containing the spectra text files.")
    parser.add_argument("--ref_txt_path", type=str, default=ref_txt_path, help="Optional path to a single reference spectra text file.")
    args = parser.parse_args()
    
    convert_calibration_spectra(target_dir=args.base_dir, single_file=args.ref_txt_path)
