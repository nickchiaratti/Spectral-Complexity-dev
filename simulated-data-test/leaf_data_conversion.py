import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load true wavelengths from the reference text file (or 2-column spectrum text file)
ref_txt_path = r"C:\satelliteImagery\ground_truth\NickLeaves_20261001\2\txt\1790879216-00.asd.rad.pco.txt"
base_dir = "C:/satelliteImagery/ground_truth/NickLeaves_20261001"

def load_asd_txt(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[1:]:
        if not line.strip():
            continue
        val = float(line.strip())
        if val < -1.0: # typically -1.23e34
            data.append(np.nan)
        else:
            data.append(val)
    return np.array(data, dtype=np.float32)

def load_spectrum_txt(filepath):
    """
    Loads 2-column spectrum text file (wavelength [nm], reflectance).
    Handles headers, scientific notation, different significant digits, and tab/space delimiters.
    """
    wavelengths, reflectances = [], []
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
                continue
    w_arr = np.array(wavelengths, dtype=np.float64)
    r_arr = np.array(reflectances, dtype=np.float32)
    if len(w_arr) == 0:
        raise ValueError(f"No numeric 2-column spectral data found in {filepath}")
    idx = np.argsort(w_arr)
    return w_arr[idx], r_arr[idx]

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

# Check if base_dir contains results.npy (leaf ASD dataset mode)
npy_files = []
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "results.npy":
            npy_files.append((root, os.path.join(root, file)))

if npy_files:
    # --- Mode 1: Leaf results.npy dataset ---
    wavelengths = []
    with open(ref_txt_path, 'r') as f:
        start_reading = False
        for line in f:
            if line.startswith("Wavelength"):
                start_reading = True
                continue
            if start_reading and line.strip():
                wavelengths.append(float(line.split()[0]))

    for root, npy_path in npy_files:
        print(f"Processing {npy_path}...")
        results = np.load(npy_path, allow_pickle=True).item()
        print(results.keys())
        
        df = pd.DataFrame({
            'wavelength': wavelengths,
            'DHR_mean': results['DHR_mean'],
            'DHR_stdev': results['DHR_stdev']
        })
        
        parent_dir_name = os.path.basename(root)
        
        # Save original ASD data
        csv_path = os.path.join(base_dir, f"cvASD_{parent_dir_name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path}")
        plt.plot(wavelengths, results['DHR_mean'], label=parent_dir_name)
        
        # Resample for simulated sensors
        w_nm = np.array(wavelengths)
        dhr = np.array(results['DHR_mean'])
        
        for sensor, paths in SENSORS.items():
            s_wav = load_asd_txt(paths['wav']) * 1000.0  # Convert to nm
            s_fwhm = load_asd_txt(paths['fwhm']) * 1000.0 # Convert to nm
            
            resampled_dhr = []
            for cw, fwhm in zip(s_wav, s_fwhm):
                if np.isnan(cw) or np.isnan(fwhm) or fwhm <= 0:
                    resampled_dhr.append(np.nan)
                    continue
                
                srf = np.exp(-4 * np.log(2) * ((w_nm - cw)**2) / (fwhm**2))
                r_val = np.sum(dhr * srf) / np.sum(srf)
                resampled_dhr.append(r_val)
                
            df_resampled = pd.DataFrame({
                'wavelength': s_wav,
                'DHR_mean': resampled_dhr
            })
            
            r_csv_path = os.path.join(base_dir, f"{sensor}_{parent_dir_name}.csv")
            df_resampled.to_csv(r_csv_path, index=False)
            print(f"Saved {r_csv_path}")

    plt.legend(loc='best')
    plt.grid(True)
    plt.ylabel("Reflectance")
    plt.xlabel("Wavelength (nm)")
    plt.title("Healthy vs Diseased Leaf Spectral Reflectance")
    plt.savefig(os.path.join(base_dir, "Healthy_vs_Diseased_Leaf_Spectral_Reflectance.png"), dpi=500, pad_inches=0, bbox_inches='tight')
    plt.close()

else:
    # --- Mode 2: 2-column text spectra files (e.g. calibration targets) ---
    import glob
    if ref_txt_path and os.path.isfile(ref_txt_path) and not ref_txt_path.endswith(".asd.rad.pco.txt"):
        txt_files = [ref_txt_path]
    else:
        all_txt = glob.glob(os.path.join(base_dir, "*.txt"))
        txt_files = [f for f in all_txt if not os.path.basename(f).startswith(("S07", "s07"))]
        txt_files.sort()

    print(f"Processing {len(txt_files)} 2-column spectrum text file(s) in {base_dir}...")
    
    # Preload sensor bands
    sensor_bands = {}
    for sensor, paths in SENSORS.items():
        s_wav = load_asd_txt(paths['wav']) * 1000.0
        s_fwhm = load_asd_txt(paths['fwhm']) * 1000.0
        sensor_bands[sensor] = (s_wav, s_fwhm)

    for filepath in txt_files:
        target_name = os.path.splitext(os.path.basename(filepath))[0]
        print(f"Processing {filepath} ({target_name})...")
        w_raw, r_raw = load_spectrum_txt(filepath)
        
        # Dense 1nm grid for accurate SRF convolution
        w_nm = np.arange(int(np.ceil(w_raw.min())), int(np.floor(w_raw.max())) + 1, 1.0)
        dhr = np.interp(w_nm, w_raw, r_raw).astype(np.float32)
        
        # Save standard 2151-channel cvASD data (350-2500 nm) for simulate_inset_squares.py
        asd_dhr = np.interp(ASD_WAVELENGTHS, w_raw, r_raw).astype(np.float32)
        df_asd = pd.DataFrame({
            'wavelength': ASD_WAVELENGTHS,
            'DHR_mean': asd_dhr,
            'DHR_stdev': np.zeros_like(asd_dhr, dtype=np.float32)
        })
        asd_csv_path = os.path.join(base_dir, f"cvASD_{target_name}.csv")
        df_asd.to_csv(asd_csv_path, index=False)
        print(f"Saved {asd_csv_path}")
        plt.plot(w_nm, dhr, label=target_name)
        
        # Resample for simulated sensors
        for sensor, (s_wav, s_fwhm) in sensor_bands.items():
            resampled_dhr = []
            for cw, fwhm in zip(s_wav, s_fwhm):
                if np.isnan(cw) or np.isnan(fwhm) or fwhm <= 0:
                    resampled_dhr.append(np.nan)
                    continue
                srf = np.exp(-4 * np.log(2) * ((w_nm - cw)**2) / (fwhm**2))
                sum_srf = np.sum(srf)
                r_val = float(np.sum(dhr * srf) / sum_srf) if sum_srf > 0 else np.nan
                resampled_dhr.append(r_val)
                
            df_resampled = pd.DataFrame({
                'wavelength': s_wav,
                'DHR_mean': resampled_dhr
            })
            r_csv_path = os.path.join(base_dir, f"{sensor}_{target_name}.csv")
            df_resampled.to_csv(r_csv_path, index=False)
            print(f"Saved {r_csv_path}")

    plt.legend(loc='best', fontsize='small')
    plt.grid(True)
    plt.ylabel("Reflectance")
    plt.xlabel("Wavelength (nm)")
    plt.title("Calibration Targets Spectral Reflectance")
    plt.savefig(os.path.join(base_dir, "Calibration_Targets_Spectral_Reflectance.png"), dpi=500, pad_inches=0, bbox_inches='tight')
    plt.close()



