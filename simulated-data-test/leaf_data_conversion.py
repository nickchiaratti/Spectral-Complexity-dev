import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Load true wavelengths from the reference text file
ref_txt_path = r"C:\satelliteImagery\ground_truth\NickLeaves_20261001\2\txt\1790879216-00.asd.rad.pco.txt"
wavelengths = []
with open(ref_txt_path, 'r') as f:
    start_reading = False
    for line in f:
        if line.startswith("Wavelength"):
            start_reading = True
            continue
        if start_reading and line.strip():
            wavelengths.append(float(line.split()[0]))

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

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file == "results.npy":
            npy_path = os.path.join(root, file)
            print(f"Processing {npy_path}...")
            results = np.load(npy_path, allow_pickle=True).item()
            print(results.keys())
            
            # `w` from .npy is not wavelength, skip using it
            
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
plt.savefig(os.path.join(base_dir, "Healthy_vs_Diseased_Leaf_Spectral_Reflectance.png"),dpi=500,pad_inches=0,bbox_inches='tight')


