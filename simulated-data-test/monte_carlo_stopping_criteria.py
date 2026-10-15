import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from datetime import datetime

from simulate_inset_squares import create_mixture_layer, add_gaussian_noise, load_asd_txt

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.colormaps['okabe_ito'].colors)

# Configurations from Task
WINDOW_SIZES = [3, 5, 9,11,13,15,17, 21, 35, 55, 77, 99]
NUM_MATERIALS = [4, 5, 6, 7, 8, 9, 10, 12, 15,17,19,20,25]
SNRS = [np.inf]#[50, 60, 70, 85, 100, 150, 200, 300, 400, np.inf]
SENSOR_BANDS = [7, 13, 16, 126, 224, 2151]
NUM_MC_ITERATIONS = 10000

SENSORS = {
    7: {"sensor": "rsLandsat8", "index": r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_rsLandsat8.html", "wv": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsLandsat8\S07LSAT8_Wavelengths_Landsat8_(7_bands)_microns.txt"},
    13: {"sensor": "rsSentinel2", "index": r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_rsSentinel2.html", "wv": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsSentinel2\S07SNTL2_Wavelengths_Sentinel2_(13_bands)_microns.txt"},
    16: {"sensor": "rsWorldView3", "index": r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_rsWorldView3.html", "wv": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_rsWorldView3\S07WV3_Wavelengths_WorldView3_(16_bands)_micron.txt"},
    126: {"sensor": "cvHYMAP2014", "index": r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_cvHYMAP2014.html", "wv": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvHYMAP2014\s07_HY14_HyMap2_Alaska2014_wavelengths_126ch.txt"},
    224: {"sensor": "cvAVIRISc2014", "index": r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_cvAVIRISc2014.html", "wv": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvAVIRISc2014\s07_AV14_Wavelengths_in_microns_224_ch_AVIRIS14a.txt"},
    2151: {"sensor": "cvASD", "index": r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_cvASD.html", "wv": r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvASD\s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"}
}

def get_all_material_files(html_path):
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'html.parser')
    paths = []
    for a in soup.find_all('a'):
        if 'Spectrum Data' in a.get_text():
            href = a.get('href')
            base_dir = os.path.dirname(html_path)
            paths.append(os.path.abspath(os.path.join(base_dir, href)))
    return paths

class MaterialCache:
    def __init__(self, sensors_info):
        self.sensors_info = sensors_info
        self.materials_paths = {}
        self.wavelengths = {}
        self.valid_masks = {}
        self.cached_spectra = {}

    def load_sensor(self, bands):
        if bands not in self.materials_paths:
            sensor_info = self.sensors_info[bands]
            self.materials_paths[bands] = get_all_material_files(sensor_info['index'])
            
            wv = load_asd_txt(sensor_info['wv'])
            self.wavelengths[bands] = wv
            self.valid_masks[bands] = ~np.isnan(wv)
            self.cached_spectra[bands] = {}

    def get_random_materials(self, bands, num_mats):
        self.load_sensor(bands)
        
        selected_mats = []
        paths_to_try = random.sample(self.materials_paths[bands], len(self.materials_paths[bands]))
        
        valid_keywords = ['pine', 'fir', 'aspen', 'oak', 'maple', 'grass', 'leaf', 'leaves', 
                          'wood', 'bark', 'soil', 'sand', 'quartz', 'goethite', 'rock', 
                          'mineral', 'meadow', 'conifer', 'lichen', 'moss', 'snow', 'water', 'hyacinth', 'petunia', 'flower']
        
        # Pass 1: Try to get all materials from the "forest/natural" keyword list
        for p in paths_to_try:
            filename = os.path.basename(p).lower()
            if not any(k in filename for k in valid_keywords):
                continue
                
            if p not in self.cached_spectra[bands]:
                spec = load_asd_txt(p)
                self.cached_spectra[bands][p] = spec
            else:
                spec = self.cached_spectra[bands][p]
            
            spec_clean = spec[~np.isnan(spec)]
            if len(spec_clean) == 0:
                continue
                
            spec_max = np.max(spec_clean)
            spec_min = np.min(spec_clean)
            spec_mean = np.mean(spec_clean)
            
            # Reject invalid or highly unrealistic signatures
            if spec_max > 1.05 or spec_min < -0.05:
                continue
            if (spec_max - spec_min) < 0.05:  # Too flat
                continue
            if spec_mean > 0.8:  # Artificially shifted (e.g. +1shf)
                continue
            
            valid_mask = self.valid_masks[bands] & ~np.isnan(spec)
            if np.array_equal(valid_mask, self.valid_masks[bands]):
                selected_mats.append(spec)
                
            if len(selected_mats) == num_mats:
                break
                
        # Pass 2: If we didn't find enough, relax keyword constraint but keep physical realism filters
        if len(selected_mats) < num_mats:
            for p in paths_to_try:
                filename = os.path.basename(p).lower()
                if any(k in filename for k in valid_keywords):
                    continue
                    
                if p not in self.cached_spectra[bands]:
                    spec = load_asd_txt(p)
                    self.cached_spectra[bands][p] = spec
                else:
                    spec = self.cached_spectra[bands][p]
                    
                spec_clean = spec[~np.isnan(spec)]
                if len(spec_clean) == 0:
                    continue
                    
                spec_max = np.max(spec_clean)
                spec_min = np.min(spec_clean)
                spec_mean = np.mean(spec_clean)
                
                # Reject invalid or highly unrealistic signatures
                if spec_max > 1.05 or spec_min < -0.05:
                    continue
                if (spec_max - spec_min) < 0.05:  # Too flat
                    continue
                if spec_mean > 0.8:  # Artificially shifted (e.g. +1shf)
                    continue
                
                valid_mask = self.valid_masks[bands] & ~np.isnan(spec)
                if np.array_equal(valid_mask, self.valid_masks[bands]):
                    selected_mats.append(spec)
                    
                if len(selected_mats) == num_mats:
                    break
                    
        if len(selected_mats) < num_mats:
            raise ValueError(f"Could not find {num_mats} valid materials for sensor with {bands} bands.")
            
        combined_mask = self.valid_masks[bands].copy()
            
        final_mats = [mat[combined_mask] for mat in selected_mats]
        return final_mats, combined_mask

def run_monte_carlo():
    print(f"Running Monte Carlo Simulation with {NUM_MC_ITERATIONS} iterations...")
    
    cache = MaterialCache(SENSORS)
    results = []
    
    successful_runs = 0
    pbar = tqdm(total=NUM_MC_ITERATIONS)
    
    while successful_runs < NUM_MC_ITERATIONS:
        w_size = random.choice(WINDOW_SIZES)
        n_mats = random.choice(NUM_MATERIALS)
        snr = random.choice(SNRS)
        bands = random.choice(SENSOR_BANDS)
        
        try:
            mats, combined_mask = cache.get_random_materials(bands, n_mats)
        except ValueError:
            continue
            
        actual_bands = np.sum(combined_mask)
        # We need at least n_mats + 2 degrees of freedom to observe the stopping condition
        if min(w_size * w_size, actual_bands) < n_mats + 2:
            continue
            
        layer = create_mixture_layer(mats, w_size, variance=0.45, seed=random.randint(0, 999999))
        layer = add_gaussian_noise(layer, snr=snr, seed=random.randint(0, 999999))
        
        pixels = layer.reshape(actual_bands, -1)
        num_em_eval = min(n_mats + 5, w_size * w_size, actual_bands)
        
        try:
            em, vol, h, indices = maximumDistance_volumes(pixels, num_em_eval)
        except Exception:
            continue
            
        results.append({
            'iteration': successful_runs,
            'window_size': w_size,
            'num_materials': n_mats,
            'snr': snr,
            'sensor_bands': bands,
            'actual_bands': actual_bands,
            'num_em_eval': num_em_eval,
            'volumes': vol.tolist(),
            'heights': h.tolist()
        })
        successful_runs += 1
        pbar.update(1)
        
    pbar.close()
    
    df_results = pd.DataFrame(results)
    
    output_dir = r"C:\satelliteImagery\simulated\monte_carlo_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"mc_results_{timestamp}.csv")
    
    df_results.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    return df_results, output_dir, timestamp

def plot_visualizations(df, output_dir, timestamp):
    print("Generating visualizations...")
    unique_sensors = sorted(df['sensor_bands'].unique())
    for bands in unique_sensors:
        sensor_df = df[df['sensor_bands'] == bands]
        
        bands_dir = os.path.join(output_dir, f"{bands} bands")
        os.makedirs(bands_dir, exist_ok=True)
        
        # --- Plot 1: SNR impact for this sensor ---
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        snr_groups = sensor_df.groupby('snr')
        
        for snr, group in snr_groups:
            snr_label = "Inf" if pd.isna(snr) or np.isinf(snr) else str(snr)
            
            all_k_minus_m = []
            all_vols = []
            all_heights = []
            
            for _, row in group.iterrows():
                m = row['num_materials']
                vols = np.array(row['volumes'])
                heights = np.array(row['heights'])
                k = np.arange(1, len(vols) + 1)
                
                k_minus_m = k - m
                all_k_minus_m.extend(k_minus_m)
                all_vols.extend(vols)
                all_heights.extend(heights)
                
            if len(all_k_minus_m) > 0:
                df_plot = pd.DataFrame({'k_minus_m': all_k_minus_m, 'vol': all_vols, 'height': all_heights})
                df_plot = df_plot.replace([-np.inf, np.inf], np.nan).dropna()
                
                mean_df = df_plot.groupby('k_minus_m').mean().reset_index()
                mean_df = mean_df.sort_values('k_minus_m')
                
                axes[0].plot(mean_df['k_minus_m'], mean_df['vol'], marker='.', label=f'SNR {snr_label}')
                axes[1].plot(mean_df['k_minus_m'], mean_df['height'], marker='.', label=f'SNR {snr_label}')
                
        axes[0].set_title(f'Average Log Volume vs (k - M) by SNR ({bands} bands)')
        axes[0].set_xlabel('Endmember Index Relative to True Number of Materials (k - M)')
        axes[0].set_ylabel('Mean Log Volume')
        axes[0].axvline(0, color='red', linestyle='--', label='True M')
        axes[0].legend()
        axes[0].grid(True, alpha=0.5)
        
        axes[1].set_title(f'Average Height vs (k - M) by SNR ({bands} bands)')
        axes[1].set_xlabel('Endmember Index Relative to True Number of Materials (k - M)')
        axes[1].set_ylabel('Mean Height (Orthogonal Distance)')
        axes[1].axvline(0, color='red', linestyle='--', label='True M')
        axes[1].set_yscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(bands_dir, f"mc_snr_impact_bands_{bands}_{timestamp}.png"), dpi=300)
        plt.close()
        
        # --- Plot 2: Window Size impact for this sensor (nested by SNR) ---
        unique_snrs = sorted(sensor_df['snr'].unique())
        for snr in unique_snrs:
            snr_df = sensor_df[sensor_df['snr'] == snr]
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            ws_groups = snr_df.groupby('window_size')
            
            for ws, group in ws_groups:
                all_k_minus_m = []
                all_vols = []
                all_heights = []
                
                for _, row in group.iterrows():
                    m = row['num_materials']
                    vols = np.array(row['volumes'])
                    heights = np.array(row['heights'])
                    k = np.arange(1, len(vols) + 1)
                    
                    k_minus_m = k - m
                    all_k_minus_m.extend(k_minus_m)
                    all_vols.extend(vols)
                    all_heights.extend(heights)
                    
                if len(all_k_minus_m) > 0:
                    df_plot = pd.DataFrame({'k_minus_m': all_k_minus_m, 'vol': all_vols, 'height': all_heights})
                    df_plot = df_plot.replace([-np.inf, np.inf], np.nan).dropna()
                    
                    mean_df = df_plot.groupby('k_minus_m').mean().reset_index()
                    mean_df = mean_df.sort_values('k_minus_m')
                    
                    axes[0].plot(mean_df['k_minus_m'], mean_df['vol'], marker='.', label=f'WS {ws}x{ws}')
                    axes[1].plot(mean_df['k_minus_m'], mean_df['height'], marker='.', label=f'WS {ws}x{ws}')
                    
            snr_label = "Inf" if pd.isna(snr) or np.isinf(snr) else str(snr)
            
            axes[0].set_title(f'Average Log Volume vs (k - M) by Window Size (SNR={snr_label}, {bands} bands)')
            axes[0].set_xlabel('Endmember Index Relative to True Number of Materials (k - M)')
            axes[0].set_ylabel('Mean Log Volume')
            axes[0].axvline(0, color='red', linestyle='--', label='True M')
            axes[0].legend()
            axes[0].grid(True, alpha=0.5)
            
            axes[1].set_title(f'Average Height vs (k - M) by Window Size (SNR={snr_label}, {bands} bands)')
            axes[1].set_xlabel('Endmember Index Relative to True Number of Materials (k - M)')
            axes[1].set_ylabel('Mean Height (Orthogonal Distance)')
            axes[1].axvline(0, color='red', linestyle='--', label='True M')
            axes[1].set_yscale('log')
            axes[1].legend()
            axes[1].grid(True, alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(os.path.join(bands_dir, f"mc_window_size_impact_bands_{bands}_snr_{snr_label}_{timestamp}.png"), dpi=300)
            plt.close()
    
    print("Done!")

def extract_curve_features(volumes, heights, snr, window_size):
    """
    Extract features from volume and height curves for endmember count prediction.
    
    Features capture the shape of the curves — ratios between consecutive values,
    second differences (curvature), and the index of maximum change — which encode
    where the inflection point occurs (i.e., where additional endmembers stop
    explaining real spectral variance).
    """
    vol = np.array(volumes, dtype=np.float64)
    ht = np.array(heights, dtype=np.float64)
    
    # Replace non-finite values for safe computation
    vol = np.where(np.isfinite(vol), vol, np.nan)
    ht = np.where(np.isfinite(ht), ht, np.nan)
    
    features = {}
    
    # SNR feature (log-transform; treat inf as a large finite value)
    features['log_snr'] = np.log10(snr) if np.isfinite(snr) else 4.0
    features['window_size'] = window_size
    
    # --- Volume curve features ---
    # First differences of log-volume (rate of volume growth per added EM)
    vol_diff = np.diff(vol)
    vol_diff_clean = vol_diff[np.isfinite(vol_diff)]
    
    if len(vol_diff_clean) > 0:
        features['vol_diff_mean'] = np.nanmean(vol_diff_clean)
        features['vol_diff_std'] = np.nanstd(vol_diff_clean)
        # Index of maximum volume jump (1-indexed to match endmember count)
        features['vol_max_jump_idx'] = np.nanargmax(np.abs(vol_diff_clean)) + 1
    else:
        features['vol_diff_mean'] = 0.0
        features['vol_diff_std'] = 0.0
        features['vol_max_jump_idx'] = 1
    
    # Second differences (curvature — where the growth rate changes most)
    vol_diff2 = np.diff(vol, n=2)
    vol_diff2_clean = vol_diff2[np.isfinite(vol_diff2)]
    
    if len(vol_diff2_clean) > 0:
        features['vol_curvature_max_idx'] = np.nanargmax(np.abs(vol_diff2_clean)) + 1
        features['vol_curvature_max'] = np.nanmax(np.abs(vol_diff2_clean))
    else:
        features['vol_curvature_max_idx'] = 1
        features['vol_curvature_max'] = 0.0
    
    # Ratios of consecutive volume differences (captures where growth saturates)
    if len(vol_diff_clean) > 1:
        ratios = []
        for i in range(1, len(vol_diff_clean)):
            if np.abs(vol_diff_clean[i-1]) > 1e-30:
                ratios.append(vol_diff_clean[i] / vol_diff_clean[i-1])
        if ratios:
            ratios = np.array(ratios)
            ratios_clean = ratios[np.isfinite(ratios)]
            if len(ratios_clean) > 0:
                features['vol_ratio_min'] = np.nanmin(ratios_clean)
                features['vol_ratio_min_idx'] = np.nanargmin(ratios_clean) + 1
            else:
                features['vol_ratio_min'] = 1.0
                features['vol_ratio_min_idx'] = 1
        else:
            features['vol_ratio_min'] = 1.0
            features['vol_ratio_min_idx'] = 1
    else:
        features['vol_ratio_min'] = 1.0
        features['vol_ratio_min_idx'] = 1
    
    # --- Height curve features ---
    ht_clean = ht[np.isfinite(ht)]
    
    if len(ht_clean) > 1:
        ht_diff = np.diff(ht_clean)
        features['ht_diff_mean'] = np.nanmean(ht_diff)
        features['ht_diff_median'] = np.nanmedian(ht_diff)
        features['ht_diff_std'] = np.nanstd(ht_diff)
        features['ht_diff_min'] = np.nanmin(ht_diff)
        features['ht_diff_max'] = np.nanmax(ht_diff)
        features['ht_max_drop_idx'] = np.nanargmin(ht_diff) + 1
        
        # Overall linear slope of the height trend
        x_vals = np.arange(len(ht_clean))
        slope, _ = np.polyfit(x_vals, ht_clean, 1)
        features['ht_slope_overall'] = slope
        
        # Tail slope (last 5 elements if possible) to capture the noise floor trend
        tail_len = min(5, len(ht_diff))
        features['ht_slope_tail'] = np.nanmean(ht_diff[-tail_len:]) if tail_len > 0 else 0.0
        
        # Log-height ratios (height typically decays; large drop = meaningful EM)
        log_ht = np.log10(np.maximum(ht_clean, 1e-30))
        log_ht_diff = np.diff(log_ht)
        features['log_ht_max_drop_idx'] = np.nanargmin(log_ht_diff) + 1
        features['log_ht_max_drop_val'] = np.nanmin(log_ht_diff)
    else:
        features['ht_diff_mean'] = 0.0
        features['ht_diff_median'] = 0.0
        features['ht_diff_std'] = 0.0
        features['ht_diff_min'] = 0.0
        features['ht_diff_max'] = 0.0
        features['ht_max_drop_idx'] = 1
        features['ht_slope_overall'] = 0.0
        features['ht_slope_tail'] = 0.0
        features['log_ht_max_drop_idx'] = 1
        features['log_ht_max_drop_val'] = 0.0
    
    # Number of evaluated endmembers (context for index features)
    features['num_evaluated'] = len(vol)
    
    return features

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor

class FeatureSelectorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, base_model):
        self.base_model = base_model
        
    def fit(self, X, y):
        X_clean = X.drop(columns=['__raw_curve_volume', '__raw_curve_height'])
        self.base_model.fit(X_clean, y)
        return self
        
    def predict(self, X):
        X_clean = X.drop(columns=['__raw_curve_volume', '__raw_curve_height'])
        return self.base_model.predict(X_clean)


def train_endmember_regressors(df, output_dir, timestamp):
    """
    Train one regression model per sensor band count to predict the true
    number of endmembers (num_materials) from volume/height curve features and SNR.
    
    Uses HistGradientBoostingRegressor for robustness to heterogeneous feature
    scales and missing values without requiring normalization.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import ast
    import json
    
    print("Training endmember count regressors...")
    
    unique_sensors = sorted(df['sensor_bands'].unique())
    model_reports = {}
    
    for bands in unique_sensors:
        sensor_df = df[df['sensor_bands'] == bands].copy()
        
        if len(sensor_df) < 10:
            print(f"  Skipping {bands} bands: only {len(sensor_df)} samples")
            continue
            
        bands_dir = os.path.join(output_dir, f"{bands} bands")
        os.makedirs(bands_dir, exist_ok=True)
        
        # Build feature matrix
        feature_rows = []
        targets = []
        
        for _, row in sensor_df.iterrows():
            vols = row['volumes']
            hts = row['heights']
            m = row['num_materials']
            
            # Handle string-serialized lists from CSV reload
            if isinstance(vols, str):
                vols = eval(vols, {"nan": np.nan, "inf": np.inf})
            if isinstance(hts, str):
                hts = eval(hts, {"nan": np.nan, "inf": np.inf})
            
            # Implement growing sampling window to simulate iterative extraction.
            # Start at a minimum of 3 endmembers up to the full curve length.
            max_k = len(vols)
            for k in range(3, max_k + 1):
                partial_vols = vols[:k]
                partial_hts = hts[:k]
                
                feat = extract_curve_features(partial_vols, partial_hts, row['snr'], row['window_size'])
                feat['__raw_curve_volume'] = np.array(partial_vols, dtype=np.float64)
                feat['__raw_curve_height'] = np.array(partial_hts, dtype=np.float64)
                feature_rows.append(feat)
                targets.append(m)
        
        X = pd.DataFrame(feature_rows).fillna(0.0)
        y = np.array(targets)
        
        feature_names = [c for c in X.columns if not c.startswith('__raw_curve_')]
        
        # Define models to train
        models_to_run = {
            'hgb': FeatureSelectorWrapper(HistGradientBoostingRegressor(max_iter=200, max_depth=6, learning_rate=0.1, random_state=42)),
            'linear': FeatureSelectorWrapper(make_pipeline(StandardScaler(), LinearRegression()))
        }
        
        for model_key, model in models_to_run.items():
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()
            cv_mae_std = cv_scores.std()
            
            cv_rmse_scores = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
            cv_rmse = -cv_rmse_scores.mean()
            
            # Fit on all data for feature importances and final model
            model.fit(X, y)
            
            # Save the trained model for future use
            import joblib
            model_path = os.path.join(bands_dir, f"endmember_regressor_{model_key}_bands_{bands}_{timestamp}.joblib")
            joblib.dump(model, model_path)
            
            y_pred = model.predict(X)
            train_mae = mean_absolute_error(y, y_pred)
            
            # Round predictions to nearest integer for classification-like accuracy
            y_pred_round = np.round(y_pred).astype(int)
            exact_accuracy = np.mean(y_pred_round == y)
            within_1_accuracy = np.mean(np.abs(y_pred_round - y) <= 1)
            
            sensor_name = SENSORS[bands]['sensor']
            
            if model_key not in model_reports:
                model_reports[model_key] = {}
                
            model_reports[model_key][bands] = {
                'sensor': sensor_name,
                'bands': bands,
                'n_samples': len(y),
                'cv_mae': cv_mae,
                'cv_mae_std': cv_mae_std,
                'cv_rmse': cv_rmse,
                'train_mae': train_mae,
                'exact_accuracy': exact_accuracy,
                'within_1_accuracy': within_1_accuracy,
                'feature_names': feature_names,
                'model_path': model_path
            }
            
            print(f"  {sensor_name} ({bands} bands) [{model_key.upper()}]: "
                  f"CV MAE={cv_mae:.2f}±{cv_mae_std:.2f}, "
                  f"CV RMSE={cv_rmse:.2f}, "
                  f"Exact={exact_accuracy:.1%}, "
                  f"±1={within_1_accuracy:.1%}")
            
            # --- Plot: Predicted vs Actual ---
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            # Subplot 1: Predicted vs Actual scatter
            ax = axes[0]
            ax.scatter(y, y_pred, alpha=0.4, edgecolors='k', linewidth=0.3, s=20)
            lims = [min(y.min(), y_pred.min()) - 1, max(y.max(), y_pred.max()) + 1]
            ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')
            ax.set_xlabel('True Number of Materials')
            ax.set_ylabel('Predicted Number of Materials')
            ax.set_title(f'{model_key.upper()} Predicted vs Actual ({sensor_name}, {bands} bands)')
            ax.legend()
            ax.grid(True, alpha=0.5)
            
            # Subplot 2: Residual distribution
            ax = axes[1]
            residuals = y_pred - y
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--')
            ax.set_xlabel('Residual (Predicted - Actual)')
            ax.set_ylabel('Count')
            ax.set_title(f'Residual Distribution (MAE={cv_mae:.2f})')
            ax.grid(True, alpha=0.5)
            
            # Subplot 3: Feature importances
            from sklearn.inspection import permutation_importance
            ax = axes[2]
            # Some models use raw curves and some ignore them, but X has all columns
            result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            importances = result.importances_mean
            sorted_idx = np.argsort(importances)
            
            # Filter out zero-importance features to keep the plot clean, especially raw curves
            valid_idx = [i for i in sorted_idx if importances[i] > 1e-4 and not X.columns[i].startswith('__raw')]
            if len(valid_idx) == 0:
                valid_idx = sorted_idx[-5:] # fallback to top 5 if all are 0
                
            ax.barh(np.array(X.columns)[valid_idx], importances[valid_idx])
            ax.set_xlabel('Permutation Importance')
            ax.set_title(f'Feature Importances')
            ax.grid(True, alpha=0.5, axis='x')
            
            plt.tight_layout()
            plt.savefig(os.path.join(bands_dir, f"mc_{model_key}_regression_bands_{bands}_{timestamp}.png"), dpi=300)
            plt.close()
            
        # --- Train Median Slope Multiplier ---
        print(f"  Training Median Slope Multiplier for {bands} bands...")
        best_p = 0.1
        best_mae = float('inf')
        p_candidates = np.linspace(0.001, 0.5, 100)
        
        for p in p_candidates:
            maes = []
            for _, row in sensor_df.iterrows():
                hts = row['heights']
                if isinstance(hts, str):
                    hts = eval(hts, {"nan": np.nan, "inf": np.inf})
                m = row['num_materials']
                
                pred_m = len(hts)
                diffs = -np.diff(hts) # positive drops
                for k in range(3, len(hts) + 1):
                    current_drop = diffs[k-2]
                    if k-2 > 0:
                        median_drop = np.nanmedian(diffs[:k-2])
                        if current_drop < p * median_drop:
                            pred_m = k
                            break
                maes.append(abs(pred_m - m))
                
            mean_mae = np.mean(maes)
            if mean_mae < best_mae:
                best_mae = mean_mae
                best_p = p
                
        # Save the best multiplier
        multiplier_path = os.path.join(bands_dir, f"slope_multiplier_bands_{bands}_{timestamp}.json")
        with open(multiplier_path, 'w') as f:
            json.dump({'best_multiplier': best_p, 'train_mae': best_mae}, f)
        print(f"  Best Median Slope Multiplier for {bands} bands: {best_p:.4f} (MAE: {best_mae:.2f})")
    
    # Save summary report
    report_path = os.path.join(output_dir, f"mc_regression_report_{timestamp}.json")
    # Convert numpy types for JSON serialization
    serializable = {}
    for model_key, rep in model_reports.items():
        serializable[model_key] = {}
        for k, v in rep.items():
            serializable[model_key][str(k)] = {
                key: (
                    float(val) if isinstance(val, (np.floating, float))
                    else int(val) if isinstance(val, (np.integer, int))
                    else val
                )
                for key, val in v.items()
            }
    with open(report_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved regression report to {report_path}")
    
    return model_reports

if __name__ == "__main__":
    df, out_dir, ts = run_monte_carlo()
    plot_visualizations(df, out_dir, ts)
    train_endmember_regressors(df, out_dir, ts)
