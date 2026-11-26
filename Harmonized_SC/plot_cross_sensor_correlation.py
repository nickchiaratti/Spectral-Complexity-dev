import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mstyle
if not hasattr(mstyle, 'core'):
    mstyle.core = mstyle
from scipy import stats
from sklearn.linear_model import RANSACRegressor
from datetime import datetime, timezone
import scienceplots
plt.style.use(['science', 'ieee'])

# Maximum allowable time difference to consider frames a "Match"
MATCH_TOLERANCE_DAYS = 10.0 
MINIMUM_COMMONALITY = 0.5

script_dir = os.path.dirname(os.path.abspath(__file__))
candidate_media = os.path.abspath(os.path.join(script_dir, "..", "media"))
PRIMARY_OUTPUT_DIR = r"C:\satelliteImagery\HLST30"
MEDIA_DIR = PRIMARY_OUTPUT_DIR if os.path.exists(PRIMARY_OUTPUT_DIR) else (candidate_media if os.path.exists(candidate_media) else "./media")

PRIMARY_PATH = r"C:\satelliteImagery\HLST30\HLST_Rochesterv2.h5"
FALLBACK_PATH = r"F:\Resilio\IMGS 890 Research\JSTARS-Spectral-Complexity\source_data\HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5"
FILEPATH = PRIMARY_PATH if os.path.exists(PRIMARY_PATH) else FALLBACK_PATH


class HLST_Statistical_Extractor:
    def __init__(self, filepath, pair_name, sensor1_list, sensor2_list, match_tolerance_days=5.0, min_commonality=None, min_swath=None):
        self.h5 = h5py.File(filepath, 'r')
        self.pair_name = pair_name
        self.sensor1_list = sensor1_list
        self.sensor2_list = sensor2_list
        self.match_tolerance_days = match_tolerance_days
        self.min_commonality = min_commonality
        self.min_swath = min_swath
        
        harm_path = '/HDFEOS/GRIDS/HARMONIZED/Data Fields'
        self.harm_grp = self.h5[harm_path]
        
        self.ds_zscore = self.harm_grp['sliding_volume_z_score']
        self.ds_mask = self.harm_grp['common_mask']
        self.ds_vol = self.harm_grp['sliding_volume_map']
        
        self.times = self.ds_mask.attrs['acquisition_time']
        raw_spacecraft = self.ds_mask.attrs['source_spacecraft']
        self.spacecrafts = [g.decode('utf-8') if isinstance(g, bytes) else str(g) for g in raw_spacecraft]
        self.num_frames = len(self.times)
        self.land_mask = None
        self.common_water_mask = None

    def _get_land_mask(self):
        if self.land_mask is None:
            if '/HDFEOS/GRIDS/HLSL30/Data Fields/Fmask' in self.h5:
                fmask_l = self.h5['/HDFEOS/GRIDS/HLSL30/Data Fields/Fmask']
                fmask_s = self.h5['/HDFEOS/GRIDS/HLSS30/Data Fields/Fmask']
                shape = fmask_l.shape[1:]
                water_sum = np.zeros(shape, dtype=np.int32)
                valid_sum = np.zeros(shape, dtype=np.int32)
                for fmask in [fmask_l, fmask_s]:
                    for i in range(0, fmask.shape[0], 10):
                        fm = fmask[i, :, :]
                        val = (fm != 255)
                        water = ((fm & 32) > 0) & val
                        water_sum[val] += water[val]
                        valid_sum[val] += 1
                water_ratio = np.zeros(shape, dtype=np.float32)
                nz = valid_sum > 0
                water_ratio[nz] = water_sum[nz] / valid_sum[nz]
                self.common_water_mask = (water_ratio >= 0.33)
                self.land_mask = ~self.common_water_mask
            else:
                self.land_mask = np.zeros(self.ds_mask.shape[1:], dtype=bool)
                for i in range(0, self.num_frames, 10):
                    self.land_mask |= (self.ds_mask[i] == 0)
                self.common_water_mask = ~self.land_mask
        return self.land_mask

    def extract_matched_distributions(self):
        is_tan_1 = any("TANAGER" in s.upper() for s in self.sensor1_list)
        is_tan_2 = any("TANAGER" in s.upper() for s in self.sensor2_list)
        is_tanager_cross = (is_tan_1 ^ is_tan_2)
        
        s1_meta = [(t, self.times[t]) for t in range(self.num_frames) if self.spacecrafts[t] in self.sensor1_list]
        s2_meta = [(t, self.times[t]) for t in range(self.num_frames) if self.spacecrafts[t] in self.sensor2_list]
        
        matched_pairs = []
        
        is_self = (set(self.sensor1_list) == set(self.sensor2_list))
        
        if is_tanager_cross:
            # Tanager data determines the subset of HLS frames to extract for comparison
            tan_meta = s1_meta if is_tan_1 else s2_meta
            hls_meta = s2_meta if is_tan_1 else s1_meta
            
            land_mask = self._get_land_mask()
            used_hls = set()
            
            for idx_tan, time_tan in tan_meta:
                vol_tan = self.ds_vol[idx_tan, :, :]
                mask_tan = self.ds_mask[idx_tan, :, :]
                
                # Check Tanager swath validity over land
                swath_tan = (~np.isnan(vol_tan)) & (vol_tan > 0)
                land_swath = swath_tan & land_mask
                n_land_swath = np.sum(land_swath)
                n_valid_tan = np.sum((mask_tan == 0) & land_swath)
                
                swath_valid_ratio = n_valid_tan / n_land_swath if n_land_swath > 0 else 0
                if self.min_swath is not None and swath_valid_ratio < self.min_swath:
                    continue
                    
                # Find matching HLS frame
                best_hls = None
                min_diff = float('inf')
                for idx_hls, time_hls in hls_meta:
                    if idx_hls in used_hls: continue
                    diff = abs(time_tan - time_hls)
                    if diff < min_diff and diff <= (self.match_tolerance_days * 24 * 3600):
                        min_diff = diff
                        best_hls = idx_hls
                        
                if best_hls is not None:
                    used_hls.add(best_hls)
                    if is_tan_1:
                        matched_pairs.append({'idx_1': idx_tan, 'idx_2': best_hls, 'is_tan_cross': True})
                    else:
                        matched_pairs.append({'idx_1': best_hls, 'idx_2': idx_tan, 'is_tan_cross': True})
        elif is_self:
            # Strictly sequential chronological pairs (t_i -> t_{i+1}) without axis duplication
            meta = sorted(s1_meta, key=lambda x: x[1])
            max_dt_sec = self.match_tolerance_days * 24 * 3600 if self.match_tolerance_days is not None else float('inf')
            for i in range(len(meta) - 1):
                idx_1, time_1 = meta[i]
                idx_2, time_2 = meta[i+1]
                diff = time_2 - time_1
                if diff <= max_dt_sec:
                    matched_pairs.append({
                        'idx_1': idx_1,
                        'idx_2': idx_2,
                        'is_tan_cross': False
                    })
        else:
            used_s2 = set()
            for idx_1, time_1 in s1_meta:
                best_s2 = None
                min_diff = float('inf')
                
                for idx_2, time_2 in s2_meta:
                    if idx_1 == idx_2: continue # Prevent self-matching the identical frame
                    if idx_2 in used_s2: continue
                    diff = abs(time_1 - time_2)
                    if diff < min_diff and diff <= (self.match_tolerance_days * 24 * 3600):
                        min_diff = diff
                        best_s2 = idx_2
                        
                if best_s2 is not None:
                    used_s2.add(best_s2)
                    matched_pairs.append({
                        'idx_1': idx_1,
                        'idx_2': best_s2,
                        'is_tan_cross': False
                    })
                
        z_1_global, z_2_global = [], []
        accepted_pairs = 0
        
        for pair in matched_pairs:
            mask_1 = self.ds_mask[pair['idx_1'], :, :]
            mask_2 = self.ds_mask[pair['idx_2'], :, :]
            vol_1 = self.ds_vol[pair['idx_1'], :, :]
            vol_2 = self.ds_vol[pair['idx_2'], :, :]
            
            if pair.get('is_tan_cross', False):
                # Tanager cross-sensor comparison:
                # Tanager common_mask determines the subset of HLS footprints to extract
                is_tan_1 = any("TANAGER" in s.upper() for s in self.sensor1_list)
                mask_tan = mask_1 if is_tan_1 else mask_2
                vol_tan = vol_1 if is_tan_1 else vol_2
                mask_hls = mask_2 if is_tan_1 else mask_1
                vol_hls = vol_2 if is_tan_1 else vol_1
                
                # Extents determined by Tanager common_mask over land
                land_mask = self._get_land_mask()
                tan_subset = (mask_tan == 0) & land_mask & (vol_tan > 0) & (~np.isnan(vol_tan))
                hls_valid_sub = tan_subset & (mask_hls == 0) & (vol_hls > 0) & (~np.isnan(vol_hls))
                
                if np.sum(hls_valid_sub) < 10:
                    continue
                    
                # z-score normalize to the subset of pixels for both sensors
                log_hls_sub = np.log(vol_hls[hls_valid_sub])
                mu_hls = np.mean(log_hls_sub)
                std_hls = np.std(log_hls_sub)
                z_hls = (np.log(vol_hls) - mu_hls) / std_hls
                
                log_tan_sub = np.log(vol_tan[tan_subset])
                mu_tan = np.mean(log_tan_sub)
                std_tan = np.std(log_tan_sub)
                z_tan = (np.log(vol_tan) - mu_tan) / std_tan
                
                # Spatially equivalent unmasked pixels (excluding persistent water)
                joint_mask = (mask_hls == 0) & (mask_tan == 0) & land_mask & (vol_hls > 0) & (vol_tan > 0) & \
                             (~np.isnan(vol_hls)) & (~np.isnan(vol_tan))
                             
                intersect_1_z = z_tan[joint_mask] if is_tan_1 else z_hls[joint_mask]
                intersect_2_z = z_hls[joint_mask] if is_tan_1 else z_tan[joint_mask]
            else:
                land_mask = self._get_land_mask()
                if self.min_commonality is not None:
                    # Exclude the common water mask from the MINIMUM_COMMONALITY test
                    valid_1 = (mask_1 == 0) & land_mask
                    valid_2 = (mask_2 == 0) & land_mask
                    n1 = np.sum(valid_1)
                    n2 = np.sum(valid_2)
                    if n1 == 0 or n2 == 0:
                        continue
                    inter = np.sum(valid_1 & valid_2)
                    commonality = min(inter / n1, inter / n2)
                    if commonality < self.min_commonality:
                        continue
                        
                z_1 = self.ds_zscore[pair['idx_1'], :, :]
                z_2 = self.ds_zscore[pair['idx_2'], :, :]
                
                # Exclude persistent water mask from pixel comparisons
                joint_mask = (mask_1 == 0) & (mask_2 == 0) & land_mask & (vol_1 > 0) & (vol_2 > 0) & \
                             (~np.isnan(vol_1)) & (~np.isnan(vol_2)) & \
                             (~np.isnan(z_1)) & (~np.isnan(z_2))
                
                intersect_1_z = z_1[joint_mask]
                intersect_2_z = z_2[joint_mask]
            
            if len(intersect_1_z) > 10:
                z_1_global.append(intersect_1_z)
                z_2_global.append(intersect_2_z)
                accepted_pairs += 1
                
        self.z_1_global = np.concatenate(z_1_global) if z_1_global else np.array([])
        self.z_2_global = np.concatenate(z_2_global) if z_2_global else np.array([])
        
        self.h5.close()
        print(f"[{self.pair_name}] min_commonality={self.min_commonality}, min_swath={self.min_swath}: {accepted_pairs} pairs accepted, {len(self.z_1_global)} pixels")
        return len(self.z_1_global) > 0

def create_figure(filepath, comparisons, output_filename, eps_idx):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), layout='constrained')
    hb_list = []
    
    first_clim = None
    
    for ax, comp in zip(axes, comparisons):
        extractor = HLST_Statistical_Extractor(
            filepath, 
            comp["name"], 
            comp["s1"], 
            comp["s2"], 
            match_tolerance_days=comp.get("tolerance", 5.0),
            min_commonality=comp.get("min_commonality", None),
            min_swath=comp.get("min_swath", None)
        )
        if extractor.extract_matched_distributions():
            data_1 = extractor.z_1_global
            data_2 = extractor.z_2_global
            
            # Robust linear fit via RANSACRegressor to reject clustered mineral/rooftop leverage points
            if len(data_1) > 100000:
                sample_idx = np.random.RandomState(42).choice(len(data_1), 100000, replace=False)
                X_fit = data_1[sample_idx].reshape(-1, 1)
                y_fit = data_2[sample_idx]
            else:
                X_fit = data_1.reshape(-1, 1)
                y_fit = data_2

            ransac = RANSACRegressor(random_state=42)
            ransac.fit(X_fit, y_fit)
            
            slope = float(ransac.estimator_.coef_[0])
            intercept = float(ransac.estimator_.intercept_)
            
            # Evaluate consensus inliers across full dataset
            residuals = np.abs(data_2 - (slope * data_1 + intercept))
            thresh = (ransac.residual_threshold_ 
                      if hasattr(ransac, 'residual_threshold_') and ransac.residual_threshold_ is not None 
                      else (np.median(residuals) * 2.5))
            inlier_mask = residuals <= thresh
            inlier_pct = float(np.mean(inlier_mask) * 100.0)
            
            if np.sum(inlier_mask) > 10:
                r_val, _ = stats.pearsonr(data_1[inlier_mask], data_2[inlier_mask])
            else:
                r_val, _ = stats.pearsonr(data_1, data_2)

            if first_clim is None:
                hb = ax.hexbin(data_1, data_2, gridsize=50, cmap='cividis', bins='log', mincnt=1)
                first_clim = hb.get_clim()
            else:
                hb = ax.hexbin(data_1, data_2, gridsize=50, cmap='cividis', bins='log', mincnt=1, vmin=first_clim[0], vmax=first_clim[1])
            hb_list.append(hb)
            
            x_sp = np.array([data_1.min(), data_1.max()])
            y_sp = intercept + slope * x_sp
            sign = '+' if intercept >= 0 else '-'
            
            n_pixels = len(data_1)
            eq_text = (f"Pixels: {n_pixels:.1e}\n"
                       f"Inliers: {inlier_pct:.1f}%\n"
                       f"$r$: {r_val:.3f}\n"
                       f"$y = {slope:.2f}x {sign} {abs(intercept):.2f}$")
            
            ax.plot(x_sp, y_sp, 'cyan', linestyle='--', lw=1.5, alpha=0.8, label='RANSAC Trend')
            
            ax.set_title(comp["name"], fontsize=9, fontweight='bold')
            ax.set_xlabel(comp["s1_label"], fontsize=8)
            ax.set_ylabel(comp["s2_label"], fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', pad=2), 
                    color='black', fontsize=7, va='top')
            
    # Add a single global colorbar for the figure if anything was plotted
    if hb_list:
        cb = fig.colorbar(hb_list[0], ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cb.set_label('Pixel Pair Count', size=7)
        cb.ax.tick_params(labelsize=6)
            
    os.makedirs(MEDIA_DIR, exist_ok=True)
    plt.savefig(os.path.join(MEDIA_DIR, output_filename), dpi=600, pad_inches=0, bbox_inches='tight')
    plt.savefig(os.path.join(MEDIA_DIR, f'fig{eps_idx}.pdf'), dpi=600, pad_inches=0, bbox_inches='tight')
    plt.close()

def plot_cross_sensor_correlations():
    
    
    cross_comparisons = [
        {
            "name": "Landsat vs Sentinel-2", 
            "s1_label": "Landsat Spectral Complexity  Z-Score", "s2_label": "Sentinel-2 Spectral Complexity  Z-Score",
            "s1": ["LANDSAT_8", "LANDSAT-8", "LANDSAT_9", "LANDSAT-9"], 
            "s2": ["Sentinel-2A", "SENTINEL-2A", "Sentinel-2B", "SENTINEL-2B"],
            "tolerance": MATCH_TOLERANCE_DAYS,
            "min_commonality": MINIMUM_COMMONALITY
        },
        {
            "name": "Landsat vs Tanager", 
            "s1_label": "Landsat Spectral Complexity  Z-Score", "s2_label": "Tanager Spectral Complexity  Z-Score",
            "s1": ["LANDSAT_8", "LANDSAT-8", "LANDSAT_9", "LANDSAT-9"], 
            "s2": ["Tanager-1", "TANAGER-1"],
            "tolerance": MATCH_TOLERANCE_DAYS,
            "min_swath": MINIMUM_COMMONALITY
        },
        {
            "name": "Sentinel-2 vs Tanager", 
            "s1_label": "Sentinel-2 Spectral Complexity  Z-Score", "s2_label": "Tanager Spectral Complexity  Z-Score",
            "s1": ["Sentinel-2A", "SENTINEL-2A", "Sentinel-2B", "SENTINEL-2B"], 
            "s2": ["Tanager-1", "TANAGER-1"],
            "tolerance": MATCH_TOLERANCE_DAYS,
            "min_swath": MINIMUM_COMMONALITY
        }
    ]
    
    self_comparisons = [
        {
            "name": "Landsat vs Landsat", 
            "s1_label": "Landsat Spectral Complexity  Z-Score (t1)", "s2_label": "Landsat Spectral Complexity  Z-Score (t2)",
            "s1": ["LANDSAT_8", "LANDSAT-8", "LANDSAT_9", "LANDSAT-9"], 
            "s2": ["LANDSAT_8", "LANDSAT-8", "LANDSAT_9", "LANDSAT-9"],
            "tolerance": MATCH_TOLERANCE_DAYS,
            "min_commonality": MINIMUM_COMMONALITY
        },
        {
            "name": "Sentinel-2 vs Sentinel-2", 
            "s1_label": "Sentinel-2 Spectral Complexity  Z-Score (t1)", "s2_label": "Sentinel-2 Spectral Complexity Z-Score (t2)",
            "s1": ["Sentinel-2A", "SENTINEL-2A", "Sentinel-2B", "SENTINEL-2B"], 
            "s2": ["Sentinel-2A", "SENTINEL-2A", "Sentinel-2B", "SENTINEL-2B"],
            "tolerance": MATCH_TOLERANCE_DAYS,
            "min_commonality": MINIMUM_COMMONALITY
        },
        {
            "name": "Tanager vs Tanager", 
            "s1_label": "Tanager Spectral Complexity  Z-Score (t1)", "s2_label": "Tanager Spectral Complexity  Z-Score (t2)",
            "s1": ["Tanager-1", "TANAGER-1"], 
            "s2": ["Tanager-1", "TANAGER-1"],
            "tolerance": 30.0
        }
    ]

    
    create_figure(FILEPATH, self_comparisons, 'self_sensor_correlation_global.png', '10')
    create_figure(FILEPATH, cross_comparisons, 'cross_sensor_correlation_global.png', '11')

if __name__ == '__main__':
    plot_cross_sensor_correlations()
