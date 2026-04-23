import os
import h5py
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, TextBox
from datetime import datetime, timezone
from skimage import exposure

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
# Point to the EVALUATED HDF5 file containing the new prediction datasets
H5_PATH = "C:/satelliteImagery/LANDSAT/Rochester/LANDSAT_Stack_Rochester_EVALUATED.h5"

# Multi-Year Ground Truth paths
CDL_PATHS = {
    2023: "C:/satelliteImagery/LANDSAT/Rochester/CDL2023_Aligned_Rochester.tif",
    2024: "C:/satelliteImagery/LANDSAT/Rochester/CDL2024_Aligned_Rochester.tif",
    2025: "C:/satelliteImagery/LANDSAT/Rochester/CDL2025_Aligned_Rochester.tif",
}

# --- DYNAMIC CLASS MAPPING & CUSTOM COLORS ---
# Update these names to perfectly match your CDL_Global_Mapping_Report.txt!
# RGB values ensure the maps render with logical, natural colors.
CLASS_CONFIG = {
    1: {"name": "Developed",         "rgb": [155, 165, 165]}, # Gray
    2: {"name": "Deciduous Forest",  "rgb": [34, 139, 34]},   # Forest Green
    3: {"name": "Open Water",        "rgb": [0, 119, 190]},   # Deep Blue
    4: {"name": "Grassland/Pasture", "rgb": [154, 205, 50]},  # Yellow-Green
    5: {"name": "Woody Wetlands",    "rgb": [0, 139, 139]},   # Teal
    6: {"name": "Corn",              "rgb": [255, 215, 0]},   # Gold
    7: {"name": "Soybeans",          "rgb": [38, 112, 0]},    # Dark Green
    8: {"name": "Winter Wheat",      "rgb": [210, 180, 140]}, # Tan/Brown
    9: {"name": "Alfalfa",           "rgb": [255, 182, 193]}, # Light Pink
    10: {"name": "Fallow/Idle",      "rgb": [128, 0, 128]},   # Purple
}

# Standard Landsat 8/9 True Color Indices: [R(3), G(2), B(1)]
LANDSAT_RGB_BANDS = (3, 2, 1)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def percentile_normalize_array(arr, low=2, high=98):
    """Normalizes arrays for true-color RGB visualization."""
    if np.all(np.isnan(arr)): return np.zeros_like(arr)
    p_low, p_high = np.nanpercentile(arr, (low, high))
    if p_low == p_high: return np.zeros_like(arr)
    return exposure.rescale_intensity(arr, in_range=(p_low, p_high), out_range=(0, 1)).clip(0, 1)

def apply_color_palette(predicted_map, num_classes):
    """Converts a 2D class array into a 3D RGB image using our defined colors."""
    height, width = predicted_map.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for cls_idx in range(num_classes):
        mask = (predicted_map == cls_idx)
        if cls_idx == 0:
            rgb_image[mask] = [0, 0, 0] # Background is strictly black
        elif cls_idx in CLASS_CONFIG:
            rgb_image[mask] = CLASS_CONFIG[cls_idx]["rgb"]
        else:
            # Fallback color for unmapped classes (Bright Magenta)
            rgb_image[mask] = [255, 0, 255]
            
    return rgb_image

def calculate_frame_metrics(pred_mask, true_mask, num_classes):
    """Calculates Mean IoU, Overall Accuracy, and Per-Class Accuracies."""
    valid_mask = (true_mask != 0)
    if not np.any(valid_mask):
        return 0.0, 0.0, {}
        
    correct_pixels = (pred_mask[valid_mask] == true_mask[valid_mask]).sum()
    overall_accuracy = correct_pixels / valid_mask.sum()
    
    ious = []
    per_class = {}
    
    for cls in range(1, num_classes):
        p_c = (pred_mask == cls) & valid_mask
        t_c = (true_mask == cls) & valid_mask
        
        intersection = (p_c & t_c).sum()
        union = (p_c | t_c).sum()
        true_total = t_c.sum()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou)
            # Accuracy (Recall) for this specific class
            acc = intersection / true_total if true_total > 0 else 0.0
            per_class[cls] = {'iou': iou, 'acc': acc, 'present': True}
        else:
            per_class[cls] = {'iou': 0.0, 'acc': 0.0, 'present': False}
            
    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
    return overall_accuracy, mean_iou, per_class

# ==========================================
# 3. VIEWER APPLICATION
# ==========================================
class PredictionViewer:
    def __init__(self, h5_path, cdl_paths):
        self.h5_path = h5_path
        self.cdl_paths = cdl_paths
        
        self.h5 = h5py.File(self.h5_path, 'r')
        self.data_grp = self.h5['/HDFEOS/GRIDS/LANDSAT/Data Fields']
        
        self.sr_ds = self.data_grp['surface_reflectance']
        self.num_frames = self.sr_ds.shape[0]
        self.current_idx = 0
        
        self.sc_ds = self.data_grp.get('sliding_volume_map')
        self.pred_7_ds = self.data_grp.get('predicted_cdl_7band')
        self.pred_8_ds = self.data_grp.get('predicted_cdl_8band')
        
        acq_times = self.sr_ds.attrs.get('acquisition_time')
        self.frame_years = []
        self.timestamps = []
        
        for dt in acq_times:
            try:
                dt_obj = datetime.fromtimestamp(float(dt), tz=timezone.utc)
            except ValueError:
                dt_str = dt.decode('utf-8') if isinstance(dt, bytes) else str(dt)
                dt_obj = datetime.strptime(dt_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                
            self.timestamps.append(dt_obj)
            self.frame_years.append(dt_obj.year)
            
        print("Loading Aligned CDL Masks...")
        self.gt_masks = {}
        max_class = 0
        required_years = set(self.frame_years)
        
        for year in required_years:
            if year in self.cdl_paths and os.path.exists(self.cdl_paths[year]):
                with rasterio.open(self.cdl_paths[year]) as src:
                    mask = src.read(1).astype(np.int64)
                    self.gt_masks[year] = mask
                    max_class = max(max_class, int(np.max(mask)))
            else:
                print(f"WARNING: Ground truth for {year} not found.")
                self.gt_masks[year] = None
                
        self.num_classes = max_class + 1
        print(f"Dynamically detected {self.num_classes} total classes.")

        self._init_control_ui()
        self._init_visualization_ui()
        self.update_display()

    def _init_control_ui(self):
        self.fig_controls = plt.figure(figsize=(4, 3))
        self.fig_controls.canvas.manager.set_window_title("Navigation Controls")
        
        self.ax_meta = self.fig_controls.add_axes([0, 0, 1, 1])
        self.ax_meta.axis('off')
        
        self.ctrl_text = self.ax_meta.text(0.5, 0.75, "", ha='center', va='center', 
                                         fontsize=10, family='monospace',
                                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax_prev = self.fig_controls.add_axes([0.1, 0.2, 0.25, 0.2])
        ax_input = self.fig_controls.add_axes([0.45, 0.2, 0.15, 0.2])
        ax_next = self.fig_controls.add_axes([0.65, 0.2, 0.25, 0.2])
        
        self.btn_prev = Button(ax_prev, '<< Prev')
        self.btn_next = Button(ax_next, 'Next >>')
        self.txt_input = TextBox(ax_input, 'Go: ', initial='0')
        
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.txt_input.on_submit(self._on_submit)

    def _init_visualization_ui(self):
        self.fig_vis, self.axs = plt.subplots(2, 3, figsize=(18, 10))
        self.fig_vis.canvas.manager.set_window_title("Model Prediction Viewer")
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.05, right=0.95, hspace=0.25, wspace=0.2)
        
        self.ax_rgb = self.axs[0, 0]
        self.ax_sc = self.axs[0, 1]
        self.ax_gt = self.axs[0, 2]
        self.ax_pred7 = self.axs[1, 0]
        self.ax_pred8 = self.axs[1, 1]
        self.ax_stats = self.axs[1, 2]
        self.ax_stats.axis('off')

    def update_display(self):
        idx = self.current_idx
        year = self.frame_years[idx]
        dt_str = self.timestamps[idx].strftime('%Y-%m-%d %H:%M:%S UTC')
        
        meta_str = f"FRAME: {idx + 1} / {self.num_frames}\nDATE: {dt_str}\nYEAR MATCH: {year}"
        self.ctrl_text.set_text(meta_str)
        self.fig_vis.suptitle(f"Landsat Frame {idx} | Acquired: {dt_str}", fontsize=16)

        sr_data = self.sr_ds[idx, ...]
        r = percentile_normalize_array(sr_data[LANDSAT_RGB_BANDS[0]])
        g = percentile_normalize_array(sr_data[LANDSAT_RGB_BANDS[1]])
        b = percentile_normalize_array(sr_data[LANDSAT_RGB_BANDS[2]])
        rgb = np.nan_to_num(np.stack([r, g, b], axis=-1), nan=0.0)
        
        self.ax_rgb.clear()
        self.ax_rgb.imshow(rgb)
        self.ax_rgb.set_title("Landsat True Color")
        self.ax_rgb.axis('off')

        self.ax_sc.clear()
        if self.sc_ds is not None:
            sc_data = self.sc_ds[idx, ...]
            sc_norm = percentile_normalize_array(sc_data)
            self.ax_sc.imshow(sc_norm, cmap='viridis')
            self.ax_sc.set_title("Sliding Spectral Complexity")
        else:
            self.ax_sc.text(0.5, 0.5, "Complexity Data\nNot Found", ha='center', va='center')
        self.ax_sc.axis('off')

        self.ax_gt.clear()
        gt_mask = self.gt_masks.get(year)
        if gt_mask is not None:
            gt_rgb = apply_color_palette(gt_mask, self.num_classes)
            self.ax_gt.imshow(gt_rgb)
            self.ax_gt.set_title(f"Aligned CDL Ground Truth ({year})")
        else:
            self.ax_gt.text(0.5, 0.5, f"No CDL Found\nfor {year}", ha='center', va='center')
        self.ax_gt.axis('off')

        self.ax_pred7.clear()
        metrics_7 = (0.0, 0.0, {})
        if self.pred_7_ds is not None:
            pred7 = self.pred_7_ds[idx, ...]
            self.ax_pred7.imshow(apply_color_palette(pred7, self.num_classes))
            self.ax_pred7.set_title("7-Band Baseline Prediction")
            if gt_mask is not None:
                metrics_7 = calculate_frame_metrics(pred7, gt_mask, self.num_classes)
        else:
            self.ax_pred7.text(0.5, 0.5, "7-Band Output\nNot Found", ha='center', va='center')
        self.ax_pred7.axis('off')

        self.ax_pred8.clear()
        metrics_8 = (0.0, 0.0, {})
        if self.pred_8_ds is not None:
            pred8 = self.pred_8_ds[idx, ...]
            self.ax_pred8.imshow(apply_color_palette(pred8, self.num_classes))
            self.ax_pred8.set_title("8-Band Surgery Prediction")
            if gt_mask is not None:
                metrics_8 = calculate_frame_metrics(pred8, gt_mask, self.num_classes)
        else:
            self.ax_pred8.text(0.5, 0.5, "8-Band Output\nNot Found", ha='center', va='center')
        self.ax_pred8.axis('off')

        self.ax_stats.clear()
        self.ax_stats.axis('off')
        
        if gt_mask is not None:
            stats_text = (
                f"--- Overall Single Frame Metrics ---\n\n"
                f"7-Band Baseline:\n"
                f"  Mean IoU: {metrics_7[1]:.4f} | OA: {metrics_7[0]:.4f}\n"
                f"8-Band Complexity Model:\n"
                f"  Mean IoU: {metrics_8[1]:.4f} | OA: {metrics_8[0]:.4f}\n"
                f"Absolute Improvement:\n"
                f"  Δ mIoU:   {metrics_8[1] - metrics_7[1]:+.4f}\n"
                f"  Δ OA:     {metrics_8[0] - metrics_7[0]:+.4f}"
            )
            self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes, 
                               fontsize=12, family='monospace', va='top',
                               bbox=dict(facecolor='#f4f4f4', edgecolor='gray', boxstyle='round,pad=0.5'))

            legend_handles = []
            for cls in range(1, self.num_classes):
                cfg = CLASS_CONFIG.get(cls, {"name": f"Class {cls}", "rgb": [255, 0, 255]})
                
                # Convert the custom RGB [0, 255] to matplotlib's internal format [0.0, 1.0]
                normalized_rgba = [c / 255.0 for c in cfg["rgb"]] + [1.0] 
                
                m7_acc = metrics_7[2].get(cls, {}).get('acc', 0.0) * 100
                m8_acc = metrics_8[2].get(cls, {}).get('acc', 0.0) * 100
                present = metrics_8[2].get(cls, {}).get('present', False)
                
                class_name = cfg["name"][:18]
                
                if present:
                    label = f"{class_name:<18} | 7B: {m7_acc:>5.1f}% | 8B: {m8_acc:>5.1f}%"
                else:
                    label = f"{class_name:<18} | --- Not Present in GT ---"
                    
                patch = mpatches.Patch(color=normalized_rgba, label=label)
                legend_handles.append(patch)

            self.ax_stats.legend(handles=legend_handles, loc='lower left', 
                                 bbox_to_anchor=(0.05, 0.0), 
                                 prop={'family': 'monospace', 'size': 10},
                                 title="Map Legend & Per-Class Pixel Accuracy",
                                 title_fontsize=11,
                                 frameon=True, facecolor='white', edgecolor='lightgray')

        else:
            self.ax_stats.text(0.1, 0.5, "Cannot calculate accuracy:\nGround Truth missing for this year.", 
                               transform=self.ax_stats.transAxes, fontsize=12, va='center')

        self.fig_controls.canvas.draw_idle()
        self.fig_vis.canvas.draw_idle()

    def _on_prev(self, event):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.txt_input.set_val(str(self.current_idx))
            self.update_display()

    def _on_next(self, event):
        if self.current_idx < self.num_frames - 1:
            self.current_idx += 1
            self.txt_input.set_val(str(self.current_idx))
            self.update_display()

    def _on_submit(self, text):
        try:
            val = int(text)
            if 0 <= val < self.num_frames:
                self.current_idx = val
                self.update_display()
            else:
                self.txt_input.set_val(str(self.current_idx))
        except ValueError:
            self.txt_input.set_val(str(self.current_idx))

    def run(self):
        plt.show()

if __name__ == "__main__":
    print("Initializing Multi-Model Viewer...")
    viewer = PredictionViewer(H5_PATH, CDL_PATHS)
    viewer.run()