import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button
import rasterio.transform
from pyproj import Transformer, CRS
Location = "Rochesterv2"
DATA_DIR = f"C:/satelliteImagery/Tanager/{Location}_SourceData"
STACKED_H5 = os.path.join(DATA_DIR, f"Tanager_Native_Stack_{Location}.h5")

TS_LOCATIONS_MAP = {
    "Rochesterv2": [
        {'latlon': (43.13927, -77.50340), 'label': "ROCX NITE Tarp",                  'color': 'tab:purple'},
        {'latlon': (43.142856, -77.508451), 'label': "West Tait Forest",                'color': 'tab:green'},
        {'latlon': (43.144861, -77.501176), 'label': "East Tait Forest",                'color': 'tab:olive'},
        {'latlon': (43.151502, -77.485518), 'label': "Shadow Pines Grass Field",         'color': 'tab:red'},
        {'latlon': (43.151219, -77.486637), 'label': "Shadow Pines Pickleball Court",    'color': 'tab:blue'},
        {'latlon': (43.151877, -77.487111), 'label': "Shadow Pines Playground",          'color': 'tab:cyan'},
    ]
}

class PassViewer:
    def __init__(self, h5_path):
        self.h5f = h5py.File(h5_path, 'r')
        self.grp = self.h5f['HDFEOS/GRIDS/TANAGER/Data Fields']
        
        self.sr_z = self.grp['sr_zscore']
        self.toa_z = self.grp['toa_zscore']
        self.ortho = self.grp['ortho_visual']
        self.sr = self.grp['surface_reflectance']
        
        geo_transform = self.sr.attrs['GeoTransform']
        spatial_ref = self.sr.attrs['spatial_ref']
        if isinstance(spatial_ref, bytes):
            spatial_ref = spatial_ref.decode('utf-8')
            
        crs = CRS(spatial_ref)
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        affine = rasterio.transform.Affine.from_gdal(*geo_transform)
        inv_affine = ~affine
        
        self.locations = TS_LOCATIONS_MAP[Location]
        for loc in self.locations:
            lat, lon = loc['latlon']
            proj_x, proj_y = transformer.transform(lon, lat)
            px, py = inv_affine * (proj_x, proj_y)
            loc['yx'] = (int(round(py)), int(round(px)))
        
        self.n_passes = self.sr_z.shape[0]
        self.current_pass = 0
        
        self.fig, self.axs = plt.subplots(1, 4, figsize=(24, 6))
        for ax in self.axs.flat:
            ax.set_facecolor('dimgrey')
        self.fig.tight_layout(rect=[0, 0.15, 1, 1])
        
        self.im_sr = None
        self.im_toa = None
        self.im_rgb = None
        self.im_diff = None
        
        self.setup_ui()
        self.update_plot()
        
    def setup_ui(self):
        ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.prev_pass)
        
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_pass)
        
    def prev_pass(self, event):
        if self.current_pass > 0:
            self.current_pass -= 1
            self.update_plot()
            
    def next_pass(self, event):
        if self.current_pass < self.n_passes - 1:
            self.current_pass += 1
            self.update_plot()
            
    def update_plot(self):
        sr = self.sr_z[self.current_pass, ...]
        toa = self.toa_z[self.current_pass, ...]
        rgb = self.ortho[self.current_pass, ...]  # Shape (3, Y, X)
        rgb = np.transpose(rgb, (1, 2, 0)) # Convert to (Y, X, 3) for imshow
        
        diff = sr - toa
        
        diff_min, diff_max = np.nanmin(diff), np.nanmax(diff)
        if diff_min >= 0: diff_min = -1e-5
        if diff_max <= 0: diff_max = 1e-5
        diff_norm = mcolors.TwoSlopeNorm(vcenter=0, vmin=diff_min, vmax=diff_max)
        
        if self.im_sr is None:
            self.im_rgb = self.axs[0].imshow(rgb)
            self.axs[0].set_title("Ortho Visual (RGB)")
            
            self.im_sr = self.axs[1].imshow(sr, cmap='viridis')
            self.axs[1].set_title(f"Pass {self.current_pass + 1}/{self.n_passes} - SR Z-Score")
            self.fig.colorbar(self.im_sr, ax=self.axs[1])
            
            self.im_diff = self.axs[2].imshow(diff, cmap='bwr', norm=diff_norm)
            self.axs[2].set_title("Difference (SR - TOA)")
            self.cb_diff = self.fig.colorbar(self.im_diff, ax=self.axs[2])
            
            self.im_toa = self.axs[3].imshow(toa, cmap='viridis')
            self.axs[3].set_title(f"Pass {self.current_pass + 1}/{self.n_passes} - TOA Z-Score")
            self.fig.colorbar(self.im_toa, ax=self.axs[3])
            
            for ax in self.axs.flat:
                for loc in self.locations:
                    y, x = loc['yx']
                    ax.plot(x + 0.5, y + 0.5, marker='s', markersize=10, markeredgecolor=loc['color'], 
                            markerfacecolor='none', markeredgewidth=1.5, linestyle='None')
        else:
            self.im_sr.set_data(sr)
            self.im_sr.set_clim(vmin=np.nanmin(sr), vmax=np.nanmax(sr))
            self.axs[1].set_title(f"Pass {self.current_pass + 1}/{self.n_passes} - SR Z-Score")
            
            self.im_toa.set_data(toa)
            self.im_toa.set_clim(vmin=np.nanmin(toa), vmax=np.nanmax(toa))
            self.axs[3].set_title(f"Pass {self.current_pass + 1}/{self.n_passes} - TOA Z-Score")
            
            self.im_rgb.set_data(rgb)
            
            self.im_diff.set_data(diff)
            self.im_diff.set_norm(diff_norm)
            if hasattr(self, 'cb_diff'):
                self.cb_diff.update_normal(self.im_diff)
            
        self.fig.canvas.draw_idle()

def main():
    if not os.path.exists(STACKED_H5):
        print(f"File not found: {STACKED_H5}")
        print("Please run evaluate_sr_vs_toa_complexity.py first.")
        return
        
    viewer = PassViewer(STACKED_H5)
    plt.show()

if __name__ == "__main__":
    main()
