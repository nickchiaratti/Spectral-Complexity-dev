with open(r'F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\change_detection\harmonized-CCD\generate_presentation_plots.py', 'r') as f:
    text = f.read()

new_func = '''
def save_all_seasons_separability_plot(pixel_y, pixel_x, source_h5_path, inference_results_h5, out_path):
    print("Generating All-Seasons Separability Plot...")
    with h5py.File(inference_results_h5, 'r') as f_inf:
        rmse = f_inf['rmse_series'][:, pixel_y, pixel_x]
        
    with h5py.File(source_h5_path, 'r') as f_sc:
        harm_grp = f_sc['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        target_metric = f_sc.attrs.get('TARGET_METRIC', 'sliding_volume_z_score')
        if target_metric not in harm_grp:
            target_metric = list(harm_grp.keys())[0]
            
        attrs = harm_grp[target_metric].attrs
        acq_time = attrs['acquisition_time'][:]
        source_grids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attrs['source_grid'][:]]
        source_frames = attrs['source_frame_index'][:]
        spacecrafts = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in attrs['source_spacecraft'][:]]
        unified_masks = harm_grp['common_mask'][:, pixel_y, pixel_x]
        
    raw_h5_path = re.sub(r'_SC_.*?\.h5$', '.h5', source_h5_path)
    
    segments = []
    current_seg = []
    current_rmse = None
    for i in range(len(acq_time)):
        if unified_masks[i]: continue
        r = rmse[i]
        if np.isnan(r): continue
        if current_rmse is None:
            current_rmse = r
            current_seg.append(i)
        elif abs(r - current_rmse) < 1e-6:
            current_seg.append(i)
        else:
            segments.append(current_seg)
            current_seg = [i]
            current_rmse = r
    if current_seg:
        segments.append(current_seg)
        
    segment_colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Aggregated All-Seasons Separability Across Segments\\nPixel: x={pixel_x}, y={pixel_y}", fontsize=16)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Surface Reflectance")
    
    global_y_max = 0
    
    with h5py.File(raw_h5_path, 'r') as f_raw:
        for seg_idx, seg in enumerate(segments):
            seg_color = segment_colors[seg_idx % len(segment_colors)]
            
            from collections import defaultdict
            sensor_spectra = defaultdict(list)
            
            for i in seg:
                grid = source_grids[i]
                frame_idx = source_frames[i]
                sensor = str(spacecrafts[i]).upper()
                if 'LANDSAT' in sensor: sensor = 'Landsat'
                elif 'SENTINEL' in sensor: sensor = 'Sentinel'
                elif 'TANAGER' in sensor: sensor = 'Tanager'
                
                sr_ds = f_raw[f"/HDFEOS/GRIDS/{grid}/Data Fields/surface_reflectance"]
                patch = sr_ds[frame_idx, :, max(0, pixel_y-1):pixel_y+2, max(0, pixel_x-1):pixel_x+2]
                
                w_attr = sr_ds.attrs.get('wavelengths', sr_ds.attrs.get('wavelength'))
                wavelengths = w_attr[:] if w_attr is not None else np.arange(1, patch.shape[0]+1, dtype=float)
                
                if grid == "TANAGER":
                    gw_mask = sr_ds.attrs.get("all_good_wavelengths")[frame_idx].astype(bool)
                    patch = patch[gw_mask, :, :]
                    wavelengths = wavelengths[gw_mask]
                    
                if np.max(wavelengths) < 10: wavelengths *= 1000
                
                patch = np.transpose(patch, (1, 2, 0))
                em, _ = sc.maximumDistance(patch, N_ENDMEMBERS, strict_nan=False)
                if not np.isnan(em).all():
                    sensor_spectra[sensor].append((em, wavelengths))
                    
            if not sensor_spectra: continue
            
            for sensor, data in sensor_spectra.items():
                all_lines = []
                wl = data[0][1]
                for em, _ in data:
                    for j in range(em.shape[1]):
                        if not np.isnan(em[:, j]).all():
                            ax.plot(wl, em[:, j], color=seg_color, alpha=0.15)
                            all_lines.append(em[:, j])
                            
                if all_lines:
                    centroid = np.nanmean(np.array(all_lines), axis=0)
                    label = f"Seg {seg_idx+1} Centroid" if sensor == list(sensor_spectra.keys())[0] else None
                    ax.plot(wl, centroid, color=seg_color, linewidth=3.0, label=label)
                    max_val = np.nanmax(np.array(all_lines))
                    if not np.isnan(max_val):
                        global_y_max = max(global_y_max, max_val * 1.1)
                        
        ax.legend(loc='upper left')
        ax.set_ylim(0, global_y_max if global_y_max > 0 else 1.0)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
'''

text = text.replace('if __name__ ==', new_func + '\nif __name__ ==')

main_target = '''    p3 = os.path.join(out_dir, f'separability_y{px_y}_x{px_x}.png')
    
    save_ortho_visual(px_y, px_x, source_h5, inf_h5, p1)
    save_time_series(px_y, px_x, source_h5, inf_h5, p2)
    save_separability_plot(px_y, px_x, source_h5, inf_h5, p3)'''

main_replacement = '''    p3 = os.path.join(out_dir, f'separability_y{px_y}_x{px_x}.png')
    p4 = os.path.join(out_dir, f'all_seasons_separability_y{px_y}_x{px_x}.png')
    
    save_ortho_visual(px_y, px_x, source_h5, inf_h5, p1)
    save_time_series(px_y, px_x, source_h5, inf_h5, p2)
    save_separability_plot(px_y, px_x, source_h5, inf_h5, p3)
    save_all_seasons_separability_plot(px_y, px_x, source_h5, inf_h5, p4)'''

text = text.replace(main_target, main_replacement)

with open(r'F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\change_detection\harmonized-CCD\generate_presentation_plots.py', 'w') as f:
    f.write(text)
print("Patched all seasons")
