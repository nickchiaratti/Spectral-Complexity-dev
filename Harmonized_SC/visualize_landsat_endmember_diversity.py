import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import time
import numpy as np
import h5py
import torch
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import distance_transform_edt
import pyproj
from datetime import datetime, timezone

# Ensure parent directory is accessible for SpecComplexTorch import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import scienceplots
plt.style.use(['science','ieee'])
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from SpecComplexTorch import maximumDistance_torch
except ImportError as e:
    raise RuntimeError(f"CRITICAL ERROR: Failed to import maximumDistance_torch from SpecComplexTorch: {e}")


def decode_str(val):
    """Safely decode HDF5 string or bytes attributes."""
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return str(val)


def get_lat_lon(h5_ds, base_ds, pixel_y, pixel_x):
    """Converts pixel coordinates (row, col) to geographic (latitude, longitude)."""
    gt = h5_ds.attrs.get("GeoTransform")
    spatial_ref = h5_ds.attrs.get("spatial_ref")
    if gt is None or spatial_ref is None:
        gt = base_ds.attrs.get("GeoTransform")
        spatial_ref = base_ds.attrs.get("spatial_ref")
    if gt is None or spatial_ref is None:
        return 0.0, 0.0
    try:
        x_geo = gt[0] + (pixel_x + 0.5) * gt[1] + (pixel_y + 0.5) * gt[2]
        y_geo = gt[3] + (pixel_x + 0.5) * gt[4] + (pixel_y + 0.5) * gt[5]
        spatial_ref_str = decode_str(spatial_ref)
        crs = pyproj.CRS.from_wkt(spatial_ref_str)
        transformer = pyproj.Transformer.from_crs(crs, "epsg:4326", always_xy=True)
        lon, lat = transformer.transform(x_geo, y_geo)
        return float(lat), float(lon)
    except Exception as e:
        print(f"Warning: Could not convert pixel to lat/lon: {e}")
        return 0.0, 0.0


def get_collection_date(sc_ds, base_ds, global_idx, local_idx):
    """Retrieves collection date string (YYYY-MM-DD) from acquisition timestamp."""
    acq_time = sc_ds.attrs.get("acquisition_time")
    ts = None
    if acq_time is not None and len(acq_time) > global_idx:
        ts = acq_time[global_idx]
    if ts is None or ts == 0:
        base_acq = base_ds.attrs.get("acquisition_time")
        if base_acq is not None and len(base_acq) > local_idx:
            ts = base_acq[local_idx]
    if ts is not None and ts > 0:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
    return f"Date Unknown (#{global_idx})"


def find_landsat_candidates(h5_file, base_file, separation_pixels=10, chip_size=10, top_n=10):
    """
    Scans all frames in the harmonized HDF5 file for LANDSAT (HLSL30) imagery.
    Identifies valid pixels in sliding_volume_map that maintain at least
    `separation_pixels` distance from any masked pixel in common_mask.
    
    Returns:
        top_high_candidates, top_low_candidates (lists of tuples)
    """
    print("=" * 70)
    print("STEP 1: SCANNING LANDSAT FRAMES FOR SLIDING VOLUME EXTREMES")
    print("=" * 70)
    
    harm_grp = h5_file["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
    slide_ds = harm_grp["sliding_volume_map"]
    mask_ds = harm_grp["common_mask"]
    
    source_grids = slide_ds.attrs.get("source_grid")
    source_indices = slide_ds.attrs.get("source_frame_index")
    
    if source_grids is None or source_indices is None:
        raise KeyError("CRITICAL ERROR: 'source_grid' or 'source_frame_index' provenance attributes missing on sliding_volume_map.")
    
    total_frames, height, width = slide_ds.shape
    print(f"Total harmonized frames: {total_frames} | Spatial dimensions: {height}x{width} | Configured Chip Size: {chip_size}x{chip_size}")
    
    all_candidates = []
    
    landsat_frames_processed = 0
    total_candidates_found = 0
    
    # Enforce separation from masked pixels AND image edges
    effective_sep = max(float(separation_pixels), float(chip_size // 2))
    edge_margin = max(int(separation_pixels), chip_size // 2)
    
    for idx in range(total_frames):
        grid_str = decode_str(source_grids[idx])
        
        # Filter exclusively for LANDSAT collections (HLSL30)
        if "HLSL30" not in grid_str.upper() and "LANDSAT" not in grid_str.upper():
            continue
            
        local_idx = int(source_indices[idx])
        landsat_frames_processed += 1
        
        base_sr_ds = base_file[f"/HDFEOS/GRIDS/{grid_str}/Data Fields/surface_reflectance"]
        date_str = get_collection_date(slide_ds, base_sr_ds, idx, local_idx)
        
        slide_frame = slide_ds[idx, :, :]
        mask_frame = mask_ds[idx, :, :]
        
        # A pixel is valid if common_mask == 0 and sliding_volume_map is finite
        valid_pixels = (mask_frame == 0) & np.isfinite(slide_frame)
        
        # Compute exact Euclidean distance from masked/invalid pixels
        dist_from_masked = distance_transform_edt(valid_pixels)
        
        candidate_mask = (dist_from_masked >= effective_sep)
        candidate_mask[:edge_margin, :] = False
        candidate_mask[-edge_margin:, :] = False
        candidate_mask[:, :edge_margin] = False
        candidate_mask[:, -edge_margin:] = False
        
        num_valid_candidates = np.sum(candidate_mask)
        if num_valid_candidates == 0:
            continue
            
        total_candidates_found += num_valid_candidates
        
        # Extract candidate values and locations
        candidate_ys, candidate_xs = np.where(candidate_mask)
        candidate_vals = slide_frame[candidate_ys, candidate_xs]
        
        # Keep track of local frame max and min to add to our overall lists
        max_idx = np.argmax(candidate_vals)
        local_max_val = candidate_vals[max_idx]
        y_max, x_max = int(candidate_ys[max_idx]), int(candidate_xs[max_idx])
        lat_max, lon_max = get_lat_lon(slide_ds, base_sr_ds, y_max, x_max)
        all_candidates.append(
            (float(local_max_val), idx, y_max, x_max, grid_str, local_idx, lat_max, lon_max, date_str)
        )
            
        min_idx = np.argmin(candidate_vals)
        local_min_val = candidate_vals[min_idx]
        y_min, x_min = int(candidate_ys[min_idx]), int(candidate_xs[min_idx])
        lat_min, lon_min = get_lat_lon(slide_ds, base_sr_ds, y_min, x_min)
        all_candidates.append(
            (float(local_min_val), idx, y_min, x_min, grid_str, local_idx, lat_min, lon_min, date_str)
        )
            
        print(f"  Date [{date_str}] ({grid_str:10s}): {num_valid_candidates:6d} candidates | "
              f"Max: {local_max_val:.6e} | Min: {local_min_val:.6e}")

    print(f"\nProcessed {landsat_frames_processed} Landsat frames. Total valid candidates evaluated: {total_candidates_found}")
    
    if not all_candidates:
        raise RuntimeError(
            f"CRITICAL ERROR: No valid candidates found in Landsat imagery with >= {effective_sep} pixel separation "
            f"from masked pixels and image boundaries. Check masking criteria or reduce separation/chip size."
        )
        
    # Sort by sliding volume value
    all_candidates.sort(key=lambda x: x[0])
    
    unique_candidates = []
    seen = set()
    for c in all_candidates:
        if c not in seen:
            unique_candidates.append(c)
            seen.add(c)
            
    top_low = unique_candidates[:min(top_n, len(unique_candidates))]
    top_high = unique_candidates[-min(top_n, len(unique_candidates)):][::-1] # Reverse so highest is first

    print(f"\n[TOP {len(top_high)} HIGHEST SLIDING VOLUME CANDIDATES]")
    for i, c in enumerate(top_high):
        print(f"  {i+1}. Value: {c[0]:.6e} | Date: {c[8]} | Lat {c[6]:.6f}°, Lon {c[7]:.6f}°")

    print(f"\n[TOP {len(top_low)} LOWEST SLIDING VOLUME CANDIDATES]")
    for i, c in enumerate(top_low):
        print(f"  {i+1}. Value: {c[0]:.6e} | Date: {c[8]} | Lat {c[6]:.6f}°, Lon {c[7]:.6f}°")
        
    return top_high, top_low


def extract_and_compute_endmembers(sc_file, base_file, candidate_info, device, chip_size=15):
    """
    Extracts chip_size x chip_size ortho_visual chip from sc_file and 3x3 surface_reflectance square from base_file.
    Computes 7 endmembers using GPU-accelerated maximumDistance_torch.
    
    Returns:
        dict containing chip_rgb, endmembers_spectra, wavelengths, metadata
    """
    val, global_idx, y, x, grid_str, local_idx, lat, lon, date_str = candidate_info
    
    # Extract datasets
    sr_ds = base_file[f"/HDFEOS/GRIDS/{grid_str}/Data Fields/surface_reflectance"]
    
    half_before = chip_size // 2
    half_after = chip_size - half_before
    
    # 1. Extract chip_size x chip_size true color chip directly from surface_reflectance
    # h5py requires indexing lists to be in increasing order, so we fetch B,G,R (1,2,3) then reverse to R,G,B
    rgb_slice = sr_ds[local_idx, [1, 2, 3], y - half_before : y + half_after, x - half_before : x + half_after] 
    rgb_slice = rgb_slice[::-1, :, :] # reverse from B,G,R to R,G,B
    chip_rgb = np.transpose(rgb_slice, (1, 2, 0)).astype(np.float32)
    
    # Dynamically scale using the maximum value in the local chip to preserve all high-end detail (no flattening)
    chip_max = np.max(chip_rgb)
    if chip_max > 0:
        chip_rgb = np.clip(chip_rgb / chip_max, 0.0, 1.0)
        # Apply a modest gamma correction (0.7) to bring up midtones, since high clouds might stretch the max
        chip_rgb = np.power(chip_rgb, 0.7)
    else:
        chip_rgb = np.zeros_like(chip_rgb)
        
    # 2. Extract 3x3 surface_reflectance square from base_file
    
    if "wavelengths" not in sr_ds.attrs:
        raise KeyError(f"CRITICAL ERROR: 'wavelengths' attribute missing on dataset {sr_ds.name} in base file.")
    wavelengths = sr_ds.attrs["wavelengths"][:]
    
    # 3x3 square corresponding to sliding volume tile at center (y, x)
    sr_3x3 = sr_ds[local_idx, :, y-1:y+2, x-1:x+2] # shape (num_bands, 3, 3)
    
    # Verify no NaN or fill values inside the 3x3 square per restrict-fill-values rule
    if np.any(~np.isfinite(sr_3x3)):
        raise ValueError(f"CRITICAL DATA INTEGRITY ERROR: Non-finite values detected inside 3x3 surface reflectance square at ({y}, {x}).")
        
    num_bands = sr_3x3.shape[0]
    
    # Reshape to (Batch=1, Bands=num_bands, Pixels=9) for maximumDistance_torch
    data_tensor = torch.from_numpy(sr_3x3).reshape(1, num_bands, 9).to(device=device, dtype=torch.float32)
    valid_pixel_mask = torch.ones((1, 9), dtype=torch.bool, device=device)
    
    num_endmembers = 7
    endmembers_tensor = maximumDistance_torch(data_tensor, num_endmembers, valid_pixel_mask) # shape (1, num_bands, 7)
    endmembers_spectra = endmembers_tensor[0].cpu().numpy() # shape (num_bands, 7)
    
    return {
        "val": val,
        "global_idx": global_idx,
        "y": y,
        "x": x,
        "lat": lat,
        "lon": lon,
        "date_str": date_str,
        "grid_str": grid_str,
        "local_idx": local_idx,
        "chip_rgb": chip_rgb,
        "chip_size": chip_size,
        "sr_3x3": sr_3x3,
        "endmembers": endmembers_spectra,
        "wavelengths": wavelengths
    }


def plot_candidate_summary(res, title_prefix, threed_bands, output_path, global_limits=None, origin_method="EM2"):
    """
    Generates a 1-row by 3-column unified visualization for a single candidate:
    [ Ortho Chip | Endmember Spectra | 3D Convex Hull ]
    """
    fig = plt.figure(figsize=(20, 5))
    
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2']
    labels = ['EM 1 (Max Mag)', 'EM 2 (Min Mag)', 'EM 3', 'EM 4', 'EM 5', 'EM 6', 'EM 7']
    
    chip_size = res["chip_size"]
    main_color = '#d62728' if 'High' in title_prefix else '#1f77b4'
    
    # Make sure text mode underscores are escaped if passing raw into LaTeX
    date_str_safe = res['date_str'].replace('_', r'\_')
    grid_str_safe = res['grid_str'].replace('_', r'\_')
    
    lat_dir = "N" if res['lat'] >= 0 else "S"
    lon_dir = "E" if res['lon'] >= 0 else "W"
    coord_str = f"${abs(res['lat']):.5f}^\\circ \\mathrm{{{lat_dir}}}, {abs(res['lon']):.5f}^\\circ \\mathrm{{{lon_dir}}}$"
    
    # =========================================================================
    # Subplot 1: Ortho Visual Chip
    # =========================================================================
    ax_chip = fig.add_subplot(1, 3, 1)
    ax_chip.imshow(res["chip_rgb"], extent=[0, chip_size, chip_size, 0], interpolation='nearest')
    
    half_before = chip_size // 2
    rect_origin = half_before - 1
    rect = patches.Rectangle((rect_origin, rect_origin), 3, 3, linewidth=3, edgecolor='#FF1E1E', facecolor='none', linestyle='-')
    ax_chip.add_patch(rect)
    
    if chip_size <= 21:
        ax_chip.set_xticks(range(chip_size + 1))
        ax_chip.set_yticks(range(chip_size + 1))
    else:
        step = max(1, chip_size // 10)
        ax_chip.set_xticks(range(0, chip_size + 1, step))
        ax_chip.set_yticks(range(0, chip_size + 1, step))
    ax_chip.grid(True, color='white', alpha=0.3, linewidth=1)
    
    ax_chip.set_title(f"{coord_str}\nDate: {date_str_safe}", 
                      fontsize=11, fontweight='bold', pad=10)
    ax_chip.set_xlabel("Chip Columns (Pixels)")
    ax_chip.set_ylabel("Chip Rows (Pixels)")
    
    # =========================================================================
    # Subplot 2: Endmember Spectra
    # =========================================================================
    ax_spec = fig.add_subplot(1, 3, 2)
    wl = res["wavelengths"]
    if np.max(wl) > 100:
        wl_plot = wl / 1000.0  # Convert nm to micrometers
    else:
        wl_plot = wl
    x_label = r"Wavelength ($\mu\mathrm{m}$)"
        
    sort_idx = np.argsort(wl_plot)
    sorted_wl = wl_plot[sort_idx]
    
    for i in range(res["endmembers"].shape[1]):
        em_curve = res["endmembers"][:, i][sort_idx]
        ax_spec.plot(sorted_wl, em_curve, label=labels[i], color=colors[i % len(colors)], linewidth=2, marker='o', markersize=4)
        
    ax_spec.set_title(f"Extracted $3 \\times 3$ Endmember Signatures",#\nSpectral Complexity Value: {res['val']:.6e}", 
                      fontsize=11, fontweight='bold', pad=10)
    ax_spec.set_xlabel(x_label, fontsize=11)
    ax_spec.set_ylabel("Surface Reflectance", fontsize=11)
    ax_spec.grid(True, linestyle='--', alpha=0.5)
    ax_spec.legend(loc='best', framealpha=0.9, fontsize=9)
    
    # =========================================================================
    # Subplot 3: 3D Convex Hull
    # =========================================================================
    ax_3d = fig.add_subplot(1, 3, 3, projection='3d')
    
    em_all = res["endmembers"][list(threed_bands), :].T
    px_3d = res["sr_3x3"][list(threed_bands), :, :].reshape(len(threed_bands), 9).T
    wl_um = wl[list(threed_bands)] / 1000.0 if np.max(wl) > 100 else wl[list(threed_bands)]
    
    hull_vol = 0.0
    hull_faces = []
    
    if origin_method.upper() == "ZERO":
        n = 3
        em_3d = em_all[:n, :]
        em_unused = em_all[n:, :]
        origin_pt = np.array([0.0, 0.0, 0.0])
        basis_vectors = em_3d
        origin_label = 'Origin (0,0,0)'
    else:
        n = 4
        em_3d = em_all[:n, :]
        em_unused = em_all[n:, :]
        # Generate parallelotope originating at Endmember 2 (index 1)
        origin_idx = 1
        origin_pt = em_3d[origin_idx] if len(em_3d) > origin_idx else em_3d[0]
        if len(em_3d) > 1:
            basis_vectors = np.delete(em_3d, origin_idx, axis=0) - origin_pt
        else:
            basis_vectors = np.zeros((0, 3))
        origin_label = 'Origin (EM 2)'
    
    combinations = list(itertools.product([0, 1], repeat=len(basis_vectors)))
    em_hull_pts = []
    for c in combinations:
        em_hull_pts.append(origin_pt + np.dot(c, basis_vectors))
    em_hull_pts = np.array(em_hull_pts)
    
    try:
        hull = ConvexHull(em_hull_pts)
        hull_vol = hull.volume
        for simplex in hull.simplices:
            hull_faces.append(em_hull_pts[simplex])
    except Exception as e:
        print(f"Warning: Could not compute 3D Convex Hull (likely a 2D shape for N=3): {e}")
        
    if hull_faces:
        # Plot faces without triangulated edges
        poly = Poly3DCollection(hull_faces, alpha=0.15, facecolor=main_color, edgecolor='none')
        ax_3d.add_collection3d(poly)
        
    # Draw true structural edges of the parallelotope regardless of 3D volume
    for i, c in enumerate(combinations):
        for j in range(len(basis_vectors)):
            if c[j] == 0:
                c_neighbor = list(c)
                c_neighbor[j] = 1
                neighbor_idx = combinations.index(tuple(c_neighbor))
                
                pt1 = em_hull_pts[i]
                pt2 = em_hull_pts[neighbor_idx]
                ax_3d.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], color=main_color, lw=1.0, alpha=0.6)

    ax_3d.scatter([origin_pt[0]], [origin_pt[1]], [origin_pt[2]], color='black', marker='s', s=80, label=origin_label)
    
    # The original endmembers act as the basis vectors
    ax_3d.scatter(em_3d[:, 0], em_3d[:, 1], em_3d[:, 2], color=main_color, s=120, marker='*', label='Endmember Vertex')     
    ax_3d.scatter(em_hull_pts[:, 0], em_hull_pts[:, 1], em_hull_pts[:, 2], color=main_color, s=40, marker='.', alpha=0.3)#, label='Parallelotope Vertices')
    ax_3d.scatter(px_3d[:, 0], px_3d[:, 1], px_3d[:, 2], color='black', alpha=0.5, s=35, marker='o', label=r'$3 \times 3$ Sample Pixels')
    
    # Scatter Unused Vertices identically to sample pixels
    if len(em_unused) > 0:
        ax_3d.scatter(em_unused[:, 0], em_unused[:, 1], em_unused[:, 2], color='black', alpha=0.5, s=35, marker='o',)#, label='Unused Endmembers')
    
    for i in range(7):
        c = main_color if i < n else '#7f7f7f'
        ax_3d.text(em_all[i, 0], em_all[i, 1], em_all[i, 2], f" EM{i+1}", color=c, fontweight='bold', fontsize=9)
        
    ax_3d.set_title(f"Hypervolume 3D Slice ({n} Endmembers)")#,\n3D Hull Vol: {hull_vol:.6e}")
    
    ax_3d.set_xlabel(fr"Band {threed_bands[0]+1} ({wl_um[0]:.3f} $\mu\mathrm{{m}}$)")
    ax_3d.set_ylabel(fr"Band {threed_bands[1]+1} ({wl_um[1]:.3f} $\mu\mathrm{{m}}$)")
    ax_3d.set_zlabel(fr"Band {threed_bands[2]+1} ({wl_um[2]:.3f} $\mu\mathrm{{m}}$)")
    
    if global_limits is not None:
        ax_3d.set_xlim(global_limits[0])
        ax_3d.set_ylim(global_limits[1])
        ax_3d.set_zlim(global_limits[2])
    else:
        loc_pts = np.vstack([em_3d, px_3d])
        l_min = loc_pts.min(axis=0)
        l_max = loc_pts.max(axis=0)
        l_span = l_max - l_min
        l_pad = np.maximum(l_span * 0.1, 0.002)
        ax_3d.set_xlim(l_min[0] - l_pad[0], l_max[0] + l_pad[0])
        ax_3d.set_ylim(l_min[1] - l_pad[1], l_max[1] + l_pad[1])
        ax_3d.set_zlim(l_min[2] - l_pad[2], l_max[2] + l_pad[2])
    
    ax_3d.grid(True, linestyle='--', alpha=0.4)
    ax_3d.legend(loc='best')
    
    plt.tight_layout()
    fig.subplots_adjust(right=0.95, left=0.05)
    plt.savefig(output_path, dpi=400, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"  -> Saved Visualization: {os.path.abspath(output_path)}")


def plot_encapsulation_series(res, title_prefix, threed_bands, output_path, global_limits=None, origin_method="EM2"):
    """
    Generates a 1-row by 5-column figure showing 3D convex hulls formed by 3, 4, 5, 6, and 7 endmembers.
    Remaining endmembers are plotted as scatter points.
    """
    fig = plt.figure(figsize=(30, 6))
    main_color = '#d62728' if 'High' in title_prefix else '#1f77b4'
    unused_color = '#7f7f7f'
    
    wl = res["wavelengths"]
    wl_um = wl[list(threed_bands)] / 1000.0 if np.max(wl) > 100 else wl[list(threed_bands)]
    
    em_all = res["endmembers"][list(threed_bands), :].T
    px_3d = res["sr_3x3"][list(threed_bands), :, :].reshape(len(threed_bands), 9).T
    
    for n in range(3, 8):
        ax = fig.add_subplot(1, 5, n - 2, projection='3d')
        
        em_hull = em_all[:n, :]
        em_unused = em_all[n:, :]
        
        if origin_method.upper() == "ZERO":
            origin_pt = np.array([0.0, 0.0, 0.0])
            basis_vectors = em_hull
            origin_label = 'Origin (0,0,0)'
        else:
            # Generate parallelotope vertices originating at Endmember 2
            origin_idx = 1
            origin_pt = em_hull[origin_idx] if len(em_hull) > origin_idx else em_hull[0]
            if len(em_hull) > 1:
                basis_vectors = np.delete(em_hull, origin_idx, axis=0) - origin_pt
            else:
                basis_vectors = np.zeros((0, 3))
            origin_label = 'Origin (EM 2)'
            
        combinations = list(itertools.product([0, 1], repeat=len(basis_vectors)))
        em_hull_pts = []
        for c in combinations:
            if len(basis_vectors) > 0:
                em_hull_pts.append(origin_pt + np.dot(c, basis_vectors))
            else:
                em_hull_pts.append(origin_pt)
        em_hull_pts = np.array(em_hull_pts)
        
        hull_vol = 0.0
        hull_faces = []
        try:
            hull = ConvexHull(em_hull_pts)
            hull_vol = hull.volume
            for simplex in hull.simplices:
                hull_faces.append(em_hull_pts[simplex])
        except Exception as e:
            print(f"Warning: Could not compute 3D Convex Hull for {n} endmembers: {e}")
            
        if hull_faces:
            # Plot faces without triangulated edges
            poly = Poly3DCollection(hull_faces, alpha=0.15, facecolor=main_color, edgecolor='none')
            ax.add_collection3d(poly)
            
        # Draw true structural edges of the parallelotope regardless of 3D volume
        for i, c in enumerate(combinations):
            for j in range(len(basis_vectors)):
                if c[j] == 0:
                    c_neighbor = list(c)
                    c_neighbor[j] = 1
                    neighbor_idx = combinations.index(tuple(c_neighbor))
                    
                    pt1 = em_hull_pts[i]
                    pt2 = em_hull_pts[neighbor_idx]
                    ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], color=main_color, lw=1.0, alpha=0.6)
                
        # Scatter Origin
        ax.scatter([origin_pt[0]], [origin_pt[1]], [origin_pt[2]], color='black', marker='s', s=50, label=origin_label)
        
        # Scatter Sample Pixels
        ax.scatter(px_3d[:, 0], px_3d[:, 1], px_3d[:, 2], color='black', alpha=0.5, s=35, marker='o', label=r'$3 \times 3$ Sample Pixels')
        
        # Scatter Hull Vertices (basis) and the full parallelotope corners
        ax.scatter(em_hull[:, 0], em_hull[:, 1], em_hull[:, 2], color=main_color, s=120, marker='*', label='Endmember Basis')
        ax.scatter(em_hull_pts[:, 0], em_hull_pts[:, 1], em_hull_pts[:, 2], color=main_color, s=40, marker='.', alpha=0.3)
        
        # Scatter Unused Vertices
        if len(em_unused) > 0:
            ax.scatter(em_unused[:, 0], em_unused[:, 1], em_unused[:, 2], color='black', alpha=0.5, s=35, marker='o', label='Unused Endmembers')
            
        for i in range(7):
            c = main_color if i < n else unused_color
            ax.text(em_all[i, 0], em_all[i, 1], em_all[i, 2], f" EM{i+1}", color=c, fontweight='bold', fontsize=9)
            
        ax.set_title(
            f"{n} Endmember Encapsulation\n3D Hull Vol: {hull_vol:.6e}",
            fontsize=11, fontweight='bold', pad=15
        )
        
        ax.set_xlabel(fr"Band {threed_bands[0]+1} ({wl_um[0]:.3f} $\mu\mathrm{{m}}$)")
        ax.set_ylabel(fr"Band {threed_bands[1]+1} ({wl_um[1]:.3f} $\mu\mathrm{{m}}$)")
        ax.set_zlabel(fr"Band {threed_bands[2]+1} ({wl_um[2]:.3f} $\mu\mathrm{{m}}$)")
        
        if global_limits is not None:
            ax.set_xlim(global_limits[0])
            ax.set_ylim(global_limits[1])
            ax.set_zlim(global_limits[2])
        else:
            loc_pts = np.vstack([em_all, px_3d])
            l_min = loc_pts.min(axis=0)
            l_max = loc_pts.max(axis=0)
            l_span = l_max - l_min
            l_pad = np.maximum(l_span * 0.1, 0.002)
            ax.set_xlim(l_min[0] - l_pad[0], l_max[0] + l_pad[0])
            ax.set_ylim(l_min[1] - l_pad[1], l_max[1] + l_pad[1])
            ax.set_zlim(l_min[2] - l_pad[2], l_max[2] + l_pad[2])
            
        ax.grid(True, linestyle='--', alpha=0.4)
        if n == 3:
            ax.legend(loc='best', fontsize=8)
            
    plt.tight_layout()
    fig.subplots_adjust(right=0.97, left=0.03)
    plt.savefig(output_path, dpi=400, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    print(f"  -> Saved Encapsulation Series: {os.path.abspath(output_path)}")


def main():
    # =========================================================================
    # CONFIGURABLE VISUALIZATION SETTINGS
    # Adjust CHIP_SIZE below to widen or narrow the spatial context window.
    # THREED_BANDS specifies 0-indexed array band positions for 3D hull plotting
    # (indices 3, 4, 5 correspond to 1-indexed Landsat Bands 4 [Red], 5 [NIR], 6 [SWIR 1]).
    # =========================================================================
    CHIP_SIZE = 13           # Width and height (in pixels) of the visualization chip (e.g., 15x15)
    SEPARATION_PIXELS = 13   # Minimum distance from masked pixels (defaults to 7)
    THREED_BANDS = (3, 4, 5) # Spectral bands for 3D hypervolume vertices visualization
    PARALLELOTOPE_ORIGIN = "ZERO" # Options: "ZERO" for (0,0,0) or "EM2" for Endmember 2
    # Optional manual limits for the 3D plot axes, formatted as ((xmin, xmax), (ymin, ymax), (zmin, zmax))
    # If set to None, the script will automatically calculate the global min/max across all candidates.
    #Example: MANUAL_3D_LIMITS = ((0.0, 0.4), (0.0, 0.5), (0.0, 0.6))
    #MANUAL_3D_LIMITS = None
    MANUAL_3D_LIMITS = ((-0.01, 1.4), (-0.01, 1.4), (-0.01, 1.2))
    Location = 'Tait'
    
    sc_h5_path = fr"C:\satelliteImagery\HLST30\HLST_{Location}_Harmonized_SC_EM-7_Norm-None.h5"
    base_h5_path = fr"C:\satelliteImagery\HLST30\HLST_{Location}_Harmonized.h5"
    
    output_dir = r"C:\satelliteImagery\HLST30\Convex Hull Visuals\Landsat"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(sc_h5_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Spectral complexity HDF5 cube not found at: {sc_h5_path}")
    if not os.path.exists(base_h5_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Base ARD HDF5 cube not found at: {base_h5_path}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Compute Engine initialized on device: {device}")
    
    with h5py.File(sc_h5_path, 'r') as sc_file, h5py.File(base_h5_path, 'r') as base_file:
        top_high_cands, top_low_cands = find_landsat_candidates(
            sc_file, base_file, separation_pixels=SEPARATION_PIXELS, chip_size=CHIP_SIZE, top_n=10
        )
        
        print("\n" + "=" * 70)
        print("STEP 2: EXTRACTING ALL CANDIDATES & COMPUTING GLOBAL SCALES")
        print("=" * 70)
        
        high_results = []
        for cand in top_high_cands:
            high_results.append(extract_and_compute_endmembers(sc_file, base_file, cand, device, chip_size=CHIP_SIZE))
            
        low_results = []
        for cand in top_low_cands:
            low_results.append(extract_and_compute_endmembers(sc_file, base_file, cand, device, chip_size=CHIP_SIZE))
            
        # Calculate global 3D limits for consistent volume comparison
        all_pts = []
        for res in high_results + low_results:
            em_pts = res["endmembers"][list(THREED_BANDS), :].T
            px_pts = res["sr_3x3"][list(THREED_BANDS), :, :].reshape(len(THREED_BANDS), 9).T
            all_pts.extend(em_pts)
            all_pts.extend(px_pts)
        all_pts = np.array(all_pts)
        
        if MANUAL_3D_LIMITS is not None:
            g_lims = MANUAL_3D_LIMITS
            print(f"Using manual global 3D limits: {g_lims}")
        else:
            g_min = all_pts.min(axis=0)
            g_max = all_pts.max(axis=0)
            g_span = g_max - g_min
            g_pad = np.maximum(g_span * 0.1, 0.01)
            g_lims = [ (g_min[i] - g_pad[i], g_max[i] + g_pad[i]) for i in range(3) ]
            print(f"Calculated automatic global 3D limits: {g_lims}")
        
        print("\n" + "=" * 70)
        print("STEP 3: GENERATING PUBLICATION-READY VISUALIZATIONS")
        print("=" * 70)
        
        print("\nProcessing High Spectral Complexity Candidates...")
        for rank, res in enumerate(high_results, 1):
            title_prefix = f"High Spectral Complexity"
            date_str = res['date_str']
            out_name = f"High_SC_Rank{rank}_Landsat_{date_str}.png"
            out_path = os.path.join(output_dir, out_name)
            plot_candidate_summary(res, title_prefix, THREED_BANDS, out_path, global_limits=g_lims, origin_method=PARALLELOTOPE_ORIGIN)
            
            encap_name = f"High_SC_Rank{rank}_Landsat_{date_str}_Encapsulation.png"
            encap_path = os.path.join(output_dir, encap_name)
            plot_encapsulation_series(res, title_prefix, THREED_BANDS, encap_path, global_limits=g_lims, origin_method=PARALLELOTOPE_ORIGIN)
            
        print("\nProcessing Low Spectral Complexity Candidates...")
        for rank, res in enumerate(low_results, 1):
            title_prefix = f"Low Spectral Complexity (Rank {rank})"
            date_str = res['date_str']
            out_name = f"Low_SC_Rank{rank}_Landsat_{date_str}.png"
            out_path = os.path.join(output_dir, out_name)
            plot_candidate_summary(res, title_prefix, THREED_BANDS, out_path, global_limits=g_lims, origin_method=PARALLELOTOPE_ORIGIN)
            
            encap_name = f"Low_SC_Rank{rank}_Landsat_{date_str}_Encapsulation.png"
            encap_path = os.path.join(output_dir, encap_name)
            plot_encapsulation_series(res, title_prefix, THREED_BANDS, encap_path, global_limits=g_lims, origin_method=PARALLELOTOPE_ORIGIN)

    print(f"\nAll visualizations generated successfully in {output_dir}")


if __name__ == "__main__":
    main()

