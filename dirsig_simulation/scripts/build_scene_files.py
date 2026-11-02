#!/usr/bin/env python3
"""
DIRSIG5 Scene & Material Generator for Flat-Tile Remote Sensing Simulations
==========================================================================
Generates a complete DIRSIG5 scene on the local machine:
  1. Material spectral curves (.ems) from USGS splib07 ASD, FieldSpec leaf data, and calibration targets.
  2. Master material database (materials.mat) with radiometric and thermal properties.
  3. 3D Wavefront geometry (.obj) for:
     a) 12-tile grid layout (10m x 10m tiles + ground plane)
     b) Concentric nested squares layout matching simulate_inset_squares.py
  4. Geometry list XML files (.glist) mapping OBJ usemtl identifiers to DIRSIG material IDs.
  5. Top-level DIRSIG scene XML files (.scene) georeferenced to RIT Tait Preserve / ROCX 2025.
"""

import os
import sys
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup

# --- Path Configurations ---
WORKSPACE_DIR = Path(r"F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\dirsig_simulation")
GEOMETRY_DIR = WORKSPACE_DIR / "geometry"
MATERIALS_DIR = WORKSPACE_DIR / "materials"
SCENES_DIR = WORKSPACE_DIR / "scenes"
SCRIPTS_DIR = WORKSPACE_DIR / "scripts"

SPLIB07_HTML_PATH = r"C:\satelliteImagery\ground_truth\splib07\indexes\datatable_splib07b_cvASD.html"
SPLIB07_WV_FILE = r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata\ASCIIdata_splib07b_cvASD\s07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"
NICKS_LEAVES_DIR = Path(r"C:\satelliteImagery\ground_truth\NickLeaves_20261001")
CALIBRATION_DIR = Path(r"C:\satelliteImagery\ground_truth\calibration_targets")

# Georeference origin: RIT Tait Preserve (Rochester, NY)
ORIGIN_LAT = 43.084400
ORIGIN_LON = -77.674300
ORIGIN_ALT = 168.00  # meters above sea level


def lookup_splib07_path(html_path, search_str, soup=None):
    """Parses splib07 HTML index to locate ASD ASCII spectrum path."""
    if soup is None:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f, 'html.parser')
            
    for tr in soup.find_all('tr'):
        th_or_td = tr.find(lambda tag: tag.name in ['th', 'td'] and tag.get('scope') == 'row')
        if th_or_td and search_str.lower() in th_or_td.get_text().lower():
            for a in tr.find_all('a'):
                if 'Spectrum Data' in a.get_text():
                    href = a.get('href')
                    return Path(html_path).parent / href
    raise ValueError(f"Could not find splib07 entry for: {search_str}")


def load_splib07_asd(filepath):
    """Loads 2151-channel ASD spectrum, returning float array with NaNs for invalid channels."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        val = float(s)
        data.append(np.nan if val < -1.0 else val)
    return np.array(data, dtype=np.float64)


def load_splib07_wavelengths(wv_filepath):
    """Loads 2151-channel ASD wavelengths (already in microns)."""
    with open(wv_filepath, 'r') as f:
        lines = f.readlines()
    wvs = []
    for line in lines[1:]:
        s = line.strip()
        if not s:
            continue
        wvs.append(float(s))
    return np.array(wvs, dtype=np.float64)


def load_leaf_csv(csv_path):
    """Loads ASD leaf spectrum from CSV (wavelength in nm, DHR_mean in 0-1)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    # Wavelength in CSV is nm -> convert to microns
    wvs = (df['wavelength'].values / 1000.0).astype(np.float64)
    ref = df['DHR_mean'].values.astype(np.float64)
    return wvs, ref


def load_cal_target_txt(txt_path):
    """Loads calibration panel spectrum (Wavelength_nm, Reflectance_0_1)."""
    wvs = []
    refs = []
    with open(txt_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#') or s.startswith('Wavelength'):
                continue
            parts = s.split()
            if len(parts) >= 2:
                try:
                    w_nm = float(parts[0])
                    r_val = float(parts[1])
                    wvs.append(w_nm / 1000.0)  # Convert nm to microns
                    refs.append(r_val)
                except ValueError:
                    continue
    return np.array(wvs, dtype=np.float64), np.array(refs, dtype=np.float64)


def write_ems_file(ems_path, wavelengths_um, reflectance):
    """
    Writes a DIRSIG classic emissivity (.ems) file using the modern micron format.
    DIRSIG calculates directional-hemispherical reflectance as DHR = 1.0 - DHE.
    By setting emissivity = 1.0 - reflectance, ray-tracing precisely reproduces measured reflectance.
    
    Channels with NaN reflectance are omitted rather than filled with synthetic values,
    preserving raw measurement fidelity (DIRSIG natively supports non-uniform spectral spacing).
    """
    valid_mask = ~np.isnan(reflectance) & ~np.isnan(wavelengths_um) & (reflectance >= 0.0) & (reflectance <= 1.0)
    w_valid = wavelengths_um[valid_mask]
    r_valid = reflectance[valid_mask]
    
    # Ensure strictly increasing order
    sort_idx = np.argsort(w_valid)
    w_valid = w_valid[sort_idx]
    r_valid = r_valid[sort_idx]
    
    # Calculate emissivity
    emissivity = np.clip(1.0 - r_valid, 0.0, 1.0)
    
    with open(ems_path, 'w') as f:
        # Header: total curve count (1)
        f.write("1\n")
        # 91 lines of angular weighting (0 to 90 degrees, set to isotropic 1.0)
        for _ in range(91):
            f.write("1.0\n")
        # Curve 0
        f.write("CURVE_BEGIN 0\n")
        for w, e in zip(w_valid, emissivity):
            f.write(f"{w:.6f}\t{e:.6f}\n")


def write_reflectance_txt(txt_path, wavelengths_um, reflectance):
    """Writes a 2-column ASCII reflectance file (microns vs 0-1 reflectance)."""
    valid_mask = ~np.isnan(reflectance) & ~np.isnan(wavelengths_um)
    w_valid = wavelengths_um[valid_mask]
    r_valid = reflectance[valid_mask]
    with open(txt_path, 'w') as f:
        f.write("# Wavelength_um\tReflectance\n")
        for w, r in zip(w_valid, r_valid):
            f.write(f"{w:.6f}\t{r:.6f}\n")


# --- Material Definition Catalog ---
MATERIAL_CATALOG = [
    # ID, Key, Display Name, Type, Source Name / File
    (100, "ground_soil", "Neutral Background Ground", "cal", "10_percent_gray_panel.txt"),
    (101, "oak_leaf", "Oak Oak-Leaf-1", "splib07", "Oak Oak-Leaf-1"),
    (102, "lodgepole_pine", "Lodgepole-Pine", "splib07", "Lodgepole-Pine"),
    (103, "aspen1", "Aspen Aspen-1", "splib07", "Aspen Aspen-1"),
    (104, "aspen4", "Aspen Aspen-4", "splib07", "Aspen Aspen-4"),
    (105, "douglas_fir", "Douglas-Fir", "splib07", "Douglas-Fir"),
    (106, "conifer_meadow", "Conifer-Meadow", "splib07", "Conifer-Meadow"),
    (107, "maple_leaves", "Maple Leaves", "splib07", "Maple_Leaves"),
    (108, "goethite_quartz", "Goethite 0.02 + Quartz", "splib07", "Goethite0.02+Quartz"),
    (109, "healthy_leaf", "Norway Maple Healthy Leaf", "leaf", "HealthyLeaf1.csv"),
    (110, "diseased_leaf", "Norway Maple Tar Spot Diseased", "leaf", "DiseaseLeaf1.csv"),
    (111, "cal_white", "Big White Calibration Panel", "cal", "big_white_panel.txt"),
    (112, "cal_50gray", "50 Percent Calibration Panel", "cal", "50_percent_cal_panel.txt"),
    (113, "cal_10gray", "10 Percent Gray Calibration Panel", "cal", "10_percent_gray_panel.txt"),
]


def generate_all_materials():
    """Extracts spectra and produces .ems, .txt, and materials.mat files."""
    print("Loading splib07 ASD wavelength axis...")
    splib_wvs = load_splib07_wavelengths(SPLIB07_WV_FILE)
    
    with open(SPLIB07_HTML_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        splib_soup = BeautifulSoup(f, 'html.parser')

    mat_entries = []

    for mat_id, key, name, mat_type, source_ref in MATERIAL_CATALOG:
        print(f"Processing material [{mat_id}] {name}...")
        ems_file = f"{key}.ems"
        ems_path = MATERIALS_DIR / ems_file
        txt_path = MATERIALS_DIR / f"{key}_reflectance.txt"

        if mat_type == "splib07":
            spec_path = lookup_splib07_path(SPLIB07_HTML_PATH, source_ref, splib_soup)
            ref = load_splib07_asd(spec_path)
            wvs = splib_wvs
        elif mat_type == "leaf":
            csv_path = NICKS_LEAVES_DIR / source_ref
            wvs, ref = load_leaf_csv(csv_path)
        elif mat_type == "cal":
            cal_path = CALIBRATION_DIR / source_ref
            wvs, ref = load_cal_target_txt(cal_path)
        else:
            raise ValueError(f"Unknown material type: {mat_type}")

        # Write .ems and .txt
        write_ems_file(ems_path, wvs, ref)
        write_reflectance_txt(txt_path, wvs, ref)

        # Assemble DIRSIG .mat entry
        mat_entry = f"""MATERIAL_ENTRY {{
  NAME = {name}
  ID = {mat_id}
  SPECIFIC_HEAT = 0.5000
  THERMAL_CONDUCTIVITY = 0.5000
  MASS_DENSITY = 0.5000
  SPECULARITY = 0.0000
  SOLAR_ABSORPTION = 0.7000
  THERMAL_EMISSIVITY = 0.9500
  EXPOSED_AREA = 1.0000
  EMISSIVITY_FILE = {ems_file}
  THICKNESS = 1.0000
}}"""
        mat_entries.append(mat_entry)

    # Write master materials.mat
    mat_file_path = MATERIALS_DIR / "materials.mat"
    with open(mat_file_path, 'w') as f:
        f.write("# DIRSIG Master Material Database for Spectral Complexity Simulation\n")
        f.write("# Automatically generated by build_scene_files.py\n\n")
        f.write("\n\n".join(mat_entries))
        f.write("\n")
    print(f"Saved material database: {mat_file_path}")


def generate_flat_tiles_grid_obj():
    """
    Generates a 3D Wavefront .obj geometry containing:
      - 1 large background ground plane (60m x 50m at Z = -0.01m)
      - 12 distinct material tiles (each 10m x 10m at Z = 0.0m) in a 4 x 3 grid with 4m borders.
    """
    obj_path = GEOMETRY_DIR / "flat_tiles_grid.obj"
    
    # 4 columns x 3 rows grid
    # Tile size = 10m, gap = 4m
    tile_size = 10.0
    gap = 4.0
    
    # Ground plane dimensions
    ground_x_min = -gap
    ground_x_max = 4 * tile_size + 3 * gap + gap  # 40 + 12 + 4 = 56m
    ground_y_min = -gap
    ground_y_max = 3 * tile_size + 2 * gap + gap  # 30 + 8 + 4 = 42m
    ground_z = -0.01  # Slightly below tiles to avoid coplanar artifacts

    tile_layout = [
        # (col, row, material_key, label)
        (0, 0, "oak_leaf", "Tile_OakLeaf"),
        (1, 0, "lodgepole_pine", "Tile_LodgepolePine"),
        (2, 0, "aspen1", "Tile_Aspen1"),
        (3, 0, "aspen4", "Tile_Aspen4"),
        (0, 1, "douglas_fir", "Tile_DouglasFir"),
        (1, 1, "conifer_meadow", "Tile_ConiferMeadow"),
        (2, 1, "maple_leaves", "Tile_MapleLeaves"),
        (3, 1, "goethite_quartz", "Tile_GoethiteQuartz"),
        (0, 2, "healthy_leaf", "Tile_HealthyLeaf"),
        (1, 2, "diseased_leaf", "Tile_DiseasedLeaf"),
        (2, 2, "cal_50gray", "Tile_Cal50Gray"),
        (3, 2, "cal_10gray", "Tile_Cal10Gray"),
    ]

    lines = []
    lines.append("# Wavefront OBJ: Flat Tiles Grid Scene for DIRSIG5")
    lines.append("# Units: Meters (ENU frame)")
    lines.append("vn 0.0000 0.0000 1.0000")  # Upward normal for all flat tiles
    lines.append("vt 0.0 0.0")
    lines.append("vt 1.0 0.0")
    lines.append("vt 1.0 1.0")
    lines.append("vt 0.0 1.0")
    lines.append("")

    v_idx = 1

    # 1. Background Ground Plane
    lines.append("g ground_plane")
    lines.append("usemtl ground_soil")
    lines.append(f"v {ground_x_min:.4f} {ground_y_min:.4f} {ground_z:.4f}")
    lines.append(f"v {ground_x_max:.4f} {ground_y_min:.4f} {ground_z:.4f}")
    lines.append(f"v {ground_x_max:.4f} {ground_y_max:.4f} {ground_z:.4f}")
    lines.append(f"v {ground_x_min:.4f} {ground_y_max:.4f} {ground_z:.4f}")
    lines.append(f"f {v_idx}/1/1 {v_idx+1}/2/1 {v_idx+2}/3/1 {v_idx+3}/4/1")
    lines.append("")
    v_idx += 4

    # 2. Material Tiles
    for col, row, mat_key, label in tile_layout:
        x0 = col * (tile_size + gap)
        x1 = x0 + tile_size
        y0 = row * (tile_size + gap)
        y1 = y0 + tile_size
        z = 0.0

        lines.append(f"g {label}")
        lines.append(f"usemtl {mat_key}")
        lines.append(f"v {x0:.4f} {y0:.4f} {z:.4f}")
        lines.append(f"v {x1:.4f} {y0:.4f} {z:.4f}")
        lines.append(f"v {x1:.4f} {y1:.4f} {z:.4f}")
        lines.append(f"v {x0:.4f} {y1:.4f} {z:.4f}")
        lines.append(f"f {v_idx}/1/1 {v_idx+1}/2/1 {v_idx+2}/3/1 {v_idx+3}/4/1")
        lines.append("")
        v_idx += 4

    with open(obj_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved grid geometry OBJ: {obj_path}")


def generate_concentric_tiles_obj():
    """
    Generates a 3D Wavefront .obj geometry with concentric nested squares,
    directly matching simulate_inset_squares.py concentric layer layout.
    """
    obj_path = GEOMETRY_DIR / "concentric_tiles.obj"
    
    # Concentric sizes in meters, centered at (0, 0)
    layers = [
        (100.0, 0.000, "oak_leaf", "Layer1_OakLeaf_100m"),
        (70.0,  0.001, "lodgepole_pine", "Layer2_Pine_70m"),
        (50.0,  0.002, "aspen1", "Layer3_Aspen1_50m"),
        (30.0,  0.003, "douglas_fir", "Layer4_DouglasFir_30m"),
        (15.0,  0.004, "goethite_quartz", "Layer5_GoethiteQuartz_15m"),
        (5.0,   0.005, "cal_white", "Layer6_CalWhite_5m"),
    ]

    lines = []
    lines.append("# Wavefront OBJ: Concentric Inset Squares for DIRSIG5")
    lines.append("# Units: Meters (ENU frame)")
    lines.append("vn 0.0000 0.0000 1.0000")
    lines.append("vt 0.0 0.0")
    lines.append("vt 1.0 0.0")
    lines.append("vt 1.0 1.0")
    lines.append("vt 0.0 1.0")
    lines.append("")

    v_idx = 1
    for size, z, mat_key, label in layers:
        half = size / 2.0
        lines.append(f"g {label}")
        lines.append(f"usemtl {mat_key}")
        lines.append(f"v {-half:.4f} {-half:.4f} {z:.4f}")
        lines.append(f"v {half:.4f} {-half:.4f} {z:.4f}")
        lines.append(f"v {half:.4f} {half:.4f} {z:.4f}")
        lines.append(f"v {-half:.4f} {half:.4f} {z:.4f}")
        lines.append(f"f {v_idx}/1/1 {v_idx+1}/2/1 {v_idx+2}/3/1 {v_idx+3}/4/1")
        lines.append("")
        v_idx += 4

    with open(obj_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved concentric geometry OBJ: {obj_path}")


def generate_glist_files():
    """Generates DIRSIG5 XML geometry list files (.glist) mapping usemtl to material IDs."""
    # Key-to-ID lookup
    key_to_id = {key: mat_id for mat_id, key, name, m_type, s_ref in MATERIAL_CATALOG}

    # 1. Grid GLIST
    grid_glist_path = GEOMETRY_DIR / "flat_tiles_grid.glist"
    grid_assigns = "\n".join([f'            <assign id="{key}">{mat_id}</assign>' for key, mat_id in key_to_id.items()])

    grid_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<geometrylist enabled="true">
    <object name="FlatTilesGridObject">
        <basegeometry>
            <obj>
                <filename>$SCENE_DIR/geometry/flat_tiles_grid.obj</filename>
{grid_assigns}
            </obj>
        </basegeometry>
        <staticinstance>
            <translation>
                <point>
                    <x>0.0000</x>
                    <y>0.0000</y>
                    <z>0.0000</z>
                </point>
            </translation>
            <rotation>
                <cartesiantriple>
                    <x>0.0000</x>
                    <y>0.0000</y>
                    <z>0.0000</z>
                </cartesiantriple>
            </rotation>
        </staticinstance>
    </object>
</geometrylist>
"""
    with open(grid_glist_path, 'w') as f:
        f.write(grid_xml)
    print(f"Saved grid GLIST: {grid_glist_path}")

    # 2. Concentric GLIST
    concentric_glist_path = GEOMETRY_DIR / "concentric_tiles.glist"
    concentric_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<geometrylist enabled="true">
    <object name="ConcentricTilesObject">
        <basegeometry>
            <obj>
                <filename>$SCENE_DIR/geometry/concentric_tiles.obj</filename>
{grid_assigns}
            </obj>
        </basegeometry>
        <staticinstance>
            <translation>
                <point>
                    <x>0.0000</x>
                    <y>0.0000</y>
                    <z>0.0000</z>
                </point>
            </translation>
            <rotation>
                <cartesiantriple>
                    <x>0.0000</x>
                    <y>0.0000</y>
                    <z>0.0000</z>
                </cartesiantriple>
            </rotation>
        </staticinstance>
    </object>
</geometrylist>
"""
    with open(concentric_glist_path, 'w') as f:
        f.write(concentric_xml)
    print(f"Saved concentric GLIST: {concentric_glist_path}")


def generate_scene_files():
    """Generates DIRSIG5 XML classicscene files (.scene) referencing geometry and materials."""
    # 1. Grid Scene
    grid_scene_path = SCENES_DIR / "flat_tiles_grid.scene"
    grid_scene_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<classicscene name="Flat Tiles Grid Scene">
  <metadata>
    <entry>
      <name>Description</name>
      <value><![CDATA[Flat 12-tile material grid scene for cross-sensor spectral complexity and VSNR simulation.]]></value>
    </entry>
    <entry>
      <name>Location</name>
      <value><![CDATA[RIT Tait Preserve / ROCX 2025 Test Site (Rochester, NY)]]></value>
    </entry>
  </metadata>

  <sceneorigin>
    <location frame="geodetic">
      <latitude>{ORIGIN_LAT:.6f}</latitude>
      <longitude>{ORIGIN_LON:.6f}</longitude>
      <altitude>{ORIGIN_ALT:.2f}</altitude>
    </location>
  </sceneorigin>

  <odbdirectory>$SCENE_DIR/geometry</odbdirectory>
  <gdbdirectory>$SCENE_DIR/geometry</gdbdirectory>

  <geometrylist enabled="true">
    <geometrylistinclude name="TileGrid" enabled="true">geometry/flat_tiles_grid.glist</geometrylistinclude>
  </geometrylist>

  <matfilename>$SCENE_DIR/materials/materials.mat</matfilename>
  <emsdirectory>$SCENE_DIR/materials</emsdirectory>
  <extdirectory>$SCENE_DIR/materials</extdirectory>
  <absdirectory>$SCENE_DIR/materials</absdirectory>
  <srcdirectory>$SCENE_DIR/materials</srcdirectory>
</classicscene>
"""
    with open(grid_scene_path, 'w') as f:
        f.write(grid_scene_xml)
    print(f"Saved grid scene: {grid_scene_path}")

    # 2. Concentric Scene
    concentric_scene_path = SCENES_DIR / "concentric_tiles.scene"
    concentric_scene_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<classicscene name="Concentric Inset Squares Scene">
  <metadata>
    <entry>
      <name>Description</name>
      <value><![CDATA[Concentric inset squares scene matching simulate_inset_squares.py simulation layout.]]></value>
    </entry>
    <entry>
      <name>Location</name>
      <value><![CDATA[RIT Tait Preserve / ROCX 2025 Test Site (Rochester, NY)]]></value>
    </entry>
  </metadata>

  <sceneorigin>
    <location frame="geodetic">
      <latitude>{ORIGIN_LAT:.6f}</latitude>
      <longitude>{ORIGIN_LON:.6f}</longitude>
      <altitude>{ORIGIN_ALT:.2f}</altitude>
    </location>
  </sceneorigin>

  <odbdirectory>$SCENE_DIR/geometry</odbdirectory>
  <gdbdirectory>$SCENE_DIR/geometry</gdbdirectory>

  <geometrylist enabled="true">
    <geometrylistinclude name="ConcentricTiles" enabled="true">geometry/concentric_tiles.glist</geometrylistinclude>
  </geometrylist>

  <matfilename>$SCENE_DIR/materials/materials.mat</matfilename>
  <emsdirectory>$SCENE_DIR/materials</emsdirectory>
  <extdirectory>$SCENE_DIR/materials</extdirectory>
  <absdirectory>$SCENE_DIR/materials</absdirectory>
  <srcdirectory>$SCENE_DIR/materials</srcdirectory>
</classicscene>
"""
    with open(concentric_scene_path, 'w') as f:
        f.write(concentric_scene_xml)
    print(f"Saved concentric scene: {concentric_scene_path}")


def main():
    print("=== DIRSIG5 Local Scene Builder Starting ===")
    generate_all_materials()
    generate_flat_tiles_grid_obj()
    generate_concentric_tiles_obj()
    generate_glist_files()
    generate_scene_files()
    print("=== DIRSIG5 Local Scene Builder Complete ===")


if __name__ == "__main__":
    main()
