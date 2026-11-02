#!/usr/bin/env python3
"""
generate_slide1_airborne_comparisons.py
Slide 1 Embedded Visuals: Airborne HySpex Flightline Validation vs. Physical Simulation

This script generates the high-resolution comparison figures for Slide 1 of the
October 27, 2026 Spectral Complexity presentation:
  1. Homogeneous Calibration Panel:
     - Center Coords: (map_x=1059, map_y=258)
     - Output: C:\\satelliteImagery\\simulated\\hyspex_comparison\\hyspex_comparison_real_vs_simulated_1059_258.png
     - Scenario: Pure white panel, plateaus at k* = 1.
  2. Structural Boundary Edge:
     - Center Coords: (map_x=1106, map_y=326)
     - Output: C:\\satelliteImagery\\simulated\\hyspex_comparison\\hyspex_comparison_real_vs_simulated_1106_326.png
     - Scenario: White calibration panel / asphalt tarmac boundary, plateaus at k* = 2.

Designed for IDE execution:
  - Simply open in VS Code / PyCharm and run (Run / F5).
  - Adjust configuration constants below for fine tuning.
"""

import os
import sys

# Ensure repository packages are in Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
HYSPEX_DIR = os.path.join(REPO_ROOT, "hyspex")
SIM_DIR = os.path.join(REPO_ROOT, "simulated-data-test")

for d in [HYSPEX_DIR, SIM_DIR, REPO_ROOT]:
    if d not in sys.path:
        sys.path.insert(0, d)

from compare_real_tile_vs_simulated import (
    retrieve_real_tile,
    generate_simulated_tile,
    plot_comparison,
    display_comparison_table,
    DEFAULT_H5_PATH,
    DEFAULT_OUTPUT_DIR
)

# =============================================================================
# USER CONFIGURATION PARAMETERS (Adjust for fine-tuning in IDE)
# =============================================================================
H5_PATH = DEFAULT_H5_PATH
OUTPUT_DIR = r"C:\satelliteImagery\simulated\hyspex_comparison"

# Window & Algorithmic Parameters
TILE_SIZE = 15                 # Core analysis window: 15x15 pixels
CONTEXT_SIZE = 150             # Airborne context window: 150x150 pixels
NUM_ENDMEMBERS = 10            # Number of MaxD endmembers to extract
SHADE_VARIABILITY = 0.055       # 3% shading noise modeling micro-relief and BRDF
SHOT_NOISE_FRACTION = 0.7      # Shot noise fraction
SNR = 200                      # Sensor signal-to-noise ratio
SEED = 42                      # Random seed for reproducible simulation

# Execution Toggles (Set to True / False as desired)
RUN_HOMOGENEOUS_PANEL = True   # Generates tile (1059, 258)
RUN_BOUNDARY_EDGE = True       # Generates tile (1106, 326)
SHOW_PLOT_WINDOW = False       # If True, displays interactive matplotlib window


def run_homogeneous_panel():
    """Generates comparison figure for the interior white calibration panel (1059, 258)."""
    map_x = 1059
    map_y = 258
    print(f"\n=======================================================")
    print(f"Generating Scenario 1: Homogeneous Panel at ({map_x}, {map_y})")
    print(f"=======================================================")

    # 1. Retrieve Real Airborne Tile
    real_results = retrieve_real_tile(
        h5_path=H5_PATH,
        map_x=map_x,
        map_y=map_y,
        tile_size=TILE_SIZE,
        num_endmembers=NUM_ENDMEMBERS,
        context_size=CONTEXT_SIZE
    )

    # 2. Generate Physical Simulation
    sim_results = generate_simulated_tile(
        sensor_name="HySpexMjolnir",
        material_name="CALTARGET:big_white_panel",
        material_2_name=None,
        material_3_name=None,
        layout="single",
        tile_size=TILE_SIZE,
        snr=SNR,
        num_endmembers=NUM_ENDMEMBERS,
        shot_noise_fraction=SHOT_NOISE_FRACTION,
        shade_variability=SHADE_VARIABILITY,
        seed=SEED
    )

    # 3. Print Summary Table
    display_comparison_table(real_results, sim_results, NUM_ENDMEMBERS)

    # 4. Save High-Res Comparison Plot
    plot_comparison(
        real_res=real_results,
        sim_res=sim_results,
        map_x=map_x,
        map_y=map_y,
        tile_size=TILE_SIZE,
        num_endmembers=NUM_ENDMEMBERS,
        output_dir=OUTPUT_DIR,
        show_plot=SHOW_PLOT_WINDOW
    )

    out_file = os.path.join(OUTPUT_DIR, f"hyspex_comparison_real_vs_simulated_{map_x}_{map_y}.png")
    print(f"[Done] Homogeneous panel comparison saved: {out_file}")
    return out_file


def run_boundary_edge():
    """Generates comparison figure for the structural boundary edge (1106, 326)."""
    map_x = 1108
    map_y = 325
    print(f"\n=======================================================")
    print(f"Generating Scenario 2: Boundary Edge at ({map_x}, {map_y})")
    print(f"=======================================================")

    # 1. Retrieve Real Airborne Tile
    real_results = retrieve_real_tile(
        h5_path=H5_PATH,
        map_x=map_x,
        map_y=map_y,
        tile_size=TILE_SIZE,
        num_endmembers=NUM_ENDMEMBERS,
        context_size=CONTEXT_SIZE
    )

    # 2. Generate Physical Simulation
    sim_results = generate_simulated_tile(
        sensor_name="HySpexMjolnir",
        material_name="CALTARGET:big_white_panel",
        material_2_name="USGS:Asphalt",
        material_3_name=None,
        layout="split_diag",
        tile_size=TILE_SIZE,
        snr=SNR,
        num_endmembers=NUM_ENDMEMBERS,
        shot_noise_fraction=SHOT_NOISE_FRACTION,
        shade_variability=SHADE_VARIABILITY,
        seed=SEED
    )

    # 3. Print Summary Table
    display_comparison_table(real_results, sim_results, NUM_ENDMEMBERS)

    # 4. Save High-Res Comparison Plot
    plot_comparison(
        real_res=real_results,
        sim_res=sim_results,
        map_x=map_x,
        map_y=map_y,
        tile_size=TILE_SIZE,
        num_endmembers=NUM_ENDMEMBERS,
        output_dir=OUTPUT_DIR,
        show_plot=SHOW_PLOT_WINDOW
    )

    out_file = os.path.join(OUTPUT_DIR, f"hyspex_comparison_real_vs_simulated_{map_x}_{map_y}.png")
    print(f"[Done] Boundary edge comparison saved: {out_file}")
    return out_file


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Starting Slide 1 Visual Generation...")
    print(f"Output directory: {OUTPUT_DIR}")

    if RUN_HOMOGENEOUS_PANEL:
        run_homogeneous_panel()

    if RUN_BOUNDARY_EDGE:
        run_boundary_edge()

    print("\nAll requested Slide 1 visuals regenerated successfully.")


if __name__ == "__main__":
    main()
