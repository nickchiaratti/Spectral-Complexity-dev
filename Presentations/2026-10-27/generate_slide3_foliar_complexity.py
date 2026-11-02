#!/usr/bin/env python3
"""
generate_slide3_foliar_complexity.py
Slide 3 Embedded Visual: Foliar Complexity Simulations: Healthy vs. Pathological Diversity

This script generates the high-resolution foliar complexity simulation figures for
Slide 3 of the October 27, 2026 Spectral Complexity presentation:
  1. Primary Slide 3 Embedded Visual:
     - Output: C:\\satelliteImagery\\simulated\\foliar_comparison\\foliar_complexity_healthy_vs_pathological_contrast.png
     - Features: Direct side-by-side comparison of Healthy 2-Material Tile (k* = 2) vs.
       Pathological 4-Material Tile (k* = 4), showing symlog-scaled orthogonal subspace
       distances (h_k) with the 1% physical discriminability threshold (h_crit = 0.010 * sqrt(B))
       and cumulative parallelotope volume trajectories.
  2. Optional Detailed Scenario Visuals (6-panel publication layouts):
     - Healthy 2-Material Tile: foliar_complexity_2mat_healthy.png
     - Pathological 4-Material Tile: foliar_complexity_4mat_pathological.png

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

from simulate_foliar_complexity import (
    simulate_foliar_tile,
    plot_contrast_figure,
    plot_foliar_scenario,
    display_comparison_table,
    DEFAULT_OUTPUT_DIR
)

# =============================================================================
# USER CONFIGURATION PARAMETERS (Adjust for fine-tuning in IDE)
# =============================================================================
OUTPUT_DIR = r"C:\satelliteImagery\simulated\foliar_comparison"

# Simulation Parameters
TILE_SIZE = 15                 # 15x15 pixels (225 pure pixels)
SNR = 200                      # Sensor signal-to-noise ratio
SHOT_NOISE_FRACTION = 0.7      # Shot noise fraction
SHADE_VARIABILITY = 0.06       # 3% shading noise: np.random.uniform(0.97, 1.0)
NUM_ENDMEMBERS = 10            # Number of MaxD endmembers to extract
SEED = 42                      # Random seed for reproducible simulation
SENSOR_NAME = "cvHySpexMjolnir"# Resampled sensor configuration

# Figure Generation Toggles
GENERATE_CONTRAST_FIGURE = True           # Primary Slide 3 graphic
GENERATE_INDIVIDUAL_DETAILED_PLOTS = True # Detailed 6-panel supplemental plots
SHOW_PLOT_WINDOW = False                  # Display interactive matplotlib window


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n=======================================================")
    print(f"Generating Slide 3 Visual: Foliar Complexity Simulations")
    print(f"=======================================================")
    print(f"Output directory: {OUTPUT_DIR}")

    # 1. Simulate Scenario 1: Healthy 2-Material Tile (Norway vs. Silver Maple)
    print("\n--- Running Simulation: Healthy 2-Material Tile (Scenario 1) ---")
    res_2mat = simulate_foliar_tile(
        scenario="healthy_2mat",
        tile_size=TILE_SIZE,
        snr=SNR,
        shot_noise_fraction=SHOT_NOISE_FRACTION,
        shade_variability=SHADE_VARIABILITY,
        num_endmembers=NUM_ENDMEMBERS,
        seed=SEED,
        sensor_name=SENSOR_NAME
    )

    # 2. Simulate Scenario 2: Pathological 4-Material Tile (50% Tar Spot replacement)
    print("\n--- Running Simulation: Pathological 4-Material Tile (Scenario 2) ---")
    res_4mat = simulate_foliar_tile(
        scenario="pathological_4mat",
        tile_size=TILE_SIZE,
        snr=SNR,
        shot_noise_fraction=SHOT_NOISE_FRACTION,
        shade_variability=SHADE_VARIABILITY,
        num_endmembers=NUM_ENDMEMBERS,
        seed=SEED,
        sensor_name=SENSOR_NAME
    )

    # 3. Print Summary Comparison Table in Console
    print("\n--- Quantitative Foliar Metrics Comparison ---")
    display_comparison_table(res_2mat, res_4mat)

    # 4. Generate Primary Contrast Figure for Slide 3
    if GENERATE_CONTRAST_FIGURE:
        contrast_out = os.path.join(OUTPUT_DIR, "foliar_complexity_healthy_vs_pathological_contrast.png")
        print(f"\nGenerating primary contrast figure: {contrast_out}")
        plot_contrast_figure(res_2mat, res_4mat, contrast_out, show_plot=SHOW_PLOT_WINDOW)
        print(f"[Done] Slide 3 primary contrast figure saved: {contrast_out}")

    # 5. Generate Individual Detailed 6-Panel Figures
    if GENERATE_INDIVIDUAL_DETAILED_PLOTS:
        plot_2mat_out = os.path.join(OUTPUT_DIR, "foliar_complexity_2mat_healthy.png")
        print(f"\nGenerating detailed 2-material plot: {plot_2mat_out}")
        plot_foliar_scenario(res_2mat, plot_2mat_out, show_plot=SHOW_PLOT_WINDOW)
        print(f"[Done] Detailed 2-mat plot saved: {plot_2mat_out}")

        plot_4mat_out = os.path.join(OUTPUT_DIR, "foliar_complexity_4mat_pathological.png")
        print(f"\nGenerating detailed 4-material plot: {plot_4mat_out}")
        plot_foliar_scenario(res_4mat, plot_4mat_out, show_plot=SHOW_PLOT_WINDOW)
        print(f"[Done] Detailed 4-mat plot saved: {plot_4mat_out}")

    print("\nAll requested Slide 3 visuals regenerated successfully.\n")


if __name__ == "__main__":
    main()
