#!/usr/bin/env python3
"""
rebuild_all_presentation_visuals.py
Master Visual Rebuilder for "2026-10-27 gemini.pptx"

This script provides a single unified entry point in the IDE to sequentially
regenerate all embedded visuals across Slides 1-4, and optionally rebuild the
PowerPoint presentation itself.

Visuals Regenerated:
  - Slide 1:
      • hyspex_comparison_real_vs_simulated_1059_258.png (Homogeneous Panel)
      • hyspex_comparison_real_vs_simulated_1106_326.png (Boundary Edge)
  - Slide 2:
      • foliar_ground_truth_specimens_and_spectra.png (4 photos + ASD spectra)
  - Slide 3:
      • foliar_complexity_healthy_vs_pathological_contrast.png (Contrast plot)
      • foliar_complexity_2mat_healthy.png (Detailed 2-mat scenario)
      • foliar_complexity_4mat_pathological.png (Detailed 4-mat scenario)
  - Slide 4:
      • spectral_complexity_vs_pca_scree.png (PCA scree vs. MaxD residual decays)

Designed for IDE execution:
  - Simply open in VS Code / PyCharm and run (Run / F5).
  - Toggle individual slide regenerations below.
"""

import os
import sys
import time

# Script directory
PRESENTATION_DIR = os.path.dirname(os.path.abspath(__file__))
if PRESENTATION_DIR not in sys.path:
    sys.path.insert(0, PRESENTATION_DIR)

from generate_slide1_airborne_comparisons import run_homogeneous_panel, run_boundary_edge
from generate_slide2_foliar_ground_truth import generate_foliar_ground_truth
from generate_slide3_foliar_complexity import main as run_slide3_simulations
from generate_slide4_pca_vs_complexity import generate_pca_vs_complexity_figure
from build_presentation_slides import build_slides
from build_monthly_report import main as generate_monthly_report

# =============================================================================
# USER CONFIGURATION TOGGLES (Adjust in IDE)
# =============================================================================
REBUILD_SLIDE_1 = True         # Airborne HySpex Flightline vs. Simulation (Tile 1059 & 1106)
REBUILD_SLIDE_2 = True         # Foliar Ground Truth Photos & ASD Spectra
REBUILD_SLIDE_3 = True         # Foliar Complexity Simulations (Healthy vs. Pathological)
REBUILD_SLIDE_4 = True         # PCA vs. MaxD Scree & Residual Decays
UPDATE_PPTX_DECK = True        # Automatically re-insert visuals into "2026-10-27 gemini.pptx"
BUILD_REPORT_DOCS = True       # Generate Word (.docx) & Markdown research reports


def main():
    start_time = time.time()
    print("===================================================================")
    print("      REBUILDING EMBEDDED VISUALS FOR PRESENTATION          ")
    print("===================================================================")

    # 1. Slide 1 Visuals
    if REBUILD_SLIDE_1:
        t0 = time.time()
        print("\n>>> [1/4] Regenerating Slide 1 Visuals (Airborne Comparisons)...")
        run_homogeneous_panel()
        run_boundary_edge()
        print(f">>> Slide 1 Completed in {time.time() - t0:.2f}s")
    else:
        print("\n>>> [1/4] Slide 1 Skipped.")

    # 2. Slide 2 Visuals
    if REBUILD_SLIDE_2:
        t0 = time.time()
        print("\n>>> [2/4] Regenerating Slide 2 Visual (Foliar Ground Truth)...")
        generate_foliar_ground_truth()
        print(f">>> Slide 2 Completed in {time.time() - t0:.2f}s")
    else:
        print("\n>>> [2/4] Slide 2 Skipped.")

    # 3. Slide 3 Visuals
    if REBUILD_SLIDE_3:
        t0 = time.time()
        print("\n>>> [3/4] Regenerating Slide 3 Visuals (Foliar Simulations)...")
        run_slide3_simulations()
        print(f">>> Slide 3 Completed in {time.time() - t0:.2f}s")
    else:
        print("\n>>> [3/4] Slide 3 Skipped.")

    # 4. Slide 4 Visuals
    if REBUILD_SLIDE_4:
        t0 = time.time()
        print("\n>>> [4/4] Regenerating Slide 4 Visual (PCA vs. MaxD Scree)...")
        generate_pca_vs_complexity_figure()
        print(f">>> Slide 4 Completed in {time.time() - t0:.2f}s")
    else:
        print("\n>>> [4/4] Slide 4 Skipped.")

    # 5. Optional PowerPoint Rebuild
    if UPDATE_PPTX_DECK:
        t0 = time.time()
        print("\n>>> [5/5] Rebuilding PowerPoint Deck ('2026-10-27 gemini.pptx')...")
        build_slides()
        print(f">>> Deck Rebuild Completed in {time.time() - t0:.2f}s")
    else:
        print("\n>>> [5/6] PowerPoint Deck Update Skipped.")

    # 6. Monthly Report Generation (.docx & .md)
    if BUILD_REPORT_DOCS:
        t0 = time.time()
        print("\n>>> [6/6] Generating Monthly Research Reports (.docx & .md)...")
        generate_monthly_report()
        print(f">>> Report Generation Completed in {time.time() - t0:.2f}s")
    else:
        print("\n>>> [6/6] Report Generation Skipped.")

    total_time = time.time() - start_time
    print("\n===================================================================")
    print(f"  ALL TASKS COMPLETED SUCCESSFULLY IN {total_time:.2f}s")
    print("===================================================================")


if __name__ == '__main__':
    main()
