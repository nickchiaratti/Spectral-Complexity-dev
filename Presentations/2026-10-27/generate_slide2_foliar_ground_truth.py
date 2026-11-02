#!/usr/bin/env python3
"""
generate_slide2_foliar_ground_truth.py
Slide 2 Embedded Visual: Empirical Foliar Ground Truth & ASD Reflectance Characterization

This script generates the composite figure for Slide 2 of the
October 27, 2026 Spectral Complexity presentation:
  - Output: C:\\satelliteImagery\\simulated\\foliar_comparison\\foliar_ground_truth_specimens_and_spectra.png
  - Row 1: 4 Leaf Specimen Photographs:
      (a) Norway Maple (Acer platanoides) - Healthy
      (b) Norway Maple (Acer platanoides) - Rhytisma acerinum (Tar Spot)
      (c) Silver Maple (Acer saccharinum) - Healthy
      (d) Silver Maple (Acer saccharinum) - Rhytisma acerinum (Tar Spot)
  - Row 2: Calibrated ASD FieldSpec Directional Hemispherical Reflectance (DHR)
      spectra (400-2500 nm) with +/- 1sigma measurement variability envelopes.
      Telluric atmospheric water absorption bands (1350-1450 nm & 1800-1950 nm)
      are strictly omitted (masked as NaN) to preserve radiometric integrity.

Designed for IDE execution:
  - Simply open in VS Code / PyCharm and run (Run / F5).
  - Adjust configuration constants below for fine tuning.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# =============================================================================
# USER CONFIGURATION PARAMETERS (Adjust for fine-tuning in IDE)
# =============================================================================
DATA_DIR = r"C:\satelliteImagery\ground_truth\NickLeaves_20261001"
OUTPUT_DIR = r"C:\satelliteImagery\simulated\foliar_comparison"
OUTPUT_FILENAME = "foliar_ground_truth_specimens_and_spectra.png"

# Figure Layout & Resolution
FIGURE_SIZE = (13, 13)         # Figure dimensions in inches (width, height)
DPI = 300                      # Output resolution (dots per inch)
HEIGHT_RATIOS = [1.1, 1.0]     # Relative height: [Photo row, Spectral row]
HSPACE = 0.28                  # Vertical spacing between rows
WSPACE = 0.15                  # Horizontal spacing between photo subplots

# Typography & Sizing
TITLE_FONT_SIZE = 13           # Main plot title
SUBTITLE_FONT_SIZE = 11        # Specimen photo subheadings
AXIS_LABEL_FONT_SIZE = 12      # X/Y axis label font size
TICK_FONT_SIZE = 10.5          # Axis tick font size
LEGEND_FONT_SIZE = 9.5         # Legend font size
ANNOTATION_FONT_SIZE = 8.5     # Callout box font size

# Spectral Bounds & Radiometric Integrity
WAVELENGTH_MIN = 380           # nm
WAVELENGTH_MAX = 2450          # nm
REFLECTANCE_MAX = 0.65         # Y-axis upper limit
TELLURIC_RANGES = [
    (1350, 1450),              # Telluric Water Absorption Band 1
    (1800, 1950)               # Telluric Water Absorption Band 2
]

# Specimen Metadata
SPECIMENS = [
    {
        'file': 'PXL_20261001_182438231.MP.jpg',
        'title': 'Norway Maple (Healthy)',#\nAcer platanoides',
        'key': 'HealthyLeaf1',
        'color': '#2ca02c',    # Green
        'line_style': '-'
    },
    {
        'file': 'PXL_20261001_183514812.MP.jpg',
        'title': 'Norway Maple (Tar Spot)',#\nRhytisma acerinum',
        'key': 'DiseaseLeaf1',
        'color': '#8c564b',    # Brown/Amber
        'line_style': '--'
    },
    {
        'file': 'PXL_20261001_184723138.MP.jpg',
        'title': 'Silver Maple (Healthy)',#\nAcer saccharinum',
        'key': 'HealthyLeaf2',
        'color': '#1f77b4',    # Blue
        'line_style': '-'
    },
    {
        'file': 'PXL_20261001_185347528.MP.jpg',
        'title': 'Silver Maple (Tar Spot)',#\nRhytisma acerinum',
        'key': 'DiseaseLeaf2',
        'color': '#d62728',    # Red
        'line_style': '--'
    }
]


def generate_foliar_ground_truth():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    print(f"\n=======================================================")
    print(f"Generating Slide 2 Visual: Foliar Ground Truth & Spectra")
    print(f"=======================================================")

    fig = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    gs = gridspec.GridSpec(2, 4, height_ratios=HEIGHT_RATIOS, hspace=HSPACE, wspace=WSPACE)

    # 1. Row 1: Leaf Specimen Photographs
    print("Loading and rendering leaf specimen photographs...")
    for i, meta in enumerate(SPECIMENS):
        ax_img = fig.add_subplot(gs[0, i])
        img_path = os.path.join(DATA_DIR, meta['file'])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Leaf photo not found: {img_path}")
            
        img = Image.open(img_path)
        ax_img.imshow(img)
        ax_img.set_title(meta['title'], fontsize=SUBTITLE_FONT_SIZE, fontweight='bold', 
                         color=meta['color'], pad=8)
        ax_img.axis('off')
        
        # Add colored border matching spectral plot
        for spine in ax_img.spines.values():
            spine.set_visible(True)
            spine.set_color(meta['color'])
            spine.set_linewidth(2)

    # 2. Row 2: Calibrated ASD Reflectance Spectra
    print("Loading and plotting empirical ASD pure-pixel reflectance spectra...")
    ax_spec = fig.add_subplot(gs[1, :])

    for meta in SPECIMENS:
        csv_path = os.path.join(DATA_DIR, f"cvASD_{meta['key']}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ASD CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        wl = df['wavelength'].values
        mean = df['DHR_mean'].values.copy()
        std = df['DHR_stdev'].values.copy()

        # Strictly omit telluric bands by setting them to NaN (preserves raw integrity without interpolation)
        for t_min, t_max in TELLURIC_RANGES:
            mask = (wl >= t_min) & (wl <= t_max)
            mean[mask] = np.nan
            std[mask] = np.nan

        label_clean = meta['title'].replace('\n', ' - ')
        ax_spec.plot(wl, mean, label=label_clean, color=meta['color'], 
                     linestyle=meta['line_style'], linewidth=2.0)
        ax_spec.fill_between(wl, mean - std, mean + std, color=meta['color'], alpha=0.1)

        # Plot replicate spectrum for DiseaseLeaf1 right after DiseaseLeaf1 to group them together
        if meta['key'] == 'DiseaseLeaf1':
            csv_1b = os.path.join(DATA_DIR, "cvASD_DiseaseLeaf1b.csv")
            if os.path.exists(csv_1b):
                df_1b = pd.read_csv(csv_1b)
                wl_1b = df_1b['wavelength'].values
                mean_1b = df_1b['DHR_mean'].values.copy()
                std_1b = df_1b['DHR_stdev'].values.copy()
                for t_min, t_max in TELLURIC_RANGES:
                    mask = (wl_1b >= t_min) & (wl_1b <= t_max)
                    mean_1b[mask] = np.nan
                    std_1b[mask] = np.nan
                ax_spec.plot(wl_1b, mean_1b, label='Norway Maple (Tar Spot - Replicate 1b)', 
                             color='#e6550d', linestyle=':', linewidth=2.0)
                ax_spec.fill_between(wl_1b, mean_1b - std_1b, mean_1b + std_1b, color='#e6550d', alpha=0.1)

    # Shaded atmospheric telluric water vapor bands
    for t_min, t_max in TELLURIC_RANGES:
        ax_spec.axvspan(t_min, t_max, color='gray', alpha=0.2)#label='Atmospheric Water Absorption (Omitted)' if t_min == TELLURIC_RANGES[0][0] else "")

    # Physiological region annotations
    #ax_spec.axvline(680, color='darkgreen', linestyle=':', alpha=0.6)
    #ax_spec.text(680, 0.58, 'Chlorophyll Absorption\n(680 nm)', fontsize=ANNOTATION_FONT_SIZE, 
    #             color='darkgreen', ha='center',
    #             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='green'))
#
    #ax_spec.axvline(720, color='darkred', linestyle=':', alpha=0.6)
    #ax_spec.text(780, 0.49, 'Red Edge Transition\n(700-750 nm)', fontsize=ANNOTATION_FONT_SIZE, 
    #             color='darkred', ha='left',
    #             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='darkred'))
#
    #ax_spec.text(950, 0.32, 'NIR Cellular Scattering Plateau\n(Severe collapse under necrosis)', 
    #             fontsize=ANNOTATION_FONT_SIZE, color='#333333', ha='center',
    #             bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff2cc', alpha=0.9, edgecolor='#d6b656'))

    # Labels and formatting
    ax_spec.set_xlabel('Wavelength (nm)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax_spec.set_ylabel('Directional Hemispherical Reflectance (DHR)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax_spec.set_title('ASD FieldSpec Reflectance Measurements', fontsize=TITLE_FONT_SIZE, pad=10)
    ax_spec.set_xlim(WAVELENGTH_MIN, WAVELENGTH_MAX)
    ax_spec.set_ylim(0, REFLECTANCE_MAX)
    ax_spec.tick_params(axis='both', labelsize=TICK_FONT_SIZE)
    ax_spec.grid(True, linestyle='--', alpha=0.4)
    ax_spec.legend(loc='upper right', framealpha=0.95, fontsize=LEGEND_FONT_SIZE)

    # Save figure
    print(f"Saving high-resolution visual to: {out_path}")
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"[Done] Slide 2 visual regenerated successfully: {out_path}\n")
    return out_path


if __name__ == '__main__':
    generate_foliar_ground_truth()
