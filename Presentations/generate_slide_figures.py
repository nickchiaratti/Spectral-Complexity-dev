"""
generate_slide_figures.py
Generates clean, publication-quality 300 DPI figures for Slide 5 and Slide 6:
  - slide5_cholesky_formulation.png: Geometric Gram-space Cholesky projection and Schur downdate
  - slide6_elbow_vs_dropratio_concept.png: Mathematical comparison between heuristic elbow failure and rho_k drop ratio
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = r"C:\Users\nnn_s\Resilio Sync\RIT\IMGS 890 Research\Spectral-Complexity-dev\Presentations\output_hyspex_residuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# FIGURE 1: Slide 5 - Gram-Space Cholesky Geometry & Downdate
# -------------------------------------------------------------------------
def generate_slide5_figure():
    fig = plt.figure(figsize=(7.5, 5.2), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_facecolor('#fcfdfd')

    # Draw 2D/3D subspace projection geometry
    # Origin
    o = np.array([0.15, 0.20])
    v0 = o
    # Subspace line / plane
    e1 = np.array([0.55, 0.25])
    v1 = v0 + e1

    # Endmember 1 vector
    ax.annotate('', xy=v1, xytext=v0,
                arrowprops=dict(arrowstyle="->", color="#1f497d", lw=2.5))
    ax.text(v1[0]+0.02, v1[1]-0.02, r"$v_1$ (Pivot 1)", fontsize=11, fontweight='bold', color="#1f497d")

    # Pivot 2 (selected for max distance h1)
    v2_proj = v0 + 0.5 * e1
    v2_orth = np.array([-0.25 * e1[1], 0.25 * e1[0]]) * 3.5
    v2 = v2_proj + v2_orth

    # Line of subspace S1
    sub_line_x = [v0[0] - 0.05*e1[0], v1[0] + 0.35*e1[0]]
    sub_line_y = [v0[1] - 0.05*e1[1], v1[1] + 0.35*e1[1]]
    ax.plot(sub_line_x, sub_line_y, '--', color='#7c878e', lw=1.5, label=r"Subspace $\mathcal{S}_1 = \mathrm{span}(v_1 - v_0)$")

    # Vector to v2
    ax.annotate('', xy=v2, xytext=v0,
                arrowprops=dict(arrowstyle="->", color="#009cbd", lw=2.5))
    ax.text(v2[0]-0.04, v2[1]+0.03, r"$v_2$ (Pivot 2)", fontsize=11, fontweight='bold', color="#009cbd")

    # Orthogonal projection of v2 onto S1
    ax.plot([v2[0], v2_proj[0]], [v2[1], v2_proj[1]], 'r--', lw=2.0)
    ax.plot(v2_proj[0], v2_proj[1], 'ro', markersize=6)
    ax.text((v2[0]+v2_proj[0])/2 + 0.02, (v2[1]+v2_proj[1])/2, r"$h_1 = \|P^\perp_0 (v_2 - v_0)\|$",
            fontsize=11, fontweight='bold', color="#c0504d")

    # Unselected sample y_i
    yi_proj = v0 + 0.8 * e1
    yi_orth = v2_orth * 0.65
    yi = yi_proj + yi_orth
    ax.plot(yi[0], yi[1], 'o', color='#2ca02c', markersize=7)
    ax.text(yi[0]+0.02, yi[1]+0.02, r"$y_i$ (Unselected Pixel)", fontsize=10, fontweight='bold', color="#2ca02c")
    ax.plot([yi[0], yi_proj[0]], [yi[1], yi_proj[1]], ':', color='#2ca02c', lw=1.8)
    ax.text(yi[0]+0.02, (yi[1]+yi_proj[1])/2 - 0.02, r"$\sqrt{d_i^{(k+1)}} = \|P^\perp_{\mathcal{S}_k} y_i\|$",
            fontsize=10, color="#2ca02c")

    # Right angle marker at v2_proj
    size = 0.025
    u_par = e1 / np.linalg.norm(e1)
    u_perp = v2_orth / np.linalg.norm(v2_orth)
    corner = [v2_proj, v2_proj + size * u_perp, v2_proj + size * u_perp - size * u_par, v2_proj - size * u_par]
    ax.plot([corner[1][0], corner[2][0], corner[3][0]], [corner[1][1], corner[2][1], corner[3][1]], 'k-', lw=1.0)

    # Simplex shading (triangle v0-v1-v2)
    triangle = patches.Polygon([v0, v1, v2], closed=True, facecolor='#009cbd', alpha=0.15, edgecolor='#009cbd', lw=1.5)
    ax.add_patch(triangle)
    ax.text(0.35, 0.32, r"$\mathrm{Vol}(\mathcal{S}) = \frac{1}{k!} \prod_{j=1}^k h_j$", fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#009cbd', alpha=0.9))

    # Schur Downdate Formula Callout Box
    text_box = (
        r"$\mathbf{Schur\ Complement\ Downdate:}$" + "\n"
        r"$d_i^{(k+1)} = d_i^{(k)} - L_{i,k}^2 \equiv \|P^\perp_{\mathcal{S}_k} y_i\|^2$" + "\n"
        r"$\mathrm{RSS}_k = \sum_{i \notin \mathcal{S}_k} d_i^{(k+1)} = \|P^\perp_{\mathcal{S}_k} Y\|_F^2$"
    )
    ax.text(0.12, 0.65, text_box, fontsize=10.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f7fa', edgecolor='#1f497d', lw=1.5))

    ax.set_xlim(0.05, 0.85)
    ax.set_ylim(0.08, 0.78)
    ax.axis('off')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "slide5_cholesky_formulation.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

# -------------------------------------------------------------------------
# FIGURE 2: Slide 6 - Heuristic Elbow Failure vs. Drop Ratio Concept
# -------------------------------------------------------------------------
def generate_slide6_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4), dpi=300)

    k_vals = np.arange(1, 9)

    # Left plot: Heuristic Elbow Failure
    # High contrast scene signal vs low contrast vs pure noise
    h_high_contrast = np.array([0.85, 0.28, 0.09, 0.025, 0.016, 0.014, 0.013, 0.012])
    h_low_contrast = np.array([0.045, 0.018, 0.012, 0.011, 0.010, 0.009, 0.009, 0.008])
    noise_floor_level = 0.012

    ax1.plot(k_vals, h_high_contrast, 'o-', color='#1f77b4', lw=2.2, label='High Contrast Scene')
    ax1.plot(k_vals, h_low_contrast, 's-', color='#ff7f0e', lw=2.2, label='Low Contrast Scene')
    ax1.axhline(noise_floor_level, color='#c0504d', linestyle='--', lw=1.8,
                label=r'Extreme-Value Noise: $\sigma_n \sqrt{2 \ln N}$')

    # Thresholds 1% of h1
    ax1.axhline(0.01 * h_high_contrast[0], color='#1f77b4', linestyle=':', lw=1.5,
                label=r'$1\% \cdot h_1$ (High): 0.0085 (Under-segments)')
    ax1.axhline(0.01 * h_low_contrast[0], color='#ff7f0e', linestyle=':', lw=1.5,
                label=r'$1\% \cdot h_1$ (Low): 0.00045 (Over-segments into noise)')

    ax1.set_yscale('log')
    ax1.set_xlabel('Endmember Iteration $k$', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Orthogonal Distance $h_k$ (Log Scale)', fontsize=11, fontweight='bold')
    ax1.set_title('Heuristic Elbow / Ratio Test Failure\n(Macro-Contrast & Extreme Value Bias)', fontsize=11, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)
    ax1.legend(fontsize=8, loc='upper right')

    # Right plot: Residual Variance Drop Ratio rho_k
    rho_signal = np.array([8.5, 4.2, 3.8, 1.05, 1.03, 1.02, 1.01])
    k_rho = np.arange(1, 8)
    colors = ['#2ca02c' if r >= 2.0 else '#7f7f7f' for r in rho_signal]

    bars = ax2.bar(k_rho, rho_signal, color=colors, width=0.55, edgecolor='black', lw=1.0)
    ax2.axhline(1.0, color='red', linestyle='--', lw=2.0, label=r'Isotropic Noise Baseline ($\rho = 1.0\times$)')
    ax2.axhline(1.5, color='#d62728', linestyle=':', lw=1.5, label=r'Threshold Boundary ($\tau = 1.5\times$)')

    # Shaded zones
    ax2.axvspan(0.5, 3.5, color='#2ca02c', alpha=0.08)
    ax2.axvspan(3.5, 7.5, color='#7f7f7f', alpha=0.08)
    ax2.text(2.0, 7.2, "Physical Materials\n($k^* = 3$ Axes)", fontsize=10, fontweight='bold', color='#1b7837', ha='center')
    ax2.text(5.5, 5.0, "Sensor Noise Floor\n($\\rho_k \\approx 1.02\\times$)", fontsize=10, fontweight='bold', color='#4d4d4d', ha='center')

    for bar, val in zip(bars, rho_signal):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.2, f"{val:.1f}x", ha='center', fontsize=9.5, fontweight='bold')

    ax2.set_xlabel('Step $k$', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Variance Drop Ratio $\rho_k = \hat{\sigma}_{k-1} / \hat{\sigma}_k$', fontsize=11, fontweight='bold')
    ax2.set_title('Degree-of-Freedom Variance Drop Ratio\n(Scale-Free Noise Floor Identification)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 9.5)
    ax2.grid(True, axis='y', linestyle='--', alpha=0.4)
    ax2.legend(fontsize=8.5, loc='upper right')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "slide6_elbow_vs_dropratio_concept.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == '__main__':
    generate_slide5_figure()
    generate_slide6_figure()
