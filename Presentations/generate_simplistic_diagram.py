"""
generate_simplistic_diagram.py
Generates a simplistic, clean diagram illustrating:
  1. Initial distance h_0 between v_0 and v_1
  2. Orthogonal projection and perpendicular height h_1 to endmember v_2
  3. Orthogonal projection and perpendicular height h_2 to endmember v_3
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = r"C:\Users\nnn_s\Resilio Sync\RIT\IMGS 890 Research\Spectral-Complexity-dev\Presentations\output_hyspex_residuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_simplistic_diagram():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    # -------------------------------------------------------------
    # Panel 1: Step 1 & Step 2 (1D Base -> 2D Triangle Simplex)
    # -------------------------------------------------------------
    ax1.set_facecolor('#fafbfc')
    
    # Points
    v0 = np.array([0.15, 0.25])
    v1 = np.array([0.80, 0.25])
    v2_proj = np.array([0.45, 0.25])
    v2 = np.array([0.45, 0.80])
    
    # Base line v0-v1
    ax1.plot([v0[0], v1[0]], [v0[1], v1[1]], 'o-', color='#1f497d', lw=2.5, markersize=8)
    ax1.text(v0[0]-0.04, v0[1]-0.07, "v0\n(Initial)", fontsize=11, fontweight='bold', color='#1f497d', ha='center')
    ax1.text(v1[0]+0.04, v1[1]-0.07, "v1\n(Pivot 1)", fontsize=11, fontweight='bold', color='#1f497d', ha='center')
    ax1.text((v0[0]+v1[0])/2, v0[1]-0.08, "Base Length: h0 = ||v1 - v0||", fontsize=10, color='#1f497d', ha='center')

    # Unselected candidate pixels
    np.random.seed(42)
    cand_x = [0.30, 0.38, 0.60, 0.68, 0.72]
    cand_y = [0.45, 0.55, 0.50, 0.40, 0.60]
    for cx, cy in zip(cand_x, cand_y):
        ax1.plot(cx, cy, 'o', color='#94a3b8', markersize=5)
        ax1.plot([cx, cx], [0.25, cy], ':', color='#cbd5e1', lw=1.0)

    # Perpendicular height h1 to v2
    ax1.plot([v2[0], v2_proj[0]], [v2[1], v2_proj[1]], '--', color='#c0504d', lw=2.2)
    ax1.plot(v2[0], v2[1], 'o', color='#c0504d', markersize=9)
    ax1.text(v2[0], v2[1]+0.05, "v2 (Max Height Pivot)", fontsize=11, fontweight='bold', color='#c0504d', ha='center')
    ax1.text(v2[0]+0.04, (v2[1]+v2_proj[1])/2, "Height h1\n(Max Orthogonal Distance)", fontsize=10, fontweight='bold', color='#c0504d', va='center')

    # Right angle symbol
    sq_size = 0.04
    ax1.plot([v2_proj[0], v2_proj[0]+sq_size, v2_proj[0]+sq_size, v2_proj[0]],
             [v2_proj[1]+sq_size, v2_proj[1]+sq_size, v2_proj[1], v2_proj[1]], 'k-', lw=1.0)

    # Triangle face
    tri = patches.Polygon([v0, v1, v2], closed=True, facecolor='#009cbd', alpha=0.12, edgecolor='#009cbd', lw=1.5, linestyle='--')
    ax1.add_patch(tri)
    ax1.text(0.62, 0.68, "Simplex Area (2D):\nVol = (1/2!) * h0 * h1", fontsize=10.5, fontweight='bold',
             color='#007a99', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#009cbd', lw=1.0))

    ax1.set_xlim(0.05, 0.95)
    ax1.set_ylim(0.08, 0.95)
    ax1.set_title("Step 1 & 2: Base Distance h0 and Orthogonal Height h1", fontsize=11, fontweight='bold', color='#1f497d', pad=10)
    ax1.axis('off')

    # -------------------------------------------------------------
    # Panel 2: Step 3 (2D Plane -> 3D Simplex Volume with h2)
    # -------------------------------------------------------------
    ax2.set_facecolor('#fafbfc')

    # 3D isometric projection coordinates
    p0 = np.array([0.20, 0.25])
    p1 = np.array([0.75, 0.20])
    p2 = np.array([0.45, 0.50])
    
    # Plane base (v0-v1-v2)
    plane = patches.Polygon([p0, p1, p2], closed=True, facecolor='#e2e8f0', edgecolor='#1f497d', lw=2.0)
    ax2.add_patch(plane)
    ax2.plot([p0[0], p1[0], p2[0], p0[0]], [p0[1], p1[1], p2[1], p0[1]], 'o', color='#1f497d', markersize=7)
    ax2.text(p0[0]-0.04, p0[1]-0.04, "v0", fontsize=11, fontweight='bold', color='#1f497d')
    ax2.text(p1[0]+0.03, p1[1]-0.02, "v1", fontsize=11, fontweight='bold', color='#1f497d')
    ax2.text(p2[0]-0.05, p2[1]+0.03, "v2", fontsize=11, fontweight='bold', color='#1f497d')
    ax2.text(0.48, 0.30, "Subspace S2 = span(v1-v0, v2-v0)", fontsize=9.5, color='#64748b', ha='center')

    # Pivot 3 projected orthogonal to plane S2
    p3_proj = np.array([0.48, 0.32])
    p3 = np.array([0.48, 0.82])

    ax2.plot([p3[0], p3_proj[0]], [p3[1], p3_proj[1]], '--', color='#2ca02c', lw=2.5)
    ax2.plot(p3_proj[0], p3_proj[1], 'ko', markersize=4)
    ax2.plot(p3[0], p3[1], 'o', color='#2ca02c', markersize=9)
    ax2.text(p3[0], p3[1]+0.05, "v3 (Pivot 3)", fontsize=11, fontweight='bold', color='#2ca02c', ha='center')
    ax2.text(p3[0]+0.04, (p3[1]+p3_proj[1])/2, "Height h2\n(Orthogonal to Plane S2)", fontsize=10, fontweight='bold', color='#2ca02c', va='center')

    # Tetrahedron edges to v3
    ax2.plot([p0[0], p3[0]], [p0[1], p3[1]], ':', color='#2ca02c', lw=1.2)
    ax2.plot([p1[0], p3[0]], [p1[1], p3[1]], ':', color='#2ca02c', lw=1.2)
    ax2.plot([p2[0], p3[0]], [p2[1], p3[1]], ':', color='#2ca02c', lw=1.2)

    ax2.text(0.62, 0.68, "Simplex Volume (3D):\nVol = (1/3!) * h0 * h1 * h2", fontsize=10.5, fontweight='bold',
             color='#166534', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#2ca02c', lw=1.0))

    ax2.set_xlim(0.05, 0.95)
    ax2.set_ylim(0.08, 0.95)
    ax2.set_title("Step 3: Subspace Projection and Orthogonal Height h2", fontsize=11, fontweight='bold', color='#1f497d', pad=10)
    ax2.axis('off')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "diagram_endmember_heights.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == '__main__':
    generate_simplistic_diagram()
