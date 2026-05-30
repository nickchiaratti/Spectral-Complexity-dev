import numpy as np
import matplotlib.pyplot as plt
import math
import scienceplots
plt.style.use(['science','ieee'])

fig_width=6
fig_height=fig_width*0.5

# ==========================================
# 1. SYNTHETIC COEFFICIENT CONFIGURATION
# ==========================================
COEFFS = {
    'bias': 0.15,            # a0: Overall mean of the signal
    'trend': -0.1,           # c1: Slight upward ecological trend
    'ann_cos': -0.3,          # a1: Annual Cosine Amplitude
    'ann_sin': 0.35,         # b1: Annual Sine Amplitude
    'semi_cos': 0.15,        # a2: Semi-annual Cosine Amplitude
    'semi_sin': 0.1,         # b2: Semi-annual Sine Amplitude
    'tri_cos': 0.1,         # a3: Tri-annual Cosine Amplitude
    'tri_sin': 0.15,          # b3: Tri-annual Sine Amplitude
    'quad_cos':.35,          # a4: Quad-annual Cosine Amplitude
    'quad_sin':.2           # b4: Quad-annual Sine Amplitude
}

# Industry standard threshold for CCDC structural breaks
SYNTHETIC_RMSE = 0.08
RMSE_MULTIPLIER = 3.0

# ==========================================
# 2. GENERATE TEMPORAL DOMAIN
# ==========================================
# Strict enforcement of a single annual period (Fraction of Year 0.0 to 1.0)
t = np.linspace(0.0, 1.0, 500)
w = 2.0 * math.pi

# ==========================================
# 3. CALCULATE INDEPENDENT COMPONENTS
# ==========================================
comp_bias = np.full_like(t, COEFFS['bias'])
comp_trend = COEFFS['trend'] * t
comp_ann_cos = COEFFS['ann_cos'] * np.cos(w * t)
comp_ann_sin = COEFFS['ann_sin'] * np.sin(w * t)
comp_semi_cos = COEFFS['semi_cos'] * np.cos(2 * w * t)
comp_semi_sin = COEFFS['semi_sin'] * np.sin(2 * w * t)
comp_tri_cos = COEFFS['tri_cos'] * np.cos(3 * w * t)
comp_tri_sin = COEFFS['tri_sin'] * np.sin(3 * w * t)
comp_quad_cos = COEFFS['quad_cos'] * np.cos(4 * w * t)
comp_quad_sin = COEFFS['quad_sin'] * np.sin(4 * w * t)

combined_model = comp_bias + comp_trend + comp_ann_cos + comp_ann_sin + comp_semi_cos + comp_semi_sin + comp_tri_cos + comp_tri_sin + comp_quad_cos + comp_quad_sin

# ==========================================
# 4. GENERATE SYNTHETIC OBSERVATIONS
# ==========================================
def eval_harmonic_model(t_array):
    return (COEFFS['bias'] + COEFFS['trend'] * t_array +
            COEFFS['ann_cos'] * np.cos(w * t_array) + COEFFS['ann_sin'] * np.sin(w * t_array) +
            COEFFS['semi_cos'] * np.cos(2 * w * t_array) + COEFFS['semi_sin'] * np.sin(2 * w * t_array) +
            COEFFS['tri_cos'] * np.cos(3 * w * t_array) + COEFFS['tri_sin'] * np.sin(3 * w * t_array) +
            COEFFS['quad_cos'] * np.cos(4 * w * t_array) + COEFFS['quad_sin'] * np.sin(4 * w * t_array))

# Set seed for reproducible presentation graphics
np.random.seed(42)

# Simulate ~22 clear-sky Landsat observations across the year
obs_t = np.sort(np.random.uniform(0.02, 0.98, 45))
# Add standard Gaussian variance (simulating atmospheric noise within the RMSE boundary)
obs_y = eval_harmonic_model(obs_t) + np.random.normal(0, SYNTHETIC_RMSE * 0.6, size=len(obs_t))

# Deliberately inject transient outliers (e.g., unmasked clouds)
outlier_indices = [4, 15, 19, 25, 30, 35, 40, 42]
outlier_t = obs_t[outlier_indices]
# Push outliers violently outside the 3*RMSE boundary
outlier_y = eval_harmonic_model(outlier_t) + np.random.choice([-1.2, 1.2], size=len(outlier_indices)) * (SYNTHETIC_RMSE * 4.5)

# Isolate valid inlier observations for plotting
inlier_mask = np.ones(len(obs_t), dtype=bool)
inlier_mask[outlier_indices] = False
inlier_t = obs_t[inlier_mask]
inlier_y = obs_y[inlier_mask]

# ==========================================
# 5. PLOTTING FUNCTIONALITY & SEASONAL BACKGROUNDS
# ==========================================
def apply_seasonal_backgrounds(ax):
    """
    Applies color-coded seasonal backgrounds using fractional year mapping.
    Mar 1 = Day 59  (~0.1616)
    Jun 1 = Day 151 (~0.4137)
    Sep 1 = Day 243 (~0.6658)
    Dec 1 = Day 334 (~0.9151)
    """
    mar1 = 59 / 365.0
    jun1 = 151 / 365.0
    sep1 = 243 / 365.0
    dec1 = 334 / 365.0

    # Winter (Jan 1 - Mar 1) -> light gray
    ax.axvspan(0.0, mar1, color='lightgray', alpha=0.3, zorder=0, lw=0)
    # Spring (Mar 1 - Jun 1) -> light green
    ax.axvspan(mar1, jun1, color='lightgreen', alpha=0.2, zorder=0, lw=0)
    # Summer (Jun 1 - Sep 1) -> light yellow
    ax.axvspan(jun1, sep1, color='lightyellow', alpha=0.3, zorder=0, lw=0)
    # Fall (Sep 1 - Dec 1) -> light orange
    ax.axvspan(sep1, dec1, color='orange', alpha=0.15, zorder=0, lw=0)
    # Winter (Dec 1 - Dec 31) -> light gray
    ax.axvspan(dec1, 1.0, color='lightgray', alpha=0.3, zorder=0, lw=0)

def create_presentation_visuals():
    """Generates clean, publication-quality graphics for doctoral defense."""
    
    month_labels = [r'$\frac{0}{12}$', r'$\frac{1}{12}$', r'$\frac{2}{12}$', r'$\frac{3}{12}$', r'$\frac{4}{12}$', r'$\frac{5}{12}$', r'$\frac{6}{12}$', r'$\frac{7}{12}$', r'$\frac{8}{12}$', r'$\frac{9}{12}$', r'$\frac{10}{12}$', r'$\frac{11}{12}$', r'$\frac{12}{12}$']
    # ---------------------------------------------------------
    # FIGURE 1: Overlaid Harmonic Components
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots()
    fig1.canvas.manager.set_window_title("Harmonic Decomposition")
    
    apply_seasonal_backgrounds(ax1)
    
    plot_configs = [
        (comp_bias, "Bias ($c_0$)", "black", "-"),
        (comp_trend, r"Linear Trend ($c_1 \cdot t$)", "darkorange", "--"),
        (comp_ann_cos, r"Annual Cosine ($a_0 \cdot cos(2\pi \cdot 1 \cdot t)$)", "forestgreen", "-"),
        (comp_ann_sin, r"Annual Sine ($a_1 \cdot sin(2\pi \cdot 1 \cdot t)$)", "limegreen", "-."),
        (comp_semi_cos, r"Semi-Annual Cosine ($b_0 \cdot cos(2\pi \cdot 2 \cdot t)$)", "purple", "-"),
        (comp_semi_sin, r"Semi-Annual Sine ($b_1 \cdot sin(2\pi \cdot 2 \cdot t)$)", "violet", "-."),
        (comp_tri_cos, r"Tri-Annual Cosine ($c_0 \cdot cos(2\pi \cdot 3 \cdot t)$)", "brown", "-"),
        (comp_tri_sin, r"Tri-Annual Sine ($c_1 \cdot sin(2\pi \cdot 3 \cdot t)$)", "teal", "-."),
        (comp_quad_cos, r"Quad-Annual Cosine ($d_0 \cdot cos(2\pi \cdot 4 \cdot t)$)", "orange", "-"),
        (comp_quad_sin, r"Quad-Annual Sine ($d_1 \cdot sin(2\pi \cdot 4 \cdot t)$)", "cyan", "-.")
    ]
    
    for data, label, color, style in plot_configs:
        ax1.plot(t, data, color=color, linestyle=style, linewidth=3, label=label)
        
    ax1.set_title("Continuous Change Detection (CCD): 6-Parameter Harmonic Components")
    ax1.set_xlabel("Time (Fraction of Year)")
    ax1.set_ylabel("Amplitude")
    
    ax1.axhline(0, color='gray', linewidth=1, linestyle='-')
    ax1.set_xlim(0, 1)
    ax1.set_xticks(np.linspace(0, 1, 13))
    ax1.set_xticklabels(month_labels)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax1.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    
    # ---------------------------------------------------------
    # FIGURE 2: The Reconstructed CCDC Baseline
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(fig_width,fig_height))
    fig2.canvas.manager.set_window_title("Reconstructed Baseline")
    
    apply_seasonal_backgrounds(ax2)
    
    # 1. Plot the Mathematical Boundary (Confidence Interval)
    upper_bound = combined_model + (RMSE_MULTIPLIER * SYNTHETIC_RMSE)
    lower_bound = combined_model - (RMSE_MULTIPLIER * SYNTHETIC_RMSE)
    ax2.fill_between(t, lower_bound, upper_bound, color='blue', alpha=0.15, label=r'$\pm 3$ RMSE Statistical Boundary', zorder=2)
    
    # 2. Plot the Prediction Curve
    ax2.plot(t, combined_model, color='blue', linewidth=3, label='CCD Prediction Model', zorder=3)
    
    # Filter observations after 10/12
    threshold_t = 10.0 / 12.0
    valid_inlier = inlier_t <= threshold_t
    valid_outlier = outlier_t <= threshold_t
    
    # 3. Scatter the Synthetic Observations (up to 10/12)
    ax2.scatter(inlier_t[valid_inlier], inlier_y[valid_inlier], c='black', marker='o', s=20, label='Valid Observations', zorder=4)
    ax2.scatter(outlier_t[valid_outlier], outlier_y[valid_outlier], facecolors='none', edgecolors='black', marker='o', s=20, alpha=0.7, label='Quality Masked Observations', zorder=4)

    # 4. Generate and Plot Anomaly Points
    np.random.seed(42)  # reproducible anomaly generation
    anomaly_t = np.sort(np.random.uniform(threshold_t + 0.01, 0.99, 8))
    anomaly_y = np.random.normal(1.0, 0.15, size=8)
    ax2.scatter(anomaly_t, anomaly_y, c='red', marker='x', s=80, linewidths=2.5, label='Anomaly', zorder=5)
    
    ax2.set_title("Harmonic Baseline and Observations")
    ax2.set_xlabel("Time (Fraction of Year)")
    ax2.set_ylabel("Predicted Value")
    ax2.set_xlim(0, 1)
    
    ax2.set_xticks(np.linspace(0, 1, 13))
    ax2.set_xticklabels(month_labels)
    
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='best', framealpha=0.9)

    #plt.show()

if __name__ == "__main__":
    create_presentation_visuals()
    for i in plt.get_fignums():
        plt.figure(i)
        plt.savefig(f'Spectral-Complexity-dev/anomaly_detection/ContinuousChangeDetector/harmonic_visualization_{i}.png',dpi=600)