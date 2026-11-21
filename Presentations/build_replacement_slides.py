"""
build_replacement_slides.py
Generates the new replacement slides 5, 6, and 7:
  1. Updates 'C:\\Users\\nnn_s\\Resilio Sync\\RIT\\IMGS 890 Research\\Spectral Complexity Overview.pptx'
  2. Generates standalone 'C:\\Users\\nnn_s\\Resilio Sync\\RIT\\IMGS 890 Research\\Spectral-Complexity-dev\\Presentations\\Replacement_Slides_5_to_7.pptx'
"""

import os
import shutil
import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

# Paths
SOURCE_PPTX = r"C:\Users\nnn_s\Resilio Sync\RIT\IMGS 890 Research\Spectral Complexity Overview.pptx"
BACKUP_PPTX = r"C:\Users\nnn_s\Resilio Sync\RIT\IMGS 890 Research\Spectral Complexity Overview_backup.pptx"
STANDALONE_PPTX = r"C:\Users\nnn_s\Resilio Sync\RIT\IMGS 890 Research\Spectral-Complexity-dev\Presentations\Replacement_Slides_5_to_7.pptx"
FIGURE_DIR = r"C:\Users\nnn_s\Resilio Sync\RIT\IMGS 890 Research\Spectral-Complexity-dev\Presentations\output_hyspex_residuals"

FIG_SLIDE5 = os.path.join(FIGURE_DIR, "slide5_cholesky_formulation.png")
FIG_SLIDE6 = os.path.join(FIGURE_DIR, "slide6_elbow_vs_dropratio_concept.png")
FIG_SLIDE7_EXEC = os.path.join(FIGURE_DIR, "hyspex_complexity_residuals_executive_overview.png")

# Palette
CLR_PRIMARY_NAVY = RGBColor(31, 73, 125)    # #1F497D
CLR_ACCENT_TEAL  = RGBColor(0, 156, 189)    # #009CBD
CLR_DARK_TEXT    = RGBColor(40, 44, 52)     # #282C34
CLR_MUTED_TEXT   = RGBColor(100, 110, 125)  # #646E7D
CLR_CARD_BG      = RGBColor(248, 250, 252)  # #F8FAFC
CLR_CARD_BORDER  = RGBColor(203, 213, 225)  # #CBD5E1
CLR_BANNER_BG    = RGBColor(238, 242, 246)  # #EEF2F6
CLR_WHITE        = RGBColor(255, 255, 255)
CLR_ALERT_RED    = RGBColor(192, 80, 77)    # #C0504D
CLR_SUCCESS_GRN  = RGBColor(44, 160, 44)    # #2CA02C

FONT_FAMILY = "Arial"

def add_header(slide, title_text, banner_text):
    """Adds consistent header and key takeaway banner."""
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.60), Inches(0.20), Inches(12.13), Inches(0.60))
    tf = title_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = FONT_FAMILY
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY_NAVY

    # Banner
    banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.60), Inches(0.85), Inches(12.13), Inches(0.44))
    banner.fill.solid()
    banner.fill.fore_color.rgb = CLR_BANNER_BG
    banner.line.color.rgb = CLR_ACCENT_TEAL
    banner.line.width = Pt(1.5)
    btf = banner.text_frame
    btf.word_wrap = True
    btf.margin_left = Inches(0.15)
    btf.margin_right = Inches(0.15)
    btf.margin_top = Inches(0.04)
    btf.margin_bottom = Inches(0.04)
    bp = btf.paragraphs[0]
    bp.text = banner_text
    bp.font.name = FONT_FAMILY
    bp.font.size = Pt(12)
    bp.font.bold = True
    bp.font.color.rgb = CLR_PRIMARY_NAVY

def create_card(slide, left, top, width, height, title=None):
    """Creates a stylized card container with subtle border."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = CLR_CARD_BG
    card.line.color.rgb = CLR_CARD_BORDER
    card.line.width = Pt(1.0)
    
    # Internal textbox for content
    tb = slide.shapes.add_textbox(Inches(left + 0.15), Inches(top + 0.12), Inches(width - 0.30), Inches(height - 0.24))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    if title:
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = FONT_FAMILY
        p0.font.size = Pt(13)
        p0.font.bold = True
        p0.font.color.rgb = CLR_PRIMARY_NAVY
        p0.space_after = Pt(6)

    return tf

def populate_slide_5(slide):
    """Slide 5: Gram-Space Cholesky MaxD with Exact Residual Downdating."""
    add_header(
        slide,
        "Gram-Space Cholesky MaxD & Residual Energy Tracking",
        "Gram-space formulation computes simplex volume and tracks Frobenius residual energy ||P_k^perp Y||_F^2 in O(N*k) without matrix inversions."
    )

    # Left Card: Mathematical Formulation & Schur Complement Downdate
    tf_left = create_card(slide, 0.60, 1.40, 6.00, 5.50, "Mathematical Formulation & Schur Downdate")

    # Item 1: Localized Gram Matrix
    p = tf_left.add_paragraph()
    p.text = "1. Localized Gram Matrix Representation"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_left.add_paragraph()
    p.text = "• Local pixel window Y = [y_1, ..., y_N] in R^{B x N} (e.g. 3x3 window, N=9)."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "• Form symmetric Gram matrix K = Y^T Y in R^{N x N}, where K_ij = y_i^T y_j."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "• Distance to initial vertex v_0: d_i^(1) = K_ii - 2 K_i,i0 + K_i0,i0."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT
    p.space_after = Pt(4)

    # Item 2: Greedy Pivot Selection
    p = tf_left.add_paragraph()
    p.text = "2. Greedy Pivot Selection & Triangular Factor L"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_left.add_paragraph()
    p.text = "• Select pivot index: i_k = argmax_{i not in S} d_i^(k). Max height: h_k = sqrt(d_{i_k}^(k))."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "• Update column k of Cholesky factor L in R^{N x k}:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "     L_{i,k} = (1 / h_k) * [ K_{i,i_k} - sum_{j=1}^{k-1} L_{i,j} L_{i_k,j} ]"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY_NAVY
    p.space_after = Pt(4)

    # Item 3: Exact Schur Complement Downdate
    p = tf_left.add_paragraph()
    p.text = "3. Exact Schur Complement Downdate & Residual Sum of Squares"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_left.add_paragraph()
    p.text = "• Residual distance update: d_i^(k+1) = d_i^(k) - (L_{i,k})^2 == ||P_{S_k}^perp y_i||^2."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "• Exact Total RSS: RSS_k = sum_{i not in S_k} d_i^(k+1) == ||P_{S_k}^perp Y||_F^2."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY_NAVY

    p = tf_left.add_paragraph()
    p.text = "• Simplex volume accumulates geometrically: V_k = (1 / k!) * prod_{j=1}^k h_j."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    # Right Column: Diagram and Computational Benefits
    slide.shapes.add_picture(FIG_SLIDE5, Inches(6.80), Inches(1.40), Inches(5.93), Inches(3.20))

    tf_right = create_card(slide, 6.80, 4.70, 5.93, 2.20, "Computational & Algorithmic Advantages")

    bullets = [
        ("O(N*k) Scaling: ", "Avoids computing BxB projection matrices P^perp = I - U(U^T U)^-1 U^T (O(B^2 N + B^3))."),
        ("1.5x - 2.0x Execution Speedup: ", "Evaluated directly in spatial Gram space; highly optimized for sliding window filters."),
        ("Full float64 Precision: ", "Eliminates loss of significance in subtle spectral residuals while preserving input data types."),
        ("Simultaneous Output Tuple: ", "Returns endmembers, volumes_arr, heights_arr, indices_arr, and rss_arr in a single forward pass.")
    ]
    for bold_lead, body in bullets:
        p = tf_right.add_paragraph()
        run1 = p.add_run()
        run1.text = "• " + bold_lead
        run1.font.name = FONT_FAMILY
        run1.font.size = Pt(10)
        run1.font.bold = True
        run1.font.color.rgb = CLR_PRIMARY_NAVY
        run2 = p.add_run()
        run2.text = body
        run2.font.name = FONT_FAMILY
        run2.font.size = Pt(10)
        run2.font.color.rgb = CLR_DARK_TEXT

def populate_slide_6(slide):
    """Slide 6: Failure of Heuristic Elbow Tests vs. Degree-of-Freedom Noise Floor."""
    add_header(
        slide,
        "Failure of Heuristic Elbow Tests vs. Degree-of-Freedom Noise Floor",
        "Subspace noise persistence and extreme-value selection invalidate ratio thresholds; degree-of-freedom pooling isolates intrinsic material rank."
    )

    # Left Card: Why Heuristic Elbow Tests Fail
    tf_left = create_card(slide, 0.60, 1.40, 5.80, 5.50, "Why Heuristic Elbow & Ratio Tests Fail")

    failures = [
        ("1. Scale Invariance Breakdown (Macro-Contrast Coupling):",
         "The initial height h_1 = ||v_1 - v_0|| depends entirely on macro-scene contrast. In high-contrast scenes (h_1 = 0.8), a 1% threshold (0.008) truncates physical endmembers before discovery. In low-contrast scenes (h_1 = 0.04), a 1% threshold (0.0004) forces the algorithm deep into sub-noise artifacts."),
        
        ("2. Subspace Noise Power Persistence:",
         "Under orthogonal projection into a (B-k) dimensional null space, white noise variance is not reduced to zero. The expected residual Frobenius energy satisfies E[||P_k^perp E||_F^2] = (B-k) N sigma_n^2. Noise energy remains persistently elevated across iterations."),
        
        ("3. Extreme-Value Selection Bias:",
         "MaxD greedily selects h_k = max_i sqrt(d_i^(k)). Over N spatial samples, pure sensor noise does not decay toward zero, but plateaus at the Gumbel extreme-value ceiling approx sigma_n * sqrt(2 * ln(N)). A physical elbow does not exist in high-dimensional noise.")
    ]

    for title, text in failures:
        p = tf_left.add_paragraph()
        p.text = title
        p.font.name = FONT_FAMILY
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = CLR_ALERT_RED

        p = tf_left.add_paragraph()
        p.text = text
        p.font.name = FONT_FAMILY
        p.font.size = Pt(10)
        p.font.color.rgb = CLR_DARK_TEXT
        p.space_after = Pt(4)

    # Right Area: Concept Figure and Pooled Noise / Drop Ratio Formulation
    slide.shapes.add_picture(FIG_SLIDE6, Inches(6.60), Inches(1.40), Inches(6.13), Inches(2.70))

    tf_right = create_card(slide, 6.60, 4.20, 6.13, 2.70, "Degree-of-Freedom Pooled Noise & Drop Ratio (rho_k)")

    p = tf_right.add_paragraph()
    p.text = "• Pooled Noise Standard Deviation (sigma_hat_k):"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY_NAVY

    p = tf_right.add_paragraph()
    p.text = "     sigma_hat_k = sqrt( RSS_k / [ (B - k) * (N - k) ] )"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY_NAVY
    p.space_after = Pt(3)

    p = tf_right.add_paragraph()
    p.text = "  Consumes k spectral projection directions and k spatial pivot constraints from B*N elements."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(9.5)
    p.font.color.rgb = CLR_MUTED_TEXT

    p = tf_right.add_paragraph()
    p.text = "• Residual Variance Drop Ratio (rho_k = sigma_hat_{k-1} / sigma_hat_k):"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY_NAVY

    p = tf_right.add_paragraph()
    p.text = "  - Material Signal Regime: Resolving a physical axis yields sharp drop rho_k >= 3.0 (up to 18x)."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(9.5)
    p.font.color.rgb = CLR_SUCCESS_GRN

    p = tf_right.add_paragraph()
    p.text = "  - Noise Floor Regime: Once intrinsic rank k* is reached, projecting noise vectors removes only nominal degrees of freedom 1/(N-k), locking rho_k at approx 1.01 - 1.06x == 1.00x baseline."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(9.5)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_right.add_paragraph()
    p.text = "• Sensor-Agnostic Criterion: Floor threshold tau_noise approx 1.5x identifies rank k* without a priori noise data."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(9.5)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY_NAVY

def populate_slide_7(slide):
    """Slide 7: Empirical Validation: Airborne Flightlines & Foliar Complexity."""
    add_header(
        slide,
        "Empirical Validation: Airborne Flightlines & Foliar Complexity",
        "Flightline calibration panels and simulated foliar disease stages demonstrate exact material rank identification and collapse to 1.0x noise floor."
    )

    # Left: Executive 6-Scenario Overview Visual
    slide.shapes.add_picture(FIG_SLIDE7_EXEC, Inches(0.60), Inches(1.40), Inches(7.10), Inches(3.60))

    # Caption / Legend box below visual
    tf_cap = create_card(slide, 0.60, 5.10, 7.10, 1.80, "Visual Architecture & Metric Tracking")
    cap_points = [
        ("Vertically Linked Panels: ", "Top plots orthogonal height h_k (blue, left log axis) and pooled noise sigma_k (orange, right log axis). Bottom plots step drop ratio rho_k (green signal, gray noise)."),
        ("Noise Baseline Confirmation: ", "Red dashed line marks the theoretical isotropic noise floor rho = 1.0x. Signal regimes exhibit sharp peaks (4.0x to 18.9x) before collapsing identically to 1.01x - 1.04x."),
        ("Multi-Scenario Verification: ", "Demonstrated across real airborne panels (k*=1), spatial boundaries (k*=2), foliar shading (k*=1), and 4-material pathological leaf infection (k*=4).")
    ]
    for b_lead, b_body in cap_points:
        p = tf_cap.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + b_lead
        r1.font.name = FONT_FAMILY
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = CLR_PRIMARY_NAVY
        r2 = p.add_run()
        r2.text = b_body
        r2.font.name = FONT_FAMILY
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = CLR_DARK_TEXT

    # Right: Quantitative Summary Table & Key Findings
    # Table of Results
    table_shape = slide.shapes.add_table(5, 4, Inches(7.85), Inches(1.40), Inches(4.88), Inches(2.20))
    table = table_shape.table
    table.columns[0].width = Inches(1.90)
    table.columns[1].width = Inches(0.70)
    table.columns[2].width = Inches(1.30)
    table.columns[3].width = Inches(0.98)

    headers = ["Scenario", "Rank k*", "Signal rho_k", "Noise rho"]
    for c_idx, h in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = CLR_PRIMARY_NAVY
        p = cell.text_frame.paragraphs[0]
        p.text = h
        p.font.name = FONT_FAMILY
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = CLR_WHITE
        p.alignment = PP_ALIGN.CENTER

    row_data = [
        ("Real Panel (1059, 258)", "k* = 1", "rho_1 = 4.29x", "1.03x (k=2)"),
        ("Real Boundary (1108, 325)", "k* = 2", "4.49x, 2.45x", "1.02x (k=3)"),
        ("Foliar Shading (Single Leaf)", "k* = 1", "rho_1 = 8.16x", "1.04x (k=2)"),
        ("Foliar Pathological (4-Mat)", "k* = 4", "18.9x, 6.4x, 4.2x, 3.8x", "1.02x (k=5)")
    ]
    for r_idx, row in enumerate(row_data, start=1):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = CLR_CARD_BG if r_idx % 2 == 1 else CLR_WHITE
            p = cell.text_frame.paragraphs[0]
            p.text = val
            p.font.name = FONT_FAMILY
            p.font.size = Pt(9.5)
            p.font.color.rgb = CLR_DARK_TEXT
            if c_idx > 0:
                p.alignment = PP_ALIGN.CENTER
                p.font.bold = True if c_idx == 1 else False

    # Analytical Takeaways Card
    tf_takeaways = create_card(slide, 7.85, 3.75, 4.88, 3.15, "Key Analytical Findings")
    findings = [
        ("Isotropic Noise Floor Invariance: ", "In all real and simulated flightline scenes, once physical endmembers are accounted for, rho_k locks onto 1.01x - 1.04x, providing an objective stopping boundary."),
        ("Rejection of Illumination Shading: ", "Shading along single foliar surfaces represents 1D amplitude scaling without new spectral absorption features; rho_k correctly drops to 1.04x after k=1."),
        ("Pathological Disease Identification: ", "Resolves all four distinct biological stages (healthy, chlorotic, necrotic, dehydrated) with significant drops (>=3.8x) before hitting noise at k=5."),
        ("Cross-Sensor Invariance: ", "Consistently extracts k*=4 material axes across Landsat-8 (B=7), Sentinel-2 (B=13), AVIRIS (B=224), and HySpex (B=413), while rejecting replicate scans (rho approx 1.01x).")
    ]
    for lead, text in findings:
        p = tf_takeaways.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + lead
        r1.font.name = FONT_FAMILY
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = CLR_PRIMARY_NAVY
        r2 = p.add_run()
        r2.text = text
        r2.font.name = FONT_FAMILY
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = CLR_DARK_TEXT

def create_standalone_deck():
    """Generates the standalone 3-slide deck."""
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.500)
    blank_layout = prs.slide_layouts[6] # completely blank layout

    # Slide 1 (representing replacement slide 5)
    s5 = prs.slides.add_slide(blank_layout)
    populate_slide_5(s5)

    # Slide 2 (representing replacement slide 6)
    s6 = prs.slides.add_slide(blank_layout)
    populate_slide_6(s6)

    # Slide 3 (representing replacement slide 7)
    s7 = prs.slides.add_slide(blank_layout)
    populate_slide_7(s7)

    prs.save(STANDALONE_PPTX)
    print(f"Saved standalone presentation: {STANDALONE_PPTX}")

def update_main_deck():
    """Replaces slides 5-7 in Spectral Complexity Overview.pptx."""
    # Backup first
    if not os.path.exists(BACKUP_PPTX):
        shutil.copy2(SOURCE_PPTX, BACKUP_PPTX)
        print(f"Created backup: {BACKUP_PPTX}")

    prs = pptx.Presentation(SOURCE_PPTX)
    
    # Slides 5, 6, 7 correspond to indices 4, 5, 6
    slide_funcs = [populate_slide_5, populate_slide_6, populate_slide_7]

    for offset, func in enumerate(slide_funcs):
        idx = 4 + offset
        slide = prs.slides[idx]
        sp_tree = slide.shapes._spTree
        # Remove all existing shapes on the slide except slide number placeholder if present
        for sp in list(slide.shapes):
            if sp.has_text_frame and "Number" in sp.name:
                continue
            sp_tree.remove(sp._element)
        
        # Populate with new content
        func(slide)

    prs.save(SOURCE_PPTX)
    print(f"Updated main presentation: {SOURCE_PPTX}")

if __name__ == "__main__":
    create_standalone_deck()
    update_main_deck()
    print("All slides generated and updated successfully.")
