"""
generate_standalone_deck.py
Generates the clean standalone presentation:
  'C:\\Users\\nnn_s\\Resilio Sync\\RIT\\IMGS 890 Research\\Spectral-Complexity-dev\\Presentations\\Replacement_Slides_5_to_7.pptx'

Key Principles:
  - Straightforward, rigorous description of algorithm mathematics
  - No raw unrendered LaTeX markup; clean, human-readable mathematical formatting
  - Simplistic diagram of endmember selection and iterative projection heights
  - Clean, professional typography and generous whitespace
  - Strictly no em dashes and maintain objective academic tone
"""

import os
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_PPTX = r"C:\Users\nnn_s\Resilio Sync\RIT\IMGS 890 Research\Spectral-Complexity-dev\Presentations\Replacement_Slides_5_to_7.pptx"
DIAGRAM_PATH = r"C:\Users\nnn_s\Resilio Sync\RIT\IMGS 890 Research\Spectral-Complexity-dev\Presentations\output_hyspex_residuals\diagram_endmember_heights.png"

# Color Palette
CLR_PRIMARY_NAVY = RGBColor(31, 73, 125)    # #1F497D
CLR_ACCENT_TEAL  = RGBColor(0, 156, 189)    # #009CBD
CLR_DARK_TEXT    = RGBColor(40, 44, 52)     # #282C34
CLR_MUTED_TEXT   = RGBColor(90, 100, 115)   # #5A6473
CLR_CARD_BG      = RGBColor(248, 250, 252)  # #F8FAFC
CLR_CARD_BORDER  = RGBColor(203, 213, 225)  # #CBD5E1
CLR_BANNER_BG    = RGBColor(238, 242, 246)  # #EEF2F6
CLR_WHITE        = RGBColor(255, 255, 255)
CLR_MATH_NAVY    = RGBColor(20, 50, 95)

FONT_FAMILY = "Arial"

def add_header(slide, title_text, banner_text):
    """Adds a clear title and uncluttered takeaway banner."""
    # Title
    tb = slide.shapes.add_textbox(Inches(0.60), Inches(0.20), Inches(12.13), Inches(0.55))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = FONT_FAMILY
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY_NAVY

    # Banner
    banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.60), Inches(0.80), Inches(12.13), Inches(0.40))
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
    bp.font.size = Pt(11.5)
    bp.font.bold = True
    bp.font.color.rgb = CLR_PRIMARY_NAVY

def create_card(slide, left, top, width, height, title=None):
    """Creates a stylized, rounded card container with crisp typography."""
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = CLR_CARD_BG
    card.line.color.rgb = CLR_CARD_BORDER
    card.line.width = Pt(1.0)
    
    tb = slide.shapes.add_textbox(Inches(left + 0.18), Inches(top + 0.14), Inches(width - 0.36), Inches(height - 0.28))
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

def build_slide_1(prs):
    """Slide 1: Geometric Simplex Volume & Orthogonal Heights."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide,
        "Geometric Simplex Volume from Iterative Orthogonal Heights",
        "Iterative orthogonal projection calculates simplex volume directly from perpendicular heights without dimensionality reduction."
    )

    # Top Left Card: Geometric Endmember Selection
    tf_left = create_card(slide, 0.60, 1.35, 5.95, 2.75, "Iterative Endmember Selection")
    
    steps = [
        ("Anchor Initialization (v0): ", "Select initial vertex v0 (e.g. pixel furthest from origin or spatial mean)."),
        ("First Endmember (v1): ", "Find pixel v1 with maximum Euclidean distance h0 = ||v1 - v0||. Forms 1D base segment."),
        ("Second Endmember (v2): ", "Project all pixels orthogonal to segment v0-v1. The maximum perpendicular distance identifies v2 with height h1, forming a 2D triangle."),
        ("Third Endmember (v3): ", "Project all pixels orthogonal to plane v0-v1-v2. The maximum perpendicular distance identifies v3 with height h2, forming a 3D tetrahedron.")
    ]
    for bold_lead, body in steps:
        p = tf_left.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + bold_lead
        r1.font.name = FONT_FAMILY
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = CLR_PRIMARY_NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_FAMILY
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = CLR_DARK_TEXT
        p.space_after = Pt(3)

    # Top Right Card: Simplex Volume Progression
    tf_right = create_card(slide, 6.78, 1.35, 5.95, 2.75, "General Simplex Volume Formula")

    p = tf_right.add_paragraph()
    p.text = "• Geometric Factorial Volume Formula:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.bold = True
    p.font.color.rgb = CLR_PRIMARY_NAVY

    p = tf_right.add_paragraph()
    p.text = "     Volume_k = (1 / k!) * (h0 * h1 * h2 * ... * h_{k-1})"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_MATH_NAVY
    p.space_after = Pt(4)

    notes = [
        ("Base Simplex (1D): ", "Length = h0 (1! = 1)."),
        ("Triangle Simplex (2D): ", "Area = (1/2!) * h0 * h1 = 0.5 * Base * Height."),
        ("Tetrahedron Simplex (3D): ", "Volume = (1/3!) * h0 * h1 * h2 = (1/6) * Base_Area * Height."),
        ("Arbitrary Dimension: ", "Volume of any k-dimensional simplex equals (1/k!) times the product of orthogonal heights, requiring no coordinate system rotation or PCA truncation.")
    ]
    for bold_lead, body in notes:
        p = tf_right.add_paragraph()
        r1 = p.add_run()
        r1.text = "• " + bold_lead
        r1.font.name = FONT_FAMILY
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = CLR_PRIMARY_NAVY
        r2 = p.add_run()
        r2.text = body
        r2.font.name = FONT_FAMILY
        r2.font.size = Pt(9.5)
        r2.font.color.rgb = CLR_DARK_TEXT
        p.space_after = Pt(2)

    # Lower Half: Simplistic Diagram
    if os.path.exists(DIAGRAM_PATH):
        slide.shapes.add_picture(DIAGRAM_PATH, Inches(0.60), Inches(4.25), Inches(12.13), Inches(2.95))

def build_slide_2(prs):
    """Slide 2: Gram-Space Formulation & Exact Cholesky Downdate."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide,
        "Gram-Space Formulation & Exact Cholesky Downdate",
        "Schur complement downdates track orthogonal distances in O(N*k) operations without matrix inversions."
    )

    # Left Card: Gram Space Formulation & Distance Initialization
    tf_left = create_card(slide, 0.60, 1.35, 5.95, 5.80, "Gram-Space Representation & Initialization")

    p = tf_left.add_paragraph()
    p.text = "1. Localized Spatial Gram Matrix"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_left.add_paragraph()
    p.text = "• Local pixel window Y = [y_1, y_2, ..., y_N] with B bands and N pixels (e.g. 3x3 window, N=9)."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "• Form the N x N spatial Gram matrix once:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "     K = Y^T * Y,   where K_ij = dot_product(y_i, y_j)"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = CLR_MATH_NAVY
    p.space_after = Pt(6)

    p = tf_left.add_paragraph()
    p.text = "2. Distance to Initial Reference Vertex (v0)"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_left.add_paragraph()
    p.text = "• Squared Euclidean distance from each pixel y_i to vertex v0 = y_i0 is evaluated directly from Gram entries:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "     d_i = ||y_i - v0||^2 = K_ii - 2 * K_i,i0 + K_i0,i0"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = CLR_MATH_NAVY
    p.space_after = Pt(6)

    p = tf_left.add_paragraph()
    p.text = "3. Greedy Pivot Selection"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_left.add_paragraph()
    p.text = "• At step k, select pivot index with maximum orthogonal distance:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "     i_k = index of max(d_i),   orthogonal height h_k = sqrt(d_{i_k})"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = CLR_MATH_NAVY

    # Right Card: Cholesky Update & Schur Downdate
    tf_right = create_card(slide, 6.78, 1.35, 5.95, 5.80, "Cholesky Factorization & Schur Downdate")

    p = tf_right.add_paragraph()
    p.text = "4. Cholesky Factor Column Update"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_right.add_paragraph()
    p.text = "• Compute column k of lower triangular Cholesky factor L in R^{N x k}:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_right.add_paragraph()
    p.text = "     L_{i,k} = (1 / h_k) * [ K_{i,i_k} - sum_{j=1}^{k-1} L_{i,j} * L_{i_k,j} ]"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10.5)
    p.font.bold = True
    p.font.color.rgb = CLR_MATH_NAVY
    p.space_after = Pt(6)

    p = tf_right.add_paragraph()
    p.text = "5. Exact Schur Complement Downdate"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_right.add_paragraph()
    p.text = "• Update residual squared distance for all unselected pixels:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_right.add_paragraph()
    p.text = "     d_i <-- d_i - (L_{i,k})^2"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_MATH_NAVY

    p = tf_right.add_paragraph()
    p.text = "• Exact Subspace Equivalence: d_i exactly equals the squared perpendicular distance of pixel y_i to the subspace spanned by {v0, v1, ..., v_k}."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(9.5)
    p.font.color.rgb = CLR_DARK_TEXT
    p.space_after = Pt(6)

    p = tf_right.add_paragraph()
    p.text = "6. Algorithmic Advantages"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    advantages = [
        ("No Matrix Inversions: ", "Avoids computing BxB projection matrices P = I - U(U^T U)^-1 U^T."),
        ("O(N*k) Scaling: ", "Reduces computational complexity from O(B^2 N + B^3) to O(N*k) per step."),
        ("1.5x - 2.0x Faster: ", "Significantly accelerates sliding spatial window filters over large flightlines.")
    ]
    for b_lead, b_body in advantages:
        p = tf_right.add_paragraph()
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

def build_slide_3(prs):
    """Slide 3: Residual Tracking & Physical Noise Floor Identification."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(
        slide,
        "Residual Tracking & Physical Noise Floor Identification",
        "Degree-of-freedom pooling isolates residual sensor noise, identifying intrinsic material rank without a priori noise data."
    )

    # Left Card: Residual Sum of Squares & Degree-of-Freedom Variance
    tf_left = create_card(slide, 0.60, 1.35, 5.95, 5.80, "Residual Energy & Degree-of-Freedom Variance")

    p = tf_left.add_paragraph()
    p.text = "1. Total Residual Sum of Squares (RSS)"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_left.add_paragraph()
    p.text = "• Summing updated squared distances d_i across unselected pixels gives the total unmodeled residual energy:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "     RSS_k = sum_{i not in selected} d_i"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_MATH_NAVY
    p.space_after = Pt(6)

    p = tf_left.add_paragraph()
    p.text = "2. Degrees of Freedom Accounting"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_left.add_paragraph()
    p.text = "• Total data elements in local window: B * N measurements."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "• Selecting k endmembers consumes:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "     - k spectral projection directions (leaving B - k bands)"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(9.5)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "     - k spatial pivot samples (leaving N - k unselected pixels)"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(9.5)
    p.font.color.rgb = CLR_DARK_TEXT
    p.space_after = Pt(6)

    p = tf_left.add_paragraph()
    p.text = "3. Pooled Noise Standard Deviation"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_left.add_paragraph()
    p.text = "• Unbiased pooled estimate of residual noise standard deviation:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_left.add_paragraph()
    p.text = "     sigma_hat_k = sqrt( RSS_k / [ (B - k) * (N - k) ] )"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_MATH_NAVY

    # Right Card: Drop Ratio & Physical Stopping Criterion
    tf_right = create_card(slide, 6.78, 1.35, 5.95, 5.80, "Residual Variance Drop Ratio (rho_k)")

    p = tf_right.add_paragraph()
    p.text = "4. Step-to-Step Drop Ratio Metric"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_right.add_paragraph()
    p.text = "• Ratio of successive residual noise standard deviations:"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(10)
    p.font.color.rgb = CLR_DARK_TEXT

    p = tf_right.add_paragraph()
    p.text = "     rho_k = sigma_hat_{k-1} / sigma_hat_k"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_MATH_NAVY
    p.space_after = Pt(6)

    p = tf_right.add_paragraph()
    p.text = "5. Physical Signal vs. Noise Regimes"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    regimes = [
        ("Material Signal Regime (rho_k >= 3.0): ", "Adding a distinct physical material removes macroscopic subspace variance, producing a sharp drop (empirically 4x to 18x on flightline scenes)."),
        ("Noise Floor Regime (rho_k approx 1.0x): ", "When only sensor noise remains, projecting arbitrary noise directions removes only nominal degrees of freedom (~1/(N-k)), locking rho_k at 1.01x to 1.06x.")
    ]
    for b_lead, b_body in regimes:
        p = tf_right.add_paragraph()
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
        p.space_after = Pt(3)

    p = tf_right.add_paragraph()
    p.text = "6. Scale-Free Stopping Criterion"
    p.font.name = FONT_FAMILY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = CLR_ACCENT_TEAL

    p = tf_right.add_paragraph()
    p.text = "• Intrinsic rank k* is reached at the first step where rho_k falls below ~1.5x, stopping objectively across sensors without requiring an a priori noise calibration."
    p.font.name = FONT_FAMILY
    p.font.size = Pt(9.5)
    p.font.color.rgb = CLR_DARK_TEXT

def main():
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.500)

    build_slide_1(prs)
    build_slide_2(prs)
    build_slide_3(prs)

    prs.save(OUTPUT_PPTX)
    print(f"Successfully generated standalone deck: {OUTPUT_PPTX}")

if __name__ == "__main__":
    main()
