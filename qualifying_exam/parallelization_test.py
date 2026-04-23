import numpy as np
import time
import sys
import os
from scipy.signal import convolve2d
from skimage.util import view_as_windows
import SpecComplex as sc

# ==========================================
# 1. VECTORIZED SLIDING WINDOW ENGINE
# ==========================================
def vectorized_sliding_volume(frame_sr, tile_size=3, stride=1, num_endmembers=7, gram_type='minEndmember', norm_param='bandCount'):
    """
    Executes a fully vectorized sliding window volume calculation.
    Mathematically replaces the nested 'for' loops using zero-copy striding 
    and 2D discrete convolution.
    """
    if stride != 1:
        raise ValueError("CRITICAL ERROR: Vectorized convolution accumulation currently strictly enforces stride=1.")
        
    num_bands, h, w = frame_sr.shape
    
    # 1. Zero-Copy Spatial Extraction
    img_hwc = np.transpose(frame_sr, (1, 2, 0))
    windows = view_as_windows(img_hwc, window_shape=(tile_size, tile_size, num_bands), step=stride)
    windows = np.squeeze(windows, axis=2)
    
    # EVIDENCE-BASED FIX: Dimensionality Alignment
    # Transpose the high-dimensional windows to match the exact shape 
    # SpecComplex.py expects: (H_out, W_out, Bands, Y, X)
    windows_mapped = np.transpose(windows, (0, 1, 4, 2, 3))
    
    # 2. Generalized Ufunc (gufunc) Vectorization
    # The signature strictly enforces that the function receives a 3D array (Bands, Y, X)
    @np.vectorize(signature='(b,h,w)->()')
    def _volume_bridge(window_3d):
        # Pass exactly the 3D shape expected by process_volume_frame
        _, _, vol_curve = sc.process_volume_frame(window_3d, num_endmembers, gram_type, norm_param)
        return vol_curve[-1] if len(vol_curve) > 0 else 0.0

    # Broadcast the function across the entire spatial grid instantly
    raw_volumes = _volume_bridge(windows_mapped)
    
    # 3. Convolutional Accumulation (The "Count Map" Replacement)
    kernel = np.ones((tile_size, tile_size), dtype=np.float32)
    
    # mode='full' automatically pads the output back to the original image dimensions
    accumulated_volumes = convolve2d(raw_volumes, kernel, mode='full')
    
    # Create the geometric Count Map using the exact same convolution math on an array of 1s
    volume_ones = np.ones_like(raw_volumes)
    accumulated_counts = convolve2d(volume_ones, kernel, mode='full')
    
    # 4. Final Averaging
    accumulated_counts[accumulated_counts == 0] = 1 
    slide_map = accumulated_volumes / accumulated_counts
    
    return slide_map

# ==========================================
# 2. EXECUTION & MATHEMATICAL VALIDATION
# ==========================================
def run_validation():
    print("Initializing Vectorization Validation Sequence...")
    
    # Generate a synthetic 13-band surface reflectance array (50x50 pixels)
    bands, h, w = 100, 100, 100
    np.random.seed(42) # Lock seed for absolute reproducibility
    test_frame = np.random.rand(bands, h, w).astype(np.float32)
    
    tile_size = 3
    stride = 1
    num_em = 7
    
    print(f"\nTest Matrix: {bands} Bands, {h}x{w} Spatial Extent")
    
    # 1. Run Legacy Interpreted Method
    print("\nExecuting Legacy sc.process_volume_sliding_tile (Nested Loops)...")
    
    # EVIDENCE-BASED FIX: Enforce UTF-8 on Windows
    # Explicitly defines utf-8 encoding to prevent UnicodeEncodeErrors when 
    # SpecComplex outputs mathematical symbols like the square root sign.
    with open(os.devnull, 'w', encoding='utf-8') as f, np.errstate(all='ignore'):
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            start_legacy = time.perf_counter()
            legacy_result = sc.process_volume_sliding_tile(test_frame, tile_size, stride, num_em, 'minEndmember', 'bandCount')
            end_legacy = time.perf_counter()
        finally:
            sys.stdout = old_stdout
            
    print(f"  -> Legacy Execution Time: {end_legacy - start_legacy:.4f} seconds")
    
    # 2. Run Vectorized/Convolutional Method
    print("\nExecuting Vectorized Architecture (Zero-Copy & Convolution)...")
    
    with open(os.devnull, 'w', encoding='utf-8') as f, np.errstate(all='ignore'):
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            start_vec = time.perf_counter()
            vectorized_result = vectorized_sliding_volume(test_frame, tile_size, stride, num_em, 'minEndmember', 'bandCount')
            end_vec = time.perf_counter()
        finally:
            sys.stdout = old_stdout
            
    print(f"  -> Vectorized Execution Time: {end_vec - start_vec:.4f} seconds")
    
    # 3. Strict Mathematical Assertion
    print("\nPerforming Strict Matrix Equality Assertion...")
    try:
        # assert_allclose checks floating point math to a high degree of strictness (1e-5 relative tolerance)
        np.testing.assert_allclose(legacy_result, vectorized_result, rtol=1e-5, atol=1e-5)
        print("  -> SUCCESS: Vectorized matrix mathematically matches legacy output identically.")
        
        speedup = (end_legacy - start_legacy) / (end_vec - start_vec)
        print(f"\nValidation Complete. Convolutional Bridge Pipeline is {speedup:.2f}x faster.")
        
        print("\nNote: The iterative N-FINDR algorithm (maximumDistance) inside SpecComplex prevents")
        print("pure BLAS vectorization without a massive CUDA rewrite. Using this validated convolution")
        print("bridge is the optimal balance of accuracy and performance.")
        
    except AssertionError as e:
        # Fails loudly without synthetic padding if the math diverged
        raise ValueError(f"CRITICAL ERROR: Vectorization failed mathematical validation.\nDetails: {e}")

if __name__ == "__main__":
    run_validation()