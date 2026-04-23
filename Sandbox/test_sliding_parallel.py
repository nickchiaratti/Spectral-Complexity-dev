import unittest
import time
import multiprocessing
import numpy as np
from SourceSpecComplex import (
    process_volume_sliding_tile,
    process_volume_sliding_tile_parallel,
    process_volume_sliding_tile_gpu,
)


# TEST CONFIG

BANDS = 10
HEIGHT = 200
WIDTH = 200
TILE_SIZE = 3
STRIDE = 1
NUM_ENDMEMBERS = 7
GRAM_TYPE = 'minEndmember'
NORM_TYPE = 'bandCount'

# ---------------------------------------------------------------------------
# CuPy availability probe -- GPU tests are skipped when CuPy is not installed
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    cp.array([1.0])          # force a device initialisation to catch driver errors
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _timed(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def _print_timing(label, timings):
    """
    timings : list of (backend_label, seconds) tuples.
    First entry is treated as the serial baseline for speed-up calculation.
    """
    t_base = timings[0][1]
    print(f"\n  [{label}]")
    for name, t in timings:
        speedup = t_base / t if t > 0 else float('inf')
        marker = "(baseline)" if name == timings[0][0] else f"{speedup:.2f}x"
        print(f"    {name:<28}: {t:>8.2f} s  {marker}")


# ---------------------------------------------------------------------------
# CPU parallel tests (always run)
# ---------------------------------------------------------------------------

class TestProcessVolumeSlidingTileParallel(unittest.TestCase):
    def setUp(self):
        self.bands          = BANDS
        self.height         = HEIGHT
        self.width          = WIDTH
        self.tile_size      = TILE_SIZE
        self.stride         = STRIDE
        self.num_endmembers = NUM_ENDMEMBERS
        self.gram_type      = GRAM_TYPE
        self.norm_type      = NORM_TYPE
        self.n_jobs         = 2

        np.random.seed(42)
        self.frame_data = np.random.rand(
            self.bands, self.height, self.width).astype(np.float32)

    def test_equivalence(self):
        res_serial,   t_serial   = _timed(
            process_volume_sliding_tile,
            self.frame_data, self.tile_size, self.stride,
            self.num_endmembers, self.gram_type, self.norm_type)
        res_parallel, t_parallel = _timed(
            process_volume_sliding_tile_parallel,
            self.frame_data, self.tile_size, self.stride,
            self.num_endmembers, self.gram_type, self.norm_type, self.n_jobs)

        _print_timing(
            f"CPU parallel -- {self.bands}b {self.height}x{self.width} tile={self.tile_size}",
            [("Serial (1 worker)", t_serial),
             (f"Parallel ({self.n_jobs} workers)", t_parallel)])

        np.testing.assert_allclose(
            res_parallel, res_serial, rtol=1e-5, atol=1e-6,
            err_msg="Parallel output differs from serial output")

    def test_with_different_configs(self):
        gram_type  = 'minEndmember'
        norm_type  = 'bandCount'
        tile_size  = 5

        res_serial,   t_serial   = _timed(
            process_volume_sliding_tile,
            self.frame_data, tile_size, self.stride,
            self.num_endmembers, gram_type, norm_type)
        res_parallel, t_parallel = _timed(
            process_volume_sliding_tile_parallel,
            self.frame_data, tile_size, self.stride,
            self.num_endmembers, gram_type, norm_type, self.n_jobs)

        _print_timing(
            f"CPU parallel AltConfig -- tile={tile_size} minEndmember/bandCount",
            [("Serial (1 worker)", t_serial),
             (f"Parallel ({self.n_jobs} workers)", t_parallel)])

        np.testing.assert_allclose(
            res_parallel, res_serial, rtol=1e-5, atol=1e-6,
            err_msg="Outputs differ for minEndmember/bandCount config")


# ---------------------------------------------------------------------------
# GPU tests (skipped when CuPy is unavailable)
# ---------------------------------------------------------------------------

@unittest.skipUnless(CUPY_AVAILABLE, "CuPy not available -- GPU tests skipped")
class TestProcessVolumeSlidingTileGPU(unittest.TestCase):
    """
    Validates that process_volume_sliding_tile_gpu produces results numerically
    close to the serial CPU reference.

    Tolerance notes
    ---------------
    The GPU function uses:
      * float32 throughout (vs float64 in the CPU reference)
      * QR-based parallelotope volumes (vs sequential Gramian determinants)
      * Batched rank-1 projection (equivalent math, different FP rounding)

    These differences are mathematically equivalent but introduce ~1e-4 relative
    error, so tolerances are widened compared with the CPU-parallel test.
    """

    def setUp(self):
        self.bands          = BANDS
        self.height         = HEIGHT
        self.width          = WIDTH
        self.tile_size      = TILE_SIZE
        self.stride         = STRIDE
        self.num_endmembers = NUM_ENDMEMBERS
        self.gram_type      = GRAM_TYPE
        self.norm_type      = NORM_TYPE

        np.random.seed(7)
        self.frame_data = np.random.rand(
            self.bands, self.height, self.width).astype(np.float32)

    def test_gpu_vs_serial_minEndmember(self):
        res_serial, t_serial = _timed(
            process_volume_sliding_tile,
            self.frame_data, self.tile_size, self.stride,
            self.num_endmembers, self.gram_type, self.norm_type)
        res_gpu,    t_gpu    = _timed(
            process_volume_sliding_tile_gpu,
            self.frame_data, self.tile_size, self.stride,
            self.num_endmembers, self.gram_type, self.norm_type)

        _print_timing(
            f"GPU vs Serial -- {self.bands}b {self.height}x{self.width} "
            f"tile={self.tile_size} {self.gram_type}/{self.norm_type}",
            [("Serial CPU", t_serial), ("GPU (CuPy)", t_gpu)])

        # float32 QR vs float64 Gram/det -- expect ~1e-4 relative agreement
        np.testing.assert_allclose(
            res_gpu, res_serial,
            rtol=1e-3, atol=1e-4,
            err_msg="GPU output deviates beyond tolerance from serial reference")

    def test_gpu_vs_serial_zero_localisation(self):
        gram_type = 'zeroLocalisation'   # anything != 'minEndmember'
        norm_type = 'None'

        res_serial, t_serial = _timed(
            process_volume_sliding_tile,
            self.frame_data, self.tile_size, self.stride,
            self.num_endmembers, gram_type, norm_type)
        res_gpu,    t_gpu    = _timed(
            process_volume_sliding_tile_gpu,
            self.frame_data, self.tile_size, self.stride,
            self.num_endmembers, gram_type, norm_type)

        _print_timing(
            f"GPU vs Serial -- tile={self.tile_size} zeroLocalisation/noNorm",
            [("Serial CPU", t_serial), ("GPU (CuPy)", t_gpu)])

        np.testing.assert_allclose(
            res_gpu, res_serial,
            rtol=1e-3, atol=1e-4,
            err_msg="GPU output deviates beyond tolerance (zero-localisation path)")

    def test_gpu_shape_and_dtype(self):
        """Output must match serial shape and be float32."""
        res_serial = process_volume_sliding_tile(
            self.frame_data, self.tile_size, self.stride,
            self.num_endmembers, self.gram_type, self.norm_type)
        res_gpu = process_volume_sliding_tile_gpu(
            self.frame_data, self.tile_size, self.stride,
            self.num_endmembers, self.gram_type, self.norm_type)

        self.assertEqual(res_gpu.shape, res_serial.shape,
                         "GPU output shape does not match serial output shape")
        self.assertEqual(res_gpu.dtype, np.float32,
                         "GPU output should be float32")


# ---------------------------------------------------------------------------
# Standalone benchmark -- run as:  python test_sliding_parallel.py
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cpu_count = multiprocessing.cpu_count()
    print(f"\n{'='*65}")
    print(f"  Benchmark -- sliding tile spectral-complexity map")
    print(f"  CPU cores available : {cpu_count}")
    print(f"  CuPy available      : {CUPY_AVAILABLE}")
    print(f"{'='*65}")

    np.random.seed(42)
    bands, height, width    = BANDS, HEIGHT, WIDTH
    tile_size, stride, n_em = TILE_SIZE, STRIDE, NUM_ENDMEMBERS
    gram_type, norm_type    = GRAM_TYPE, NORM_TYPE
    frame_data = np.random.rand(bands, height, width).astype(np.float32)

    print(f"\n  Dataset  : [{bands}, {height}, {width}]   "
          f"tile={tile_size}  stride={stride}  endmembers={n_em}")
    print(f"  Settings : gram_type={gram_type}  norm_type={norm_type}\n")

    # Serial baseline
    print("  Running serial baseline ...", flush=True)
    _, t_serial = _timed(
        process_volume_sliding_tile,
        frame_data, tile_size, stride, n_em, gram_type, norm_type)

    timings = [("Serial (CPU)", t_serial)]

    # CPU-parallel sweep
    for n_jobs in sorted({1, 2, max(1, cpu_count // 2), cpu_count}):
        print(f"  Running {n_jobs}-worker parallel ...", flush=True)
        _, t_p = _timed(
            process_volume_sliding_tile_parallel,
            frame_data, tile_size, stride, n_em, gram_type, norm_type, n_jobs)
        timings.append((f"Parallel {n_jobs:>2} workers (CPU)", t_p))

    # GPU (if available)
    if CUPY_AVAILABLE:
        print("  Warming up GPU context ...", flush=True)
        process_volume_sliding_tile_gpu(
            frame_data[:, :50, :50], tile_size, stride, n_em, gram_type, norm_type)
        print("  Running GPU benchmark ...", flush=True)
        _, t_gpu = _timed(
            process_volume_sliding_tile_gpu,
            frame_data, tile_size, stride, n_em, gram_type, norm_type)
        timings.append(("GPU (CuPy)", t_gpu))
    else:
        print("  (GPU skipped -- CuPy not installed)")

    # Summary table
    print(f"\n  {'Backend':<32}  {'Time (s)':>10}  {'Speed-up':>10}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*10}")
    for name, t in timings:
        speedup = t_serial / t if t > 0 else float('inf')
        mark = "(baseline)" if name == "Serial (CPU)" else f"{speedup:.2f}x"
        print(f"  {name:<32}  {t:>10.2f}  {mark:>10}")

    print(f"\n{'='*65}\n")
    unittest.main()
