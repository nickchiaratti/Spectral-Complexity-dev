"""
test_qr_equivalence.py
======================
Compares process_volume_sliding_tile_parallel  (Gram/determinant volumes)
     vs  process_volume_sliding_tile_parallel_QR  (QR-decomposition volumes)

Both functions share the same maximumDistance() endmember extraction and the
same parallel row-chunking structure.  The only difference is the volume
computation kernel:
  - Gram path : vol_i = sqrt(det(G[:i,:i]))  where  G = V^T V
  - QR path   : vol_i = prod(|diag(R[:i,:i])|)  where  V = QR

These are mathematically equivalent (det(V^T V) = det(R)^2 for economy QR of
a tall-thin matrix), so output values should be identical to float64 precision.

Two test suites are included:
  1. TestQREquivalence          -- synthetic random data (always runs)
  2. TestQREquivalenceRealData  -- real surface_reflectance frames from the
                                   HLST ARD HDF5 cube (skipped if file absent)

Run:
    python test_qr_equivalence.py
"""

import os
import time
import unittest
import multiprocessing
import numpy as np

from SpecComplexQR import (
    process_volume_sliding_tile_parallel,
    process_volume_sliding_tile_parallel_QR,
)

# ---------------------------------------------------------------------------
# Test configuration (matches user specification)
# ---------------------------------------------------------------------------
BANDS          = 10
HEIGHT         = 200
WIDTH          = 200
TILE_SIZE      = 3
STRIDE         = 1
NUM_ENDMEMBERS = 7
GRAM_TYPE      = 'minEndmember'
NORM_TYPE      = 'bandCount'
N_JOBS         = max(2, multiprocessing.cpu_count())

# Real ARD data source
ARD_H5_PATH     = r"C:\satelliteImagery\HLST30\HLST_Tait_Harmonized.h5"
ARD_GRIDS       = ['HLSL30', 'HLSS30']   # sensor grids to sample from
MAX_REAL_FRAMES = 10                       # limit for runtime; raise for exhaustive check


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timed(fn, *args, **kwargs):
    """Return (result, elapsed_s)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def _banner(title, width=65):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def _print_comparison(label, t_gram, t_qr):
    """Print a formatted timing comparison table."""
    speedup = t_gram / t_qr if t_qr > 0 else float('inf')
    delta_s = t_gram - t_qr
    faster  = "QR faster" if delta_s > 0 else "Gram faster"
    print(f"\n  {label}")
    print(f"  {'Function':<42}  {'Time (s)':>10}  {'vs Gram':>10}")
    print(f"  {'-'*42}  {'-'*10}  {'-'*10}")
    print(f"  {'process_volume_sliding_tile_parallel':<42}  {t_gram:>10.3f}  {'(baseline)':>10}")
    print(f"  {'process_volume_sliding_tile_parallel_QR':<42}  {t_qr:>10.3f}  {speedup:>9.2f}x")
    print(f"\n  Wall-clock delta : {abs(delta_s):.3f} s  ({faster})")


def _value_report(gram_map, qr_map, h, w, tile_size, label=""):
    """Print a statistical comparison between two output maps."""
    diff     = qr_map.astype(np.float64) - gram_map.astype(np.float64)
    abs_diff = np.abs(diff)
    denom    = np.abs(gram_map.astype(np.float64))
    with np.errstate(invalid='ignore', divide='ignore'):
        rel_diff = np.where(denom > 0, abs_diff / denom, 0.0)

    interior = (slice(tile_size, h - tile_size),
                slice(tile_size, w - tile_size))

    tag = f"  [{label}] " if label else "  "
    print(f"\n{tag}Value comparison  full map {h}x{w}:")
    print(f"    Max absolute difference : {abs_diff.max():.6e}")
    print(f"    Mean absolute difference: {np.nanmean(abs_diff):.6e}")
    print(f"    Max relative difference : {rel_diff.max():.6e}")
    print(f"    Mean relative difference: {np.nanmean(rel_diff):.6e}")
    exact = int(np.sum(gram_map == qr_map))
    print(f"    Pixel-exact matches     : {exact:,} / {gram_map.size:,}")

    abs_int = abs_diff[interior]
    print(f"\n{tag}Value comparison  interior {h-2*tile_size}x{w-2*tile_size}:")
    print(f"    Max absolute difference : {abs_int.max():.6e}")
    print(f"    Mean absolute difference: {np.nanmean(abs_int):.6e}")


def _load_real_frames(h5_path, grids, max_frames):
    """
    Load up to max_frames surface_reflectance frames from the ARD HDF5 cube.

    Follows the HLST_SC_calculations.py ingestion pattern exactly:
      - surface_reflectance : [n_frames, bands, H, W]  float32  (NaN = fill)
      - common_mask         : [n_frames, H, W]          uint8   (1 = valid)
      - Pixels where common_mask != 1 are set to NaN before processing.

    Returns a list of dicts:
        {'frame_data': np.ndarray [bands, H, W] float32,
         'label'     : str,
         'valid_pct' : float,
         'bands'/'height'/'width': int}
    """
    import h5py

    frames = []
    with h5py.File(h5_path, 'r') as h5:
        for grid in grids:
            dset_path = f"/HDFEOS/GRIDS/{grid}/Data Fields"
            if dset_path not in h5:
                print(f"  [skip] {grid} not found in {h5_path}")
                continue

            dg   = h5[dset_path]
            sr   = dg['surface_reflectance']    # [n_frames, bands, H, W]
            mask = dg['common_mask']            # [n_frames, H, W]

            n_frames = sr.shape[0]
            n_bands  = sr.shape[1]
            height   = sr.shape[2]
            width    = sr.shape[3]

            acq_times = sr.attrs.get('acquisition_time', np.zeros(n_frames))

            for t in range(n_frames):
                if len(frames) >= max_frames:
                    break

                # Read one frame -- mirrors compute_frame_metrics data ingestion
                frame_sr   = sr[t, ...].astype(np.float32)   # [bands, H, W]
                frame_mask = mask[t, ...]                      # [H, W]

                # Apply common_mask: invalid pixels -> NaN  (same as calc script)
                invalid = frame_mask != 1
                frame_sr[:, invalid] = np.nan

                valid_pct = 100.0 * float((~invalid).sum()) / float(invalid.size)
                ts_str    = f"{acq_times[t]:.0f}" if len(acq_times) > t else str(t)

                frames.append({
                    'frame_data': frame_sr,
                    'label'     : f"{grid}[{t}]  ts={ts_str}",
                    'valid_pct' : valid_pct,
                    'bands'     : n_bands,
                    'height'    : height,
                    'width'     : width,
                })

            if len(frames) >= max_frames:
                break

    return frames


# ---------------------------------------------------------------------------
# Suite 1 : Synthetic random data (always runs)
# ---------------------------------------------------------------------------

class TestQREquivalence(unittest.TestCase):
    """
    Confirms that process_volume_sliding_tile_parallel_QR produces values
    identical to the Gram/determinant reference on synthetic random data.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.frame_data = np.random.rand(BANDS, HEIGHT, WIDTH).astype(np.float32)

        print()
        _banner(f"Suite 1 -- Synthetic Data  "
                f"{BANDS}b {HEIGHT}x{WIDTH} tile={TILE_SIZE} "
                f"em={NUM_ENDMEMBERS} {GRAM_TYPE}/{NORM_TYPE} n_jobs={N_JOBS}")
        print(f"\n  Running both functions ...\n")

        cls.gram_map, cls.t_gram = _timed(
            process_volume_sliding_tile_parallel,
            cls.frame_data, TILE_SIZE, STRIDE, NUM_ENDMEMBERS,
            GRAM_TYPE, NORM_TYPE, N_JOBS)
        print(f"  Gram/det parallel : {cls.t_gram:.3f} s")

        cls.qr_map, cls.t_qr = _timed(
            process_volume_sliding_tile_parallel_QR,
            cls.frame_data, TILE_SIZE, STRIDE, NUM_ENDMEMBERS,
            GRAM_TYPE, NORM_TYPE, N_JOBS)
        print(f"  QR parallel       : {cls.t_qr:.3f} s")

        _print_comparison("Synthetic data", cls.t_gram, cls.t_qr)
        _value_report(cls.gram_map, cls.qr_map, HEIGHT, WIDTH, TILE_SIZE, "synthetic")

    def test_output_shapes_match(self):
        self.assertEqual(self.gram_map.shape, (HEIGHT, WIDTH))
        self.assertEqual(self.qr_map.shape,   (HEIGHT, WIDTH))

    def test_output_dtypes_are_float32(self):
        self.assertEqual(self.gram_map.dtype, np.float32)
        self.assertEqual(self.qr_map.dtype,   np.float32)

    def test_no_nan_in_outputs(self):
        interior = (slice(TILE_SIZE - 1, HEIGHT - TILE_SIZE + 1),
                    slice(TILE_SIZE - 1, WIDTH  - TILE_SIZE + 1))
        self.assertFalse(np.any(np.isnan(self.gram_map[interior])))
        self.assertFalse(np.any(np.isnan(self.qr_map[interior])))

    def test_values_are_identical(self):
        np.testing.assert_array_equal(
            self.gram_map, self.qr_map,
            err_msg=(
                "Gram and QR maps differ on synthetic data.\n"
                f"  Max |diff| = "
                f"{np.abs(self.gram_map.astype(np.float64) - self.qr_map.astype(np.float64)).max():.6e}"
            ))

    def test_qr_not_dramatically_slower(self):
        ratio = self.t_qr / self.t_gram if self.t_gram > 0 else 1.0
        self.assertLessEqual(ratio, 3.0,
            f"QR ({self.t_qr:.2f}s) is {ratio:.1f}x slower than Gram -- possible regression")


# ---------------------------------------------------------------------------
# Suite 2 : Real HLS surface_reflectance data (skipped if file absent)
# ---------------------------------------------------------------------------

@unittest.skipUnless(os.path.isfile(ARD_H5_PATH),
                     f"Real-data ARD file not found: {ARD_H5_PATH}")
class TestQREquivalenceRealData(unittest.TestCase):
    """
    Loads actual surface_reflectance frames from the HLST ARD HDF5 cube and
    verifies that the QR-path produces exactly the same output map as the
    Gram/det path on real satellite imagery.

    Data loading follows HLST_SC_calculations.py:
      - surface_reflectance  [n_frames, bands, H, W]  float32  NaN = fill
      - common_mask          [n_frames, H, W]          uint8   1 = valid
      - Invalid pixels (mask != 1) are set to NaN before processing.
    """

    @classmethod
    def setUpClass(cls):
        print()
        _banner("Suite 2 -- Real HLS Surface Reflectance Data")
        print(f"  File    : {ARD_H5_PATH}")
        print(f"  Grids   : {ARD_GRIDS}")
        print(f"  Config  : tile={TILE_SIZE}  stride={STRIDE}  em={NUM_ENDMEMBERS}  "
              f"{GRAM_TYPE}/{NORM_TYPE}  n_jobs={N_JOBS}")
        print(f"  Frames  : up to {MAX_REAL_FRAMES} frames sampled\n")

        cls.frames = _load_real_frames(ARD_H5_PATH, ARD_GRIDS, MAX_REAL_FRAMES)

        if not cls.frames:
            raise unittest.SkipTest("No frames could be loaded from ARD file.")

        print(f"  Loaded {len(cls.frames)} frame(s):")
        for fr in cls.frames:
            print(f"    {fr['label']}  "
                  f"shape=[{fr['bands']}, {fr['height']}, {fr['width']}]  "
                  f"valid={fr['valid_pct']:.1f}%")

        cls.results     = []
        total_t_gram    = 0.0
        total_t_qr      = 0.0

        print()
        for fr in cls.frames:
            fd = fr['frame_data']      # [bands, H, W]
            h, w = fr['height'], fr['width']

            gram_map, t_gram = _timed(
                process_volume_sliding_tile_parallel,
                fd, TILE_SIZE, STRIDE, NUM_ENDMEMBERS,
                GRAM_TYPE, NORM_TYPE, N_JOBS)

            qr_map, t_qr = _timed(
                process_volume_sliding_tile_parallel_QR,
                fd, TILE_SIZE, STRIDE, NUM_ENDMEMBERS,
                GRAM_TYPE, NORM_TYPE, N_JOBS)

            total_t_gram += t_gram
            total_t_qr   += t_qr

            _print_comparison(fr['label'], t_gram, t_qr)
            _value_report(gram_map, qr_map, h, w, TILE_SIZE, fr['label'])

            cls.results.append({
                'label'   : fr['label'],
                'gram_map': gram_map,
                'qr_map'  : qr_map,
                't_gram'  : t_gram,
                't_qr'    : t_qr,
                'height'  : h,
                'width'   : w,
            })

        # Aggregate summary
        _banner("Real-Data Aggregate Timing Summary")
        n = len(cls.results)
        print(f"\n  {'Metric':<36}  {'Gram/det':>10}  {'QR':>10}")
        print(f"  {'-'*36}  {'-'*10}  {'-'*10}")
        print(f"  {'Total wall time (all frames)':<36}  {total_t_gram:>10.3f}  {total_t_qr:>10.3f}")
        print(f"  {'Mean per frame':<36}  {total_t_gram/n:>10.3f}  {total_t_qr/n:>10.3f}")
        overall_speedup = total_t_gram / total_t_qr if total_t_qr > 0 else float('inf')
        faster = "QR faster" if overall_speedup > 1 else "Gram faster"
        print(f"\n  Overall speed-up : {overall_speedup:.2f}x  ({faster})")

    def test_shapes_match_across_all_frames(self):
        for r in self.results:
            with self.subTest(frame=r['label']):
                self.assertEqual(r['gram_map'].shape, (r['height'], r['width']))
                self.assertEqual(r['qr_map'].shape,   (r['height'], r['width']))

    def test_dtypes_are_float32_for_all_frames(self):
        for r in self.results:
            with self.subTest(frame=r['label']):
                self.assertEqual(r['gram_map'].dtype, np.float32)
                self.assertEqual(r['qr_map'].dtype,   np.float32)

    def test_values_identical_across_all_frames(self):
        """
        Gram and QR outputs must be bit-for-bit equal for every real frame.
        NaN locations (masked or boundary pixels) must also agree exactly.
        """
        for r in self.results:
            with self.subTest(frame=r['label']):
                gm = r['gram_map']
                qm = r['qr_map']

                # NaN pattern must match
                np.testing.assert_array_equal(
                    np.isnan(gm), np.isnan(qm),
                    err_msg=f"NaN mask differs for {r['label']}")

                # Finite pixels must be bit-for-bit equal
                finite = ~np.isnan(gm)
                if finite.any():
                    diff = np.abs(gm[finite].astype(np.float64)
                                  - qm[finite].astype(np.float64))
                    max_diff = float(diff.max())
                    self.assertEqual(
                        max_diff, 0.0,
                        f"Finite pixels differ for {r['label']}  max|diff|={max_diff:.6e}")

    def test_qr_not_dramatically_slower_real_data(self):
        for r in self.results:
            with self.subTest(frame=r['label']):
                ratio = r['t_qr'] / r['t_gram'] if r['t_gram'] > 0 else 1.0
                self.assertLessEqual(
                    ratio, 3.0,
                    f"QR is {ratio:.1f}x slower for {r['label']}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main(verbosity=2)
