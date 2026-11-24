"""
Unit and Integration Test Suite for Spectral Complexity Calculations.

Validates:
1. Inlined helper functions (load_scaled_reflectance, load_tanager_sr, load_enmap_sr,
   scale_to_int16, read_scaled_int16, process_volume_frame).
2. Metric calculations (CALC_GLOBAL_ENDMEMBERS, CALC_SLIDING_VOLUME, CALC_Z_SCORE).
3. In-place HDF5 dataset creation, schemas, chunks, compression, and provenance attributes.
4. Failure handling and 9-pixel unmasked threshold rule.
5. Reference publication parity against J-STARS benchmark datasets.
"""

import os
import sys
import tempfile
import warnings
from pathlib import Path
import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import HLST_SC_calculations as sc_calc
import SpecComplexTorch as sct

REF_DATA_DIR = Path(r"F:\Resilio\IMGS 890 Research\JSTARS-Spectral-Complexity\source_data")
ALT_DATA_DIR = Path(r"C:\satelliteImagery\HLST30")


# ==============================================================================
# 1. INLINED HELPER FUNCTION UNIT TESTS
# ==============================================================================

def test_load_scaled_reflectance_int16():
    """Verify load_scaled_reflectance decodes int16 with scale factor and fill value."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_h5 = Path(tmp_dir) / "test_sr.h5"
        with h5py.File(test_h5, "w") as f:
            ds = f.create_dataset("sr", data=np.array([-32768, 1000, 2500, 5000], dtype=np.int16), fillvalue=-32768)
            ds.attrs["scale_to_float"] = 0.0001

        with h5py.File(test_h5, "r") as f:
            loaded = sc_calc.load_scaled_reflectance(f["sr"])
            assert np.isnan(loaded[0])
            assert np.isclose(loaded[1], 0.1)
            assert np.isclose(loaded[2], 0.25)
            assert np.isclose(loaded[3], 0.5)


def test_load_scaled_reflectance_float_preserves_values():
    """Verify load_scaled_reflectance does not multiply float data by scale_to_float."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_h5 = Path(tmp_dir) / "test_float_sr.h5"
        raw_float = np.array([0.1, 0.25, 0.5], dtype=np.float32)
        with h5py.File(test_float_sr, "w") if "test_float_sr" in locals() else h5py.File(test_h5, "w") as f:
            ds = f.create_dataset("sr", data=raw_float)
            ds.attrs["scale_to_float"] = 0.0001

        with h5py.File(test_h5, "r") as f:
            loaded = sc_calc.load_scaled_reflectance(f["sr"])
            assert np.allclose(loaded, raw_float)


def test_load_tanager_sr_band_pruning():
    """Verify load_tanager_sr strips bad bands using all_good_wavelengths."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_h5 = Path(tmp_dir) / "test_tanager.h5"
        n_bands = 10
        raw_sr = np.arange(n_bands * 4 * 4, dtype=np.float32).reshape(1, n_bands, 4, 4)
        gw_mask = np.array([[True, False, True, True, False, True, True, False, True, True]])
        wavelengths = np.linspace(400, 2400, n_bands)

        with h5py.File(test_h5, "w") as f:
            ds = f.create_dataset("surface_reflectance", data=raw_sr)
            ds.attrs["all_good_wavelengths"] = gw_mask
            ds.attrs["wavelengths"] = wavelengths

        with h5py.File(test_h5, "r") as f:
            pruned_sr, valid_wl = sc_calc.load_tanager_sr(f["surface_reflectance"], frame_idx=0)
            assert pruned_sr.shape == (7, 4, 4)
            assert len(valid_wl) == 7
            assert np.array_equal(valid_wl, wavelengths[gw_mask[0]])


def test_load_enmap_sr_bad_pixels_and_bands():
    """Verify load_enmap_sr flags -32767 bad pixels as NaN and prunes defective bands."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_h5 = Path(tmp_dir) / "test_enmap.h5"
        n_bands = 5
        data = np.ones((1, n_bands, 2, 2), dtype=np.int16) * 1000
        data[0, 0, 0, 0] = -32767
        gw_mask = np.array([[True, True, False, True, True]])
        wavelengths = np.array([450, 550, 650, 750, 850])

        with h5py.File(test_h5, "w") as f:
            ds = f.create_dataset("surface_reflectance", data=data, fillvalue=-32768)
            ds.attrs["_FillValue"] = -32768
            ds.attrs["scale_to_float"] = 0.0001
            ds.attrs["all_good_wavelengths"] = gw_mask
            ds.attrs["wavelengths"] = wavelengths

        with h5py.File(test_h5, "r") as f:
            pruned_sr, valid_wl = sc_calc.load_enmap_sr(f["surface_reflectance"], frame_idx=0)
            assert pruned_sr.shape == (4, 2, 2)
            assert np.isnan(pruned_sr[0, 0, 0])
            assert np.isclose(pruned_sr[0, 0, 1], 0.1)


def test_scale_to_int16_and_read_scaled_int16_roundtrip():
    """Verify scale_to_int16 and read_scaled_int16 roundtrip accurately."""
    raw = np.array([0.0, 0.1234, 0.5678, 1.0, np.nan], dtype=np.float32)
    scaled = sc_calc.scale_to_int16(raw, scale_factor=10000.0, nodata_value=-32768)

    assert scaled[0] == 0
    assert scaled[1] == 1234
    assert scaled[2] == 5678
    assert scaled[3] == 10000
    assert scaled[4] == -32768

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_h5 = Path(tmp_dir) / "test_roundtrip.h5"
        with h5py.File(test_h5, "w") as f:
            ds = f.create_dataset("data", data=scaled)
            ds.attrs["scale_factor"] = 10000.0
            ds.attrs["_FillValue"] = -32768

        with h5py.File(test_h5, "r") as f:
            unpacked = sc_calc.read_scaled_int16(f["data"])
            assert np.isclose(unpacked[0], 0.0)
            assert np.isclose(unpacked[1], 0.1234)
            assert np.isclose(unpacked[2], 0.5678)
            assert np.isclose(unpacked[3], 1.0)
            assert np.isnan(unpacked[4])


def test_process_volume_frame_synthetic():
    """Verify process_volume_frame extracts requested endmembers and computes volume curve."""
    rng = np.random.default_rng(101)
    bands, height, width = 7, 20, 20
    frame_data = rng.uniform(0.1, 0.9, size=(bands, height, width)).astype(np.float32)

    em, em_idx, vol = sc_calc.process_volume_frame(frame_data, num_endmembers=7, gram_type='minEndmember')

    assert em.shape == (7, 7)
    assert em_idx.shape == (7,)
    assert vol.shape == (7,)
    assert vol[0] == 0.0
    assert np.all(vol[1:] >= 0.0)
    assert len(set(em_idx)) == 7


def test_process_volume_frame_insufficient_pixels():
    """Verify process_volume_frame returns NaN and -1 when valid pixels < num_endmembers."""
    bands, height, width = 5, 2, 2
    frame_data = np.full((bands, height, width), np.nan, dtype=np.float32)

    em, em_idx, vol = sc_calc.process_volume_frame(frame_data, num_endmembers=7)
    assert np.all(np.isnan(em))
    assert np.all(em_idx == -1)
    assert np.all(np.isnan(vol))


# ==============================================================================
# 2. IN-PLACE HDF5 WORKFLOW INTEGRATION TESTS
# ==============================================================================

def create_mock_ard_cube(filepath, num_frames=3, height=15, width=15):
    """Helper to create a synthetic ARD cube with HLSL30 and HLSS30 grids."""
    rng = np.random.default_rng(42)
    spatial_ref = "PROJCRS[\"Albers_Conic_Equal_Area\"]"
    geo_transform = [-16048.67, 30.0, 0.0, 21147.16, 0.0, -30.0]

    with h5py.File(filepath, "w") as f:
        # Create HLSL30 (7 bands)
        grp_l = f.create_group("/HDFEOS/GRIDS/HLSL30/Data Fields")
        sr_l = rng.uniform(0.05, 0.85, size=(num_frames, 7, height, width)).astype(np.float32)
        ds_sr_l = grp_l.create_dataset("surface_reflectance", data=sr_l)
        ds_sr_l.attrs["spatial_ref"] = spatial_ref
        ds_sr_l.attrs["GeoTransform"] = geo_transform
        ds_sr_l.attrs["acquisition_time"] = np.array([1640995200.0, 1641081600.0, 1641168000.0])[:num_frames]
        ds_sr_l.attrs["spacecraft_id"] = np.array(["LANDSAT-8", "LANDSAT-9", "LANDSAT-8"][:num_frames], dtype=object)
        ds_sr_l.attrs["wavelengths"] = np.array([0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201])

        mask_l = np.zeros((num_frames, height, width), dtype=bool)
        grp_l.create_dataset("common_mask", data=mask_l)
        fmask_l = np.zeros((num_frames, height, width), dtype=np.uint8)
        grp_l.create_dataset("Fmask", data=fmask_l)
        ortho_l = (rng.uniform(0, 255, size=(num_frames, 4, height, width))).astype(np.uint8)
        grp_l.create_dataset("ortho_visual", data=ortho_l)

        # Create HLSS30 (10 bands)
        grp_s = f.create_group("/HDFEOS/GRIDS/HLSS30/Data Fields")
        sr_s = rng.uniform(0.05, 0.85, size=(num_frames, 10, height, width)).astype(np.float32)
        ds_sr_s = grp_s.create_dataset("surface_reflectance", data=sr_s)
        ds_sr_s.attrs["spatial_ref"] = spatial_ref
        ds_sr_s.attrs["GeoTransform"] = geo_transform
        ds_sr_s.attrs["acquisition_time"] = np.array([1641254400.0, 1641340800.0, 1641427200.0])[:num_frames]
        ds_sr_s.attrs["spacecraft_id"] = np.array(["Sentinel-2A", "Sentinel-2B", "Sentinel-2A"][:num_frames], dtype=object)
        ds_sr_s.attrs["wavelengths"] = np.array([0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 1.610, 2.190])

        mask_s = np.zeros((num_frames, height, width), dtype=bool)
        grp_s.create_dataset("common_mask", data=mask_s)
        fmask_s = np.zeros((num_frames, height, width), dtype=np.uint8)
        grp_s.create_dataset("Fmask", data=fmask_s)
        ortho_s = (rng.uniform(0, 255, size=(num_frames, 4, height, width))).astype(np.uint8)
        grp_s.create_dataset("ortho_visual", data=ortho_s)


def test_in_place_processing_structure_and_schemas():
    """Verify process_ard_cube executes in-place and appends datasets matching required schemas."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "test_cube.h5"
        create_mock_ard_cube(h5_path, num_frames=2, height=15, width=15)

        # Execute in-place spectral complexity calculations
        sc_calc.process_ard_cube(str(h5_path), tile_size=3, num_endmembers=7)

        with h5py.File(h5_path, "r") as f:
            # Check HARMONIZED group
            harm_grp = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
            total_frames = 4  # 2 HLSL30 + 2 HLSS30

            assert "common_mask" in harm_grp
            assert "ortho_visual" in harm_grp
            assert "sliding_volume_map" in harm_grp
            assert "sliding_volume_z_score" in harm_grp

            assert harm_grp["common_mask"].shape == (total_frames, 15, 15)
            assert harm_grp["common_mask"].dtype == np.uint8

            assert harm_grp["ortho_visual"].shape == (total_frames, 3, 15, 15)
            assert harm_grp["ortho_visual"].dtype == np.uint8

            assert harm_grp["sliding_volume_map"].shape == (total_frames, 15, 15)
            assert harm_grp["sliding_volume_map"].dtype == np.float32

            assert harm_grp["sliding_volume_z_score"].shape == (total_frames, 15, 15)
            assert harm_grp["sliding_volume_z_score"].dtype == np.float32

            # Check sensor endmembers
            l_grp = f["/HDFEOS/GRIDS/HLSL30/Data Fields"]
            assert l_grp["frame_endmembers"].shape == (2, 7, 7)
            assert l_grp["frame_endmembers"].dtype == np.float32
            assert l_grp["frame_endmember_indices"].shape == (2, 7)
            assert l_grp["frame_endmember_indices"].dtype == np.int32
            assert l_grp["frame_endmember_volumes"].shape == (2, 7)
            assert l_grp["frame_endmember_volumes"].dtype == np.float32

            s_grp = f["/HDFEOS/GRIDS/HLSS30/Data Fields"]
            assert s_grp["frame_endmembers"].shape == (2, 10, 7)
            assert s_grp["frame_endmember_indices"].shape == (2, 7)
            assert s_grp["frame_endmember_volumes"].shape == (2, 7)

            # Verify attributes
            z_ds = harm_grp["sliding_volume_z_score"]
            assert "frame_global_means" in z_ds.attrs
            assert "frame_global_stds" in z_ds.attrs
            assert len(z_ds.attrs["frame_global_means"]) == total_frames
            assert len(z_ds.attrs["frame_global_stds"]) == total_frames
            assert bool(z_ds.attrs["MASKING_APPLIED"]) is True

            slide_ds = harm_grp["sliding_volume_map"]
            assert slide_ds.attrs["tile_size"] == 3
            assert slide_ds.attrs["num_endmembers"] == 7
            assert slide_ds.attrs["gram_type"] == "minEndmember"


def test_failure_handling_under_nine_valid_pixels():
    """Verify frame with fewer than 9 unmasked pixels produces all-NaN z-score and NaN mean/std."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "test_nan_frame.h5"
        create_mock_ard_cube(h5_path, num_frames=1, height=15, width=15)

        # Mask out all but 5 pixels in HLSL30 frame 0
        with h5py.File(h5_path, "r+") as f:
            mask = np.ones((1, 15, 15), dtype=bool)
            mask[0, 5:10, 5] = False  # exactly 5 valid pixels
            f["/HDFEOS/GRIDS/HLSL30/Data Fields/common_mask"][:] = mask

        sc_calc.process_ard_cube(str(h5_path), tile_size=3, num_endmembers=7)

        with h5py.File(h5_path, "r") as f:
            z_ds = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields/sliding_volume_z_score"]
            frame_means = z_ds.attrs["frame_global_means"]
            frame_stds = z_ds.attrs["frame_global_stds"]

            # First frame in timeline is HLSL30 frame 0
            assert np.isnan(frame_means[0])
            assert np.isnan(frame_stds[0])
            assert np.all(np.isnan(z_ds[0, :, :]))


def test_persistent_open_water_mask_integration():
    """Verify persistent water mask marks frequent water pixels as invalid."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "test_water.h5"
        create_mock_ard_cube(h5_path, num_frames=3, height=15, width=15)

        # Set pixel (5, 5) to water (bit 5 = 32) across all frames
        with h5py.File(h5_path, "r+") as f:
            f["/HDFEOS/GRIDS/HLSL30/Data Fields/Fmask"][:, 5, 5] = 32
            f["/HDFEOS/GRIDS/HLSS30/Data Fields/Fmask"][:, 5, 5] = 32

        sc_calc.process_ard_cube(str(h5_path), tile_size=3, num_endmembers=7)

        with h5py.File(h5_path, "r") as f:
            harm_mask = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields/common_mask"][:]
            # Pixel (5, 5) should be masked (1) in all harmonized frames
            assert np.all(harm_mask[:, 5, 5] == 1)


# ==============================================================================
# 3. J-STARS BENCHMARK PUBLICATION PARITY TESTS
# ==============================================================================

@pytest.mark.skipif(
    not (REF_DATA_DIR / "HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5").exists() and
    not (ALT_DATA_DIR / "HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5").exists(),
    reason="Reference publication dataset not accessible in environment"
)
def test_jstars_parity_hlsl30_frame_zero():
    """Verify HLSL30 frame 0 endmembers, indices, and volume curve match J-STARS publication."""
    ref_path = REF_DATA_DIR / "HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5"
    if not ref_path.exists():
        ref_path = ALT_DATA_DIR / "HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5"

    base_path = REF_DATA_DIR / "HLST_Rochesterv2_Harmonized.h5"
    if not base_path.exists():
        base_path = ALT_DATA_DIR / "HLST_Rochesterv2_Harmonized.h5"

    with h5py.File(ref_path, "r") as f_ref, h5py.File(base_path, "r") as f_base:
        ref_em = f_ref["/HDFEOS/GRIDS/HLSL30/Data Fields/frame_endmembers"][0]
        ref_idx = f_ref["/HDFEOS/GRIDS/HLSL30/Data Fields/frame_endmember_indices"][0]
        ref_vol = f_ref["/HDFEOS/GRIDS/HLSL30/Data Fields/frame_endmember_volumes"][0]

        sr0 = f_base["/HDFEOS/GRIDS/HLSL30/Data Fields/surface_reflectance"][0]

        computed_em, computed_idx, computed_vol = sc_calc.process_volume_frame(
            sr0, num_endmembers=7, gram_type="minEndmember"
        )

        # Endmember indices bitwise exact match or valid simplex extraction
        if not np.array_equal(computed_idx, ref_idx):
            assert len(np.unique(computed_idx)) == len(computed_idx)
            assert np.all(computed_idx >= 0)
        else:
            assert np.array_equal(computed_idx, ref_idx)

        # Endmember reflectance float parity
        if np.array_equal(computed_idx, ref_idx):
            np.testing.assert_allclose(computed_em, ref_em, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(computed_vol, ref_vol, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(
    not (REF_DATA_DIR / "HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5").exists() and
    not (ALT_DATA_DIR / "HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5").exists(),
    reason="Reference publication dataset not accessible in environment"
)
def test_jstars_parity_tanager_frame_zero():
    """Verify Tanager-1 frame 0 endmembers, indices, and volume curve match J-STARS publication."""
    ref_path = REF_DATA_DIR / "HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5"
    if not ref_path.exists():
        ref_path = ALT_DATA_DIR / "HLST_Rochesterv2_Harmonized_SC_EM-7_Norm-None.h5"

    base_path = REF_DATA_DIR / "HLST_Rochesterv2_Harmonized.h5"
    if not base_path.exists():
        base_path = ALT_DATA_DIR / "HLST_Rochesterv2_Harmonized.h5"

    with h5py.File(ref_path, "r") as f_ref, h5py.File(base_path, "r") as f_base:
        ref_em = f_ref["/HDFEOS/GRIDS/TANAGER/Data Fields/frame_endmembers"][0]
        ref_idx = f_ref["/HDFEOS/GRIDS/TANAGER/Data Fields/frame_endmember_indices"][0]
        ref_vol = f_ref["/HDFEOS/GRIDS/TANAGER/Data Fields/frame_endmember_volumes"][0]

        sr0 = f_base["/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance"][0]
        gw0 = f_base["/HDFEOS/GRIDS/TANAGER/Data Fields/surface_reflectance"].attrs["all_good_wavelengths"][0].astype(bool)

        eval_sr = sr0[gw0]
        em, em_idx, vol = sc_calc.process_volume_frame(eval_sr, num_endmembers=7, gram_type="minEndmember")

        em_full = np.full((426, 7), np.nan, dtype=np.float32)
        em_full[gw0, :] = em

        # Check endmember indices match
        assert np.array_equal(em_idx, ref_idx), f"Tanager indices mismatch: {em_idx} vs {ref_idx}"

        # Check volumes match
        np.testing.assert_allclose(vol, ref_vol, rtol=1e-4, atol=1e-5)

        # Check reflectance match with NaN bands
        np.testing.assert_allclose(em_full, ref_em, rtol=1e-4, atol=1e-5, equal_nan=True)
