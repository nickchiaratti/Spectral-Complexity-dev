"""
End-to-End and Integration Test Suite for HarmonizedSC_run_pipeline.py.

Tests cover:
- 8-step decoupled pipeline architecture (Baseline: 1-7, Injected: 8)
- Command line interface argument parsing (steps, location, temporal bounds, config)
- Flexible step range, preset ('injected', 'baseline'), and comma-separated parser logic
- Dry-run validation modes across full chain, baseline-only, and injected-only sub-chains
- Mocked execution sequencing and artifact propagation
- Error handling and strict step failure propagation
- Clean exit code semantics (0 on success, 1 on failure)
- Real subprocess CLI invocation
- Synthetic end-to-end multi-step dataflow execution
- Strict PyTorch and legacy SpecComplex library isolation
"""

import ast
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import h5py
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import HarmonizedSC_run_pipeline as orchestrator


# =====================================================================
# 1. ARCHITECTURE AND SEQUENCING CONTRACT TESTS
# =====================================================================

def test_pipeline_step_definitions():
    """Verify that all 8 required steps are registered in exact order and partitioned correctly."""
    defs = orchestrator.STEP_DEFINITIONS
    assert len(defs) == 8

    # Baseline steps (1 - 7)
    assert defs[1]["name"] == "HLS30_earthAccess_to_hdf5"
    assert defs[1]["pipeline"] == "baseline"
    assert defs[2]["name"] == "Tanager_STAC-imagery-downloader"
    assert defs[2]["pipeline"] == "baseline"
    assert defs[3]["name"] == "HLST_constellation_to_hdf5"
    assert defs[3]["pipeline"] == "baseline"
    assert defs[4]["name"] == "HLST_SC_calculations"
    assert defs[4]["pipeline"] == "baseline"
    assert defs[5]["name"] == "pyccd_landsat_main"
    assert defs[5]["pipeline"] == "baseline"
    assert defs[6]["name"] in ("SC_CCM_main", "harmonized_CCD_main")
    assert defs[6]["pipeline"] == "baseline"
    assert defs[7]["name"] in ("SC_CCM_vis", "harmonized_CCD_vis")
    assert defs[7]["pipeline"] == "baseline"

    # Injected step (8)
    assert defs[8]["name"] == "generate_stratified_patch_benchmark"
    assert defs[8]["pipeline"] == "injected"


def test_pipeline_steps_parser_valid_ranges():
    """Verify parse_steps parses range syntax, presets, and comma-separated lists correctly."""
    assert orchestrator.parse_steps("1-7") == [1, 2, 3, 4, 5, 6, 7]
    assert orchestrator.parse_steps("8") == [8]
    assert orchestrator.parse_steps("injected") == [8]
    assert orchestrator.parse_steps("baseline") == [1, 2, 3, 4, 5, 6, 7]
    assert orchestrator.parse_steps(injected_only=True) == [8]
    assert orchestrator.parse_steps(baseline_only=True) == [1, 2, 3, 4, 5, 6, 7]
    assert orchestrator.parse_steps("1,2,3,4,5,6") == [1, 2, 3, 4, 5, 6]
    assert orchestrator.parse_steps("3-5") == [3, 4, 5]
    assert orchestrator.parse_steps("2,4,6") == [2, 4, 6]
    assert orchestrator.parse_steps("4") == [4]
    assert orchestrator.parse_steps("1-2,8") == [1, 2, 8]
    assert orchestrator.parse_steps([1, 2, 3]) == [1, 2, 3]
    assert orchestrator.parse_steps([8]) == [8]


def test_pipeline_steps_parser_invalid_inputs():
    """Verify parse_steps rejects out-of-range, descending, or malformed inputs."""
    with pytest.raises(ValueError):
        orchestrator.parse_steps("0-6")
    with pytest.raises(ValueError):
        orchestrator.parse_steps("8-11")
    with pytest.raises(ValueError):
        orchestrator.parse_steps("1-15")
    with pytest.raises(ValueError):
        orchestrator.parse_steps("5-2")
    with pytest.raises(ValueError):
        orchestrator.parse_steps("abc")
    with pytest.raises(TypeError):
        orchestrator.parse_steps(123)


# =====================================================================
# 2. CLI ARGUMENT PARSER TESTS
# =====================================================================

def test_pipeline_cli_parser_defaults():
    """Verify CLI parser default values."""
    parser = orchestrator.build_arg_parser()
    args = parser.parse_args([])
    assert args.steps is None
    assert args.injected_only is False
    assert args.baseline_only is False
    assert args.location is None
    assert args.start_date == "2015-01-01"
    assert args.end_date == "2026-06-01"
    assert args.config is None
    assert args.dry_run is False
    assert args.tile_size == 3
    assert args.num_endmembers == 7
    assert args.metric == "sliding_volume_z_score"


def test_pipeline_cli_parser_custom_options():
    """Verify CLI parser custom arguments override defaults."""
    parser = orchestrator.build_arg_parser()
    args = parser.parse_args([
        "--steps", "8",
        "--injected-only",
        "--location", "Malibu",
        "--start-date", "2023-01-01",
        "--end-date", "2025-12-31",
        "--dry-run",
        "--tile-size", "5",
        "--num-endmembers", "10",
        "--metric", "sliding_volume_map",
        "--skip-earthaccess",
    ])
    assert args.steps == ["8"]
    assert args.injected_only is True
    assert args.location == "Malibu"
    assert args.start_date == "2023-01-01"
    assert args.end_date == "2025-12-31"
    assert args.dry_run is True
    assert args.tile_size == 5
    assert args.num_endmembers == 10
    assert args.metric == "sliding_volume_map"
    assert args.skip_earthaccess is True


# =====================================================================
# 3. CONFIGURATION RESOLUTION TESTS
# =====================================================================

def test_pipeline_config_loader_fallback():
    """Verify load_pipeline_config provides standard fallback dictionary when no file found."""
    with patch.object(Path, "exists", return_value=False):
        cfg = orchestrator.load_pipeline_config(config_path="nonexistent_config_file_path.yaml")
        assert "current_run" in cfg
        assert "locations" in cfg
        assert "Rochesterv2" in cfg["locations"]
        assert cfg["locations"]["Rochesterv2"]["START_DATE"] == "2015-01-01"


def test_pipeline_config_loader_custom_file():
    """Verify load_pipeline_config correctly reads a provided YAML file."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        tmp.write("current_run:\n  location: CustomLoc\nlocations:\n  CustomLoc:\n    START_DATE: '2024-01-01'\n")
        tmp_path = tmp.name

    try:
        cfg = orchestrator.load_pipeline_config(config_path=tmp_path)
        assert cfg["current_run"]["location"] == "CustomLoc"
        assert cfg["locations"]["CustomLoc"]["START_DATE"] == "2024-01-01"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =====================================================================
# 4. DRY RUN AND SEQUENCING TESTS
# =====================================================================

def test_pipeline_dry_run_all_steps():
    """Verify dry-run mode executes all 8 steps without launching heavy compute."""
    result = orchestrator.run_pipeline(
        steps="1-8",
        location="Rochesterv2",
        dry_run=True,
    )
    assert result["status"] == "SUCCESS"
    assert result["dry_run"] is True
    assert result["executed_steps"] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_pipeline_dry_run_injected_only():
    """Verify injected-only execution isolates step 8."""
    result = orchestrator.run_pipeline(
        injected_only=True,
        location="Rochesterv2",
        dry_run=True,
    )
    assert result["status"] == "SUCCESS"
    assert result["dry_run"] is True
    assert result["executed_steps"] == [8]


def test_pipeline_dry_run_baseline_only():
    """Verify baseline-only execution isolates steps 1 through 7."""
    result = orchestrator.run_pipeline(
        baseline_only=True,
        location="Rochesterv2",
        dry_run=True,
    )
    assert result["status"] == "SUCCESS"
    assert result["dry_run"] is True
    assert result["executed_steps"] == [1, 2, 3, 4, 5, 6, 7]


def test_pipeline_dry_run_subsets_and_skip_flags():
    """Verify dry-run mode respects sub-chain ranges and skip flags."""
    result = orchestrator.run_pipeline(
        steps="1-6",
        skip_earthaccess=True,
        skip_tanager=True,
        dry_run=True,
    )
    assert result["executed_steps"] == [3, 4, 5, 6]

    result2 = orchestrator.run_pipeline(
        steps="7-8",
        dry_run=True,
    )
    assert result2["executed_steps"] == [7, 8]


# =====================================================================
# 5. MOCKED STEP EXECUTION AND ERROR PROPAGATION
# =====================================================================

def test_pipeline_mocked_injected_execution_sequence():
    """Verify mock functions are called in exact sequence for injected step (8)."""
    mock_bench = MagicMock()
    call_order = []

    def mock_import(name):
        call_order.append(name)
        if name == "generate_stratified_patch_benchmark":
            return mock_bench
        raise AssertionError(f"Unexpected module import for injected pipeline: {name}")

    with patch("importlib.import_module", side_effect=mock_import):
        with patch.object(orchestrator.os.path, "exists", return_value=True):
            result = orchestrator.run_pipeline(
                steps="8",
                location="Rochesterv2",
                h5_path="mock_master.h5",
                dry_run=False,
            )

    assert result["status"] == "SUCCESS"
    assert result["executed_steps"] == [8]
    assert call_order == ["generate_stratified_patch_benchmark"]
    assert mock_bench.MASTER_H5_PATH == "mock_master.h5"
    mock_bench.main.assert_called_once_with()
    assert result["artifacts"]["injected_h5"] == "mock_master.h5"


def test_pipeline_error_propagation_halts_chain():
    """Verify that an error in Step 2 stops subsequent steps from executing."""
    mock_hls = MagicMock()
    mock_hls.main.return_value = "native.h5"

    mock_tanager = MagicMock()
    mock_tanager.main.side_effect = RuntimeError("STAC connection failure")

    mock_harm = MagicMock()

    def mock_import(name):
        if name == "HLS30_earthAccess_to_hdf5":
            return mock_hls
        elif name == "Tanager_STAC-imagery-downloader":
            return mock_tanager
        elif name == "HLST_constellation_to_hdf5":
            return mock_harm
        return MagicMock()

    with patch("importlib.import_module", side_effect=mock_import):
        with pytest.raises(RuntimeError) as exc_info:
            orchestrator.run_pipeline(
                steps="1-6",
                location="Rochesterv2",
                dry_run=False,
            )
        assert "STAC connection failure" in str(exc_info.value)

    mock_hls.main.assert_called_once()
    mock_tanager.main.assert_called_once()
    mock_harm.main.assert_not_called()


# =====================================================================
# 6. MAIN CLI ENTRY POINT AND SUBPROCESS TESTS
# =====================================================================

def test_pipeline_main_exit_code_success():
    """Verify main returns 0 on successful dry run."""
    code = orchestrator.main(["--dry-run", "--steps", "8", "--location", "Rochesterv2"])
    assert code == 0


def test_pipeline_main_exit_code_failure():
    """Verify main returns 1 on invalid argument configuration."""
    code = orchestrator.main(["--steps", "99"])
    assert code == 1


def test_pipeline_subprocess_cli_invocation():
    """Verify real subprocess CLI execution of the pipeline orchestrator."""
    script_path = REPO_ROOT / "HarmonizedSC_run_pipeline.py"
    proc = subprocess.run(
        [sys.executable, str(script_path), "--dry-run", "--steps", "8", "--location", "Rochesterv2"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Active Steps:    [8]" in proc.stdout
    assert "PIPELINE EXECUTION COMPLETED SUCCESSFULLY" in proc.stdout


def test_pipeline_synthetic_e2e_dataflow():
    """Verify synthetic end-to-end dataflow execution on a minimal HDF5 cube."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        h5_path = Path(tmp_dir) / "HLST_SynLoc_Harmonized.h5"
        n_frames = 2
        n_bands = 6
        h, w = 4, 4

        with h5py.File(h5_path, "w") as f:
            grp = f.create_group("/HDFEOS/GRIDS/HLSL30/Data Fields")
            sr_ds = grp.create_dataset(
                "surface_reflectance",
                data=np.full((n_frames, n_bands, h, w), 0.2, dtype=np.float32)
            )
            sr_ds.attrs["acquisition_time"] = np.array([1640995200.0, 1642204800.0], dtype=np.float64)
            sr_ds.attrs["spacecraft_id"] = np.array([b"LANDSAT-8", b"LANDSAT-8"])
            sr_ds.attrs["GeoTransform"] = [-16048.67, 30.0, 0.0, 21147.16, 0.0, -30.0]
            sr_ds.attrs["spatial_ref"] = "Albers Equal Area"
            grp.create_dataset("common_mask", data=np.zeros((n_frames, h, w), dtype=np.uint8))
            grp.create_dataset("ortho_visual", data=np.zeros((n_frames, 3, h, w), dtype=np.uint8))

        # Run Step 4 (SC calculations) on synthetic cube
        res = orchestrator.run_pipeline(
            steps="4",
            location="SynLoc",
            h5_path=str(h5_path),
            tile_size=3,
            num_endmembers=2,
            dry_run=False,
        )

        assert res["status"] == "SUCCESS"
        assert 4 in res["executed_steps"]

        # Confirm datasets were appended in-place
        with h5py.File(h5_path, "r") as f:
            harm = f["/HDFEOS/GRIDS/HARMONIZED/Data Fields"]
            assert "sliding_volume_z_score" in harm
            assert "ndvi" in harm
            assert harm["sliding_volume_z_score"].shape == (n_frames, h, w)
            assert harm["ndvi"].shape == (n_frames, h, w)

            l30 = f["/HDFEOS/GRIDS/HLSL30/Data Fields"]
            assert "frame_endmembers" in l30
            assert "frame_endmember_indices" in l30
            assert "frame_endmember_volumes" in l30
            assert l30["frame_endmembers"].shape == (n_frames, n_bands, 2)


# =====================================================================
# 7. PYTORCH AND SPECCOMPLEX ISOLATION AUDIT
# =====================================================================

def test_pipeline_pytorch_and_speccomplex_isolation():
    """Verify HarmonizedSC_run_pipeline.py contains zero imports of torch or SpecComplex."""
    script_path = REPO_ROOT / "HarmonizedSC_run_pipeline.py"
    with open(script_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(script_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "torch" and not alias.name.startswith("torch."), (
                    f"HarmonizedSC_run_pipeline.py imports torch: {alias.name}"
                )
                assert alias.name != "SpecComplex", (
                    f"HarmonizedSC_run_pipeline.py imports legacy SpecComplex: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert node.module != "torch" and not node.module.startswith("torch."), (
                    f"HarmonizedSC_run_pipeline.py imports from torch: {node.module}"
                )
                assert node.module != "SpecComplex", (
                    f"HarmonizedSC_run_pipeline.py imports from legacy SpecComplex: {node.module}"
                )
