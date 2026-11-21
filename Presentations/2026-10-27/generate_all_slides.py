#!/usr/bin/env python3
"""
generate_all_slides.py
Master runner script to sequentially generate all visuals for the October 27, 2026 presentation:
  1. Slide 1: Airborne HySpex flightline comparison (generate_slide1_airborne_comparisons.py)
  2. Slide 2: Foliar ground truth & ASD spectra (generate_slide2_foliar_ground_truth.py)
  3. Slide 3: Foliar complexity simulations (generate_slide3_foliar_complexity.py)
  4. Slide 4: PCA vs. Spectral Complexity scree comparison (generate_slide4_pca_vs_complexity.py)
  5. Slide 5: Multi-sensor foliar complexity simulation (generate_slide5_foliar_complexity_multi_sensor.py)

Executes each script sequentially with its default processes and parameters without
modifying any configuration arguments.

Designed for IDE execution:
  - Simply open in VS Code / PyCharm and run (Run / F5).
"""

import os
import sys
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordered sequence of slide generation scripts
SLIDE_SCRIPTS = [
    "generate_slide1_airborne_comparisons.py",
    "generate_slide2_foliar_ground_truth.py",
    "generate_slide3_foliar_complexity.py",
    "generate_slide4_pca_vs_complexity.py",
    "generate_slide5_foliar_complexity_multi_sensor.py",
]


def run_slide_script(script_name: str, index: int, total: int) -> None:
    """Execute a slide generator script as a subprocess using default arguments."""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Slide script not found: {script_path}")

    separator = "=" * 78
    print(f"\n{separator}")
    print(f"[{index}/{total}] Starting: {script_name}")
    print(f"Path: {script_path}")
    print(f"{separator}\n", flush=True)

    start_time = time.time()

    # Run script with default process in the current Python environment
    subprocess.run(
        [sys.executable, script_path],
        cwd=SCRIPT_DIR,
        check=True
    )

    elapsed = time.time() - start_time
    print(f"\n[Completed] {script_name} in {elapsed:.2f}s\n", flush=True)


def main() -> None:
    """Sequentially execute all slide generation scripts."""
    total_start = time.time()
    total_scripts = len(SLIDE_SCRIPTS)

    print("=" * 78)
    print("SEQUENTIAL SLIDE VISUAL GENERATION RUNNER")
    print(f"Executing {total_scripts} presentation visual scripts in order...")
    print("=" * 78, flush=True)

    for idx, script_name in enumerate(SLIDE_SCRIPTS, start=1):
        run_slide_script(script_name, idx, total_scripts)

    total_elapsed = time.time() - total_start
    print("=" * 78)
    print(f"SUCCESS: All {total_scripts} slide generation scripts completed successfully.")
    print(f"Total elapsed time: {total_elapsed:.2f}s")
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()

