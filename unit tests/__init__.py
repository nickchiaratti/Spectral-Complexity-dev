"""
Spectral Complexity Dev Pipeline Test Suite Package.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEV_SUBDIRS = [
    "",
    "HLS30",
    "Tanager-1",
    "Harmonized_SC",
    "change_detection/SC-CCM",
    "change_detection/harmonized-CCD",
    "change_detection/pyccd",
    "simulated-data-test",
]

for sub in DEV_SUBDIRS:
    target_dir = (REPO_ROOT / sub).resolve() if sub else REPO_ROOT.resolve()
    if target_dir.exists() and str(target_dir) not in sys.path:
        sys.path.insert(0, str(target_dir))
