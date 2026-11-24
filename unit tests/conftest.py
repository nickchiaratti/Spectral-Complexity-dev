"""
Pytest configuration and environment initialization for unit tests in Spectral-Complexity-dev.

Injects all segregated development directories into sys.path:
- Repository root
- HLS30/
- Tanager-1/
- Harmonized_SC/
- change_detection/SC-CCM/
- change_detection/harmonized-CCD/
- change_detection/pyccd/
- simulated-data-test/
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


def resolve_repo_file(filename: str) -> Path:
    """
    Locates a repository file either at the repository root or within any segregated development folder.
    """
    direct = REPO_ROOT / filename
    if direct.exists():
        return direct
    for sub in DEV_SUBDIRS:
        if sub:
            cand = REPO_ROOT / sub / filename
            if cand.exists():
                return cand
    matches = list(REPO_ROOT.glob(f"**/{filename}"))
    for m in matches:
        if ".git" not in m.parts:
            return m
    return direct
