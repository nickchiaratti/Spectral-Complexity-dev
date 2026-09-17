import os
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))
    
from change_detection.harmonized_CCD.harmonized_CCD_vis import *
import change_detection.harmonized_CCD.harmonized_CCD_vis as orig

if __name__ == '__main__':
    print("Launching Consolidated CCD Viewer (Index Map architecture)...")
    orig.main()
