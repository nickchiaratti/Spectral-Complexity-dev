import sys
import os
import numpy as np
from scipy.ndimage import uniform_filter
import spectral.io.envi as envi

# Add the parent directory to Python path to import SpecComplex
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import SpecComplex

def main():
    if len(sys.argv) < 7:
        print("Usage: envi_bridge.py <task_name> <input_uri> <output_uri> <tile_size> <stride> <num_endmembers>")
        sys.exit(1)
        
    task_name = sys.argv[1]
    input_uri = sys.argv[2]
    output_uri = sys.argv[3]
    tile_size = int(sys.argv[4])
    stride = int(sys.argv[5])
    num_endmembers = int(sys.argv[6])
    
    # Handle optional .dat vs .hdr extensions safely
    hdr_uri = input_uri + '.hdr' if not input_uri.endswith('.hdr') else input_uri
    dat_uri = input_uri.replace('.hdr', '') if input_uri.endswith('.hdr') else input_uri
    
    print(f"Reading {dat_uri}...")
    img = envi.open(hdr_uri, dat_uri)
    data = img.load() # Shape: (rows, cols, bands)
    
    print(f"Processing Spectral Complexity ({task_name})...")
    vol_map = SpecComplex.process_volume_sliding_tile(data, tile_size, stride, num_endmembers)
    
    if task_name == 'final':
        out_map = uniform_filter(vol_map, size=tile_size, mode='constant')
        band_name = 'Spectral Complexity Final Map'
    else:
        out_map = vol_map
        band_name = 'Spectral Complexity Neighborhood Map'
        
    md = img.metadata.copy()
    md['bands'] = 1
    md['band names'] = [band_name]
    md['data ignore value'] = 'NaN'
    
    out_hdr_uri = output_uri + '.hdr' if not output_uri.endswith('.hdr') else output_uri
    out_dat_uri = output_uri.replace('.hdr', '') if output_uri.endswith('.hdr') else output_uri
    
    print(f"Saving to {out_dat_uri}...")
    out_img = envi.create_image(out_hdr_uri, md, ext='', force=True)
    out_mm = out_img.open_memmap(writable=True)
    out_mm[:,:,0] = out_map
    
    print("Done!")

if __name__ == '__main__':
    main()
