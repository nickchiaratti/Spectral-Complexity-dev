import os
import glob
import numpy as np
from scipy.spatial.distance import cdist

class USGS_SAM_Classifier:
    """
    Optimized Spectral Angle Mapper (SAM) classifier using the full USGS splib07 library.
    Maps thousands of physical laboratory spectra to Anderson Level 1 (LCMAP) classes.
    """
    
    def __init__(self, npz_path):
        """
        Initializes the classifier by loading the precompiled NumPy library arrays.
        """
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Compiled library not found at {npz_path}. Run compile_library() first.")
            
        data = np.load(npz_path)
        self.spectra = data['spectra']  # Shape: (N_spectra, N_bands)
        self.anderson_classes = data['anderson_classes']  # Shape: (N_spectra,)
        self.material_names = data['material_names']  # Shape: (N_spectra,)
        
        # Unique classes
        self.unique_classes = np.unique(self.anderson_classes)
        
    def classify_pixels(self, pixels_array):
        """
        Classifies an array of pixels using vectorized Cosine Distance (SAM).
        
        Args:
            pixels_array (np.ndarray): Shape (N_pixels, N_bands)
            
        Returns:
            list of str: The predicted Anderson Level 1 class for each pixel.
        """
        # Filter out NaN or zero-only pixels to prevent division by zero in cosine metric
        valid_mask = ~np.isnan(pixels_array).any(axis=1) & (np.sum(pixels_array, axis=1) > 0)
        
        predictions = np.array(["Unclassified"] * len(pixels_array), dtype=object)
        
        if not np.any(valid_mask):
            return predictions.tolist()
            
        valid_pixels = pixels_array[valid_mask]
        
        # Calculate cosine distance from every valid pixel to every spectrum in the library
        # cdist computes 1 - cosine_similarity. Lower is closer.
        dists = cdist(valid_pixels, self.spectra, metric='cosine')
        
        # Find the index of the closest spectrum
        best_match_indices = np.argmin(dists, axis=1)
        
        # Map back to Anderson classes
        predictions[valid_mask] = self.anderson_classes[best_match_indices]
        
        return predictions.tolist()


def compile_usgs_library(usgs_ascii_base_dir, output_npz_path, sensor_type="Landsat"):
    """
    Parses the massive USGS ASCII directory to extract spectra, apply heuristic 
    metadata mapping to Anderson classes, and compile into a fast .npz format.
    """
    print(f"Compiling USGS library for {sensor_type}...")
    
    if sensor_type == "Landsat":
        folder = "ASCIIdata_splib07b_rsLandsat8"
        band_indices = [1, 2, 3, 4, 5, 6]  # Skip coastal aerosol
    elif sensor_type == "Sentinel":
        folder = "ASCIIdata_splib07b_rsSentinel2"
        band_indices = [1, 2, 3, 8, 11, 12] # Match to the 6 HLS bands
    else:
        raise ValueError("Unsupported sensor type. Use 'Landsat' or 'Sentinel'.")
        
    search_dir = os.path.join(usgs_ascii_base_dir, folder)
    if not os.path.exists(search_dir):
        raise FileNotFoundError(f"USGS library directory not found: {search_dir}")
        
    spectra_list = []
    anderson_list = []
    material_list = []
    
    # Heuristic mapping function for Anderson Level 1 based on filename and directory
    def map_to_anderson(chapter, filename):
        fname_lower = filename.lower()
        
        # Exclude errorbars entirely
        if "errorbar" in fname_lower:
            return None
            
        if "artificial" in chapter.lower() or "coatings" in chapter.lower():
            return "Developed"
        elif "minerals" in chapter.lower() or "soils" in chapter.lower():
            return "Barren"
        elif "liquids" in chapter.lower():
            if any(k in fname_lower for k in ["snow", "ice", "frost", "glacier"]):
                return "Ice/Snow"
            else:
                return "Water"
        elif "vegetation" in chapter.lower():
            # Keyword routing for vegetation types
            if any(k in fname_lower for k in ["marsh", "phragmites", "swamp", "wetland"]):
                return "Wetland"
            elif any(k in fname_lower for k in ["corn", "wheat", "soy", "crop", "alfalfa"]):
                return "Cropland"
            elif any(k in fname_lower for k in ["grass", "sage", "shrub", "tundra", "meadow", "cheat"]):
                return "Grass/Shrub"
            elif any(k in fname_lower for k in ["pine", "oak", "maple", "spruce", "fir", "tree", "forest", "aspen", "needle", "leaf"]):
                return "Tree Cover"
            else:
                # Fallback based on typical USGS vegetation signatures
                return "Tree Cover" 
        return "Unknown"

    for root, dirs, files in os.walk(search_dir):
        chapter = os.path.basename(root)
        for f in files:
            if not f.endswith('.txt'): continue
            
            anderson_class = map_to_anderson(chapter, f)
            if anderson_class is None or anderson_class == "Unknown":
                continue
                
            file_path = os.path.join(root, f)
            try:
                with open(file_path, 'r') as fh:
                    lines = fh.readlines()
                    if len(lines) < max(band_indices) + 1:
                        continue
                    
                    header = lines[0].strip()
                    vals = [float(lines[i].strip()) for i in range(1, len(lines))]
                    
                    # Extract the selected bands
                    selected_vals = [vals[i] for i in band_indices]
                    
                    spectra_list.append(selected_vals)
                    anderson_list.append(anderson_class)
                    material_list.append(header)
            except Exception as e:
                pass # Skip corrupt or misformatted files
                
    spectra_array = np.array(spectra_list)
    anderson_array = np.array(anderson_list)
    material_array = np.array(material_list)
    
    # Save to compressed numpy archive for instantaneous loading
    np.savez_compressed(
        output_npz_path, 
        spectra=spectra_array, 
        anderson_classes=anderson_array, 
        material_names=material_array
    )
    print(f"Compilation successful! Saved {len(spectra_array)} spectra to {output_npz_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compile the USGS spectral library into a fast array format.")
    parser.add_argument("--usgs_dir", type=str, default=r"C:\satelliteImagery\ground_truth\splib07\ASCIIdata", help="Base directory of unzipped USGS ASCIIdata")
    parser.add_argument("--output_dir", type=str, default=r"C:\satelliteImagery\ground_truth\splib07", help="Output directory for compiled arrays")
    args = parser.parse_args()
    
    landsat_npz = os.path.join(args.output_dir, "usgs_landsat_compiled.npz")
    sentinel_npz = os.path.join(args.output_dir, "usgs_sentinel_compiled.npz")
    
    if os.path.exists(os.path.join(args.usgs_dir, "ASCIIdata_splib07b_rsLandsat8")):
        compile_usgs_library(args.usgs_dir, landsat_npz, "Landsat")
        
    if os.path.exists(os.path.join(args.usgs_dir, "ASCIIdata_splib07b_rsSentinel2")):
        compile_usgs_library(args.usgs_dir, sentinel_npz, "Sentinel")
