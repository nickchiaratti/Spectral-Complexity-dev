import os
import requests
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- Configuration ---
# This is the directory where the downloaded GeoTIFF files will be saved.
DOWNLOAD_DIR = "C:/dragonette/"

# Sample Dragonette-1 data from opendata.wyvern.space.
SAMPLE_FILE = {
    "url": "https://wyvern-prod-public-open-data-program.s3.ca-central-1.amazonaws.com/wyvern_dragonette-002_20250526T200831_1e1ce2d7/wyvern_dragonette-002_20250526T200831_1e1ce2d7.tiff",
    "filename": "wyvern_dragonette-002_20250526T200831_1e1ce2d7.tiff"
}

# Default bands to display initially. These are chosen to approximate a
# natural color view. We use 1-based indexing for user display.
INITIAL_BANDS = {
    'red': 5,
    'green': 8,
    'blue': 12,
}

def create_download_dir():
    """Create the directory for storing downloads if it doesn't exist."""
    if not os.path.exists(DOWNLOAD_DIR):
        print(f"Creating directory: {DOWNLOAD_DIR}")
        os.makedirs(DOWNLOAD_DIR)

def download_file(url, filename, download_directory):
    """Downloads a large file with a progress bar."""
    filepath = os.path.join(download_directory, filename)
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return filepath

    print(f"Downloading {filename}...")
    print("This is a large file and may take several minutes.")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            bytes_downloaded = 0
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    done = int(50 * bytes_downloaded / total_size)
                    print(f"\r[{'=' * done}{' ' * (50-done)}] {bytes_downloaded/1e6:.2f} / {total_size/1e6:.2f} MB", end='')
        print(f"\nSuccessfully downloaded to {filepath}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading file: {e}")
        return None

def create_interactive_viewer(filepath):
    """
    Loads a hyperspectral image and displays it with sliders to change bands.
    """
    try:
        print("Opening GeoTIFF file. This may take a moment...")
        with rasterio.open(filepath) as src:
            # Load all bands into memory
            # For very large files, this could be memory-intensive.
            hyper_cube = src.read()
            num_bands, height, width = hyper_cube.shape
            print(f"Image loaded: {width}x{height} pixels with {num_bands} bands.")
    except Exception as e:
        print(f"Failed to read GeoTIFF file: {e}")
        return

    # --- Create the plot and image display ---
    # We leave some space at the bottom for the sliders
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.subplots_adjust(bottom=0.25)
    ax.set_title(f'Hyperspectral Viewer: {os.path.basename(filepath)}')
    ax.axis('off')

    # This will hold the displayed image data
    image_display = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))

    # --- Create slider axes ---
    ax_red = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_green = plt.axes([0.25, 0.05, 0.65, 0.03])
    ax_blue = plt.axes([0.25, 0.0, 0.65, 0.03])

    # --- Create Slider widgets ---
    # Note: Band numbers are 1-based for user-friendliness
    slider_red = Slider(
        ax=ax_red, label='Red Band', valmin=1, valmax=num_bands,
        valinit=INITIAL_BANDS['red'], valstep=1, color='red'
    )
    slider_green = Slider(
        ax=ax_green, label='Green Band', valmin=1, valmax=num_bands,
        valinit=INITIAL_BANDS['green'], valstep=1, color='green'
    )
    slider_blue = Slider(
        ax=ax_blue, label='Blue Band', valmin=1, valmax=num_bands,
        valinit=INITIAL_BANDS['blue'], valstep=1, color='blue'
    )

    def update(val):
        """Function to be called when a slider value is changed."""
        # Get the current band numbers from the sliders (and convert to 0-based index)
        r_band_idx = int(slider_red.val) - 1
        g_band_idx = int(slider_green.val) - 1
        b_band_idx = int(slider_blue.val) - 1

        # Select the bands from the data cube
        r = hyper_cube[r_band_idx, :, :]
        g = hyper_cube[g_band_idx, :, :]
        b = hyper_cube[b_band_idx, :, :]

        # Normalize each band to the 0-255 range for display based on data type.
        if np.issubdtype(hyper_cube.dtype, np.integer):
            # For integers, scale based on the data type's maximum possible value.
            dtype_info = np.iinfo(hyper_cube.dtype)
            max_val = dtype_info.max
            r_norm = (r / max_val * 255).astype(np.uint8)
            g_norm = (g / max_val * 255).astype(np.uint8)
            b_norm = (b / max_val * 255).astype(np.uint8)
        else:
            # Fallback for other data types - scale by the max value in the entire cube
            print(f"Warning: Unsupported data type {hyper_cube.dtype}. Attempting to scale by max value.")
            max_val = hyper_cube.max()
            if max_val == 0: max_val = 1 # Avoid division by zero
            r_norm = (r / max_val * 255).astype(np.uint8)
            g_norm = (g / max_val * 255).astype(np.uint8)
            b_norm = (b / max_val * 255).astype(np.uint8)

        rgb_image = np.dstack((r_norm, g_norm, b_norm))

        # Update the image data and redraw the plot
        image_display.set_data(rgb_image)
        fig.canvas.draw_idle()

    # Connect the 'on_changed' event of each slider to the update function
    slider_red.on_changed(update)
    slider_green.on_changed(update)
    slider_blue.on_changed(update)

    # Call update once to initialize the first view
    update(None)

    plt.show()

def main():
    """Main function to run the application."""
    print("--- Interactive Hyperspectral Viewer ---")
    create_download_dir()
    
    # Check for the file and prompt for download if it's missing.
    filepath = os.path.join(DOWNLOAD_DIR, SAMPLE_FILE['filename'])
    if not os.path.exists(filepath):
        choice = input("Sample Dragonette file not found. Download now? (y/n): ").lower()
        if choice == 'y':
            filepath = download_file(SAMPLE_FILE['url'], SAMPLE_FILE['filename'], DOWNLOAD_DIR)
        else:
            print("Cannot start viewer without data file. Exiting.")
            return
    
    if filepath:
        create_interactive_viewer(filepath)

if __name__ == "__main__":
    main()
