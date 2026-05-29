# =============================================================
# utils.py — SP13: Image Smoothing and Sharpening
# Image loading, conversion, plotting, and export helpers
# Author: Gagan Sharma | Roll: 23f3000472
# =============================================================

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io


def load_image_as_grayscale(uploaded_file):
    """
    Load a Streamlit uploaded file and convert to grayscale NumPy array.

    Parameters:
        uploaded_file: Streamlit UploadedFile object (from st.file_uploader)

    Returns:
        pil_image (PIL.Image): Grayscale PIL image (mode "L")
        image_array (np.ndarray): 2D float64 array, shape=(H, W), values 0.0–255.0

    Why float64?
        Integer arrays cause overflow during convolution math.
        e.g., 200 + 100 = 300 which clips to 255 in uint8 — wrong result.
        float64 lets values go above 255 temporarily during filtering,
        which we then clip back correctly at the end.
    """
    pil_image = Image.open(uploaded_file).convert("L")
    image_array = np.array(pil_image, dtype=np.float64)
    return pil_image, image_array


def array_to_pil(image_array):
    """
    Convert a filtered NumPy float64 array back to a displayable PIL Image.

    Parameters:
        image_array (np.ndarray): 2D float64 array (values may be outside 0–255)

    Returns:
        PIL.Image: Grayscale image ready for st.image() display

    Steps:
        1. Clip: force all values into [0, 255] range
        2. Cast to uint8: Pillow only accepts 8-bit unsigned integers
        3. Wrap in PIL.Image.fromarray
    """
    clipped = np.clip(image_array, 0, 255).astype(np.uint8)
    return Image.fromarray(clipped, mode="L")


def create_comparison_figure(original, results_dict):
    """
    Generate the classic 2×3 Matplotlib grid comparing all filter outputs.
    Matches the figure in the mid-term report exactly.

    Parameters:
        original (np.ndarray): 2D float64 original grayscale image
        results_dict (dict): Keys = filter names (str), Values = 2D float64 arrays
                             Expected 5 entries for 2×3 layout (original + 5)

    Returns:
        matplotlib.figure.Figure: The complete comparison figure

    Layout:
        Row 1: Original | Mean 3×3   | Mean 7×7
        Row 2: Gauss σ=1| Gauss σ=2  | Laplacian
    """
    fig = plt.figure(figsize=(14, 9), facecolor='white')
    all_images = [("Original", original)] + list(results_dict.items())

    for idx, (title, arr) in enumerate(all_images):
        ax = fig.add_subplot(2, 3, idx + 1)

        # Laplacian fix: force full intensity range so edges are visible
        if "Laplacian" in title:
            ax.imshow(arr, cmap='gray', vmin=0, vmax=255)
        else:
            ax.imshow(arr, cmap='gray')

        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        ax.axis('off')

    plt.suptitle("SP13 — Spatial Filter Comparison", fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def create_fft_figure(original, results_dict):
    """
    Generate a figure showing FFT magnitude spectra for each filter output.
    Useful for understanding which frequencies each filter preserves or kills.

    Parameters:
        original (np.ndarray): 2D float64 original grayscale image
        results_dict (dict): Same format as create_comparison_figure

    Returns:
        matplotlib.figure.Figure

    Reading the FFT plot:
        - CENTER = low frequencies (background, large shapes)
        - EDGES = high frequencies (fine textures, noise, edges)
        - Smoothing filters kill the bright spots near edges (they remove HF)
        - Laplacian kills the center (it removes LF — the opposite)
    """
    from filters import compute_fft_magnitude

    fig = plt.figure(figsize=(14, 9), facecolor='white')
    all_images = [("Original", original)] + list(results_dict.items())

    for idx, (title, arr) in enumerate(all_images):
        ax = fig.add_subplot(2, 3, idx + 1)
        fft_mag = compute_fft_magnitude(arr)
        ax.imshow(fft_mag, cmap='inferno')
        ax.set_title(f"FFT: {title}", fontsize=11, fontweight='bold', pad=8)
        ax.axis('off')

    plt.suptitle("SP13 — Frequency Domain (2D FFT Magnitude Spectra)", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def figure_to_bytes(fig):
    """
    Convert a Matplotlib figure to PNG bytes for Streamlit download button.

    Parameters:
        fig (matplotlib.figure.Figure): Any matplotlib figure

    Returns:
        bytes: PNG image data ready for st.download_button(data=...)
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    return buffer.getvalue()


def pil_to_bytes(pil_image):
    """
    Convert a PIL Image to PNG bytes for Streamlit download button.

    Parameters:
        pil_image (PIL.Image): Any PIL image

    Returns:
        bytes: PNG image data
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.getvalue()


def get_image_stats(image_array):
    """
    Compute basic statistics about an image array.

    Parameters:
        image_array (np.ndarray): 2D float64 image array

    Returns:
        dict with keys: min, max, mean, std, shape
    """
    return {
        "min": float(np.min(image_array)),
        "max": float(np.max(image_array)),
        "mean": float(np.mean(image_array)),
        "std": float(np.std(image_array)),
        "shape": image_array.shape
    }
