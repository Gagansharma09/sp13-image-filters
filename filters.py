# =============================================================
# filters.py — SP13: Image Smoothing and Sharpening
# All spatial filtering logic lives here
# Author: Gagan Sharma | Roll: 23f3000472
# =============================================================

import numpy as np
from scipy.ndimage import convolve, gaussian_filter


def apply_mean_filter(image_array, kernel_size=3):
    """
    Apply a Mean (Box) filter — uniform low-pass smoothing.

    Parameters:
        image_array (np.ndarray): 2D float64 grayscale image array
        kernel_size (int): Size of the square kernel (e.g., 3 → 3x3, 7 → 7x7)
                           Must be an odd integer.

    Returns:
        np.ndarray: Filtered 2D float64 array

    How it works:
        - Kernel = matrix of (1 / kernel_size²) values
        - Every pixel becomes the average of its kernel_size×kernel_size neighborhood
        - Larger kernel = more pixels averaged = stronger blur
        - mode='reflect' mirrors border pixels to avoid dark edge artifacts
    """
    total_pixels = kernel_size * kernel_size
    kernel = np.ones((kernel_size, kernel_size)) / total_pixels
    return convolve(image_array, kernel, mode='reflect')


def apply_gaussian_filter(image_array, sigma=1.0):
    """
    Apply a Gaussian filter — weighted low-pass smoothing.

    Parameters:
        image_array (np.ndarray): 2D float64 grayscale image array
        sigma (float): Standard deviation of Gaussian bell curve
                       sigma=0.5 → very mild blur
                       sigma=1.0 → mild blur (≈ Mean 3x3)
                       sigma=2.0 → strong blur (≈ Mean 7x7)
                       sigma=3.0 → very strong blur

    Returns:
        np.ndarray: Filtered 2D float64 array

    How it works:
        - Pixels closer to center get HIGHER weights (unlike Mean filter)
        - Formula: G(x,y) = (1/2πσ²) * exp(-(x²+y²)/2σ²)
        - SciPy auto-generates the kernel size based on sigma
        - Result looks "softer" than Mean — no blocky artifacts
    """
    return gaussian_filter(image_array, sigma=sigma, mode='reflect')


def apply_laplacian_filter(image_array):
    """
    Apply a Laplacian filter — high-pass edge detection.

    Parameters:
        image_array (np.ndarray): 2D float64 grayscale image array

    Returns:
        np.ndarray: Edge-detected 2D float64 array (clipped to 0–255)

    How it works:
        - 4-connected kernel:  [[ 0,-1, 0],
                                 [-1, 4,-1],
                                 [ 0,-1, 0]]
        - At flat regions: center ≈ neighbors → output ≈ 0 (dark)
        - At edges: center ≠ neighbors → large output (bright)
        - np.clip(0,255) removes negatives that would appear black
        - IMPORTANT: When displaying with matplotlib, always use vmin=0, vmax=255
    """
    laplacian_kernel = np.array([
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
    ], dtype=np.float64)

    filtered = convolve(image_array, laplacian_kernel, mode='reflect')
    return np.clip(filtered, 0, 255)


def sharpen_image(image_array, strength=1.0):
    """
    Sharpen image by adding Laplacian edges back to the original.

    Parameters:
        image_array (np.ndarray): 2D float64 grayscale image array
        strength (float): How aggressively to sharpen
                          0.5 → subtle enhancement
                          1.0 → standard sharpening
                          2.0 → aggressive, may introduce halos
                          3.0 → extreme, mostly for demonstration

    Returns:
        np.ndarray: Sharpened 2D float64 array (clipped to 0–255)

    Formula:
        sharpened = original + strength × laplacian_response
        - This is the "unsharp masking" technique used in photo editors
        - Laplacian detects WHERE edges are, we amplify those spots
    """
    laplacian_response = apply_laplacian_filter(image_array)
    sharpened = image_array + (strength * laplacian_response)
    return np.clip(sharpened, 0, 255)


def compute_fft_magnitude(image_array):
    """
    Compute 2D FFT magnitude spectrum (log-scaled) for frequency analysis.

    Parameters:
        image_array (np.ndarray): 2D float64 grayscale image array

    Returns:
        np.ndarray: Log-scaled magnitude spectrum, same shape as input

    How it works:
        - np.fft.fft2 → 2D Discrete Fourier Transform
        - np.fft.fftshift → moves DC (zero-frequency) component to center
        - np.abs → magnitude of complex numbers
        - np.log1p (= log(1+x)) → compresses huge range for visibility
        - Center of output = low frequencies (large shapes, gradual changes)
        - Edges of output = high frequencies (fine textures, noise, edges)
    """
    fft = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)
    return log_magnitude
