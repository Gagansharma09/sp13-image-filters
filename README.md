# SP13 — Web App for Image Smoothing and Sharpening

**Author:** Gagan Sharma | Roll No: 23f3000472  
**Institute:** IIT Madras — BS Electronic Systems  
**Course:** Signal Processing Project (SP13)  
**Instructor:** Vishal  
**Submission:** 2026

---

## What I Built

An interactive web application for spatial image filtering, built with Python and deployed on Streamlit Cloud. The app lets anyone upload a photo and instantly see what happens when you apply classical image processing filters — Mean, Gaussian, and Laplacian — with adjustable parameters, all in real time inside a browser.

The project started as a Jupyter Notebook and was extended into a fully deployed web app with 6 tabs, interactive sliders, PSNR/MSE metrics, 2D FFT frequency analysis, and download buttons for every output.

**Live App:** [https://sp13-image-filters.streamlit.app](https://sp13-image-filters.streamlit.app)  
**Demo Video:** [Google Drive](https://drive.google.com/file/d/1kpxxkY1Ug3Fsf_Aj-UGr8vy7ZEXlotaF/view?usp=sharing)  
**Source Code:** [GitHub](https://github.com/Gagansharma09/sp13-image-filters/tree/main)

---

## App Preview

The app has a sidebar with filter controls and a tabbed main area:

| Tab | What it shows |
|-----|--------------|
| Mean Filter | Original vs Mean 3×3 vs Mean 7×7 |
| Gaussian Filter | Original vs Gaussian σ=1 vs Gaussian σ=2 |
| Laplacian & Sharpening | Edge map + sharpened output |
| Full Comparison Grid | Classic 2×3 grid with all 5 filters |
| Frequency Domain (FFT) | 2D FFT magnitude spectra for all outputs |
| Quantitative Metrics | PSNR and MSE table + bar chart |

---

## Filters Implemented

### Mean Filter
Replaces each pixel with the uniform average of its neighborhood. Built manually using NumPy — a matrix of equal weights that sum to 1.

```
h = (1/9) × [[1,1,1],[1,1,1],[1,1,1]]   (3×3)
h = (1/49) × 1_{7×7}                    (7×7)
```

The 7×7 kernel averages 49 pixels per output pixel, producing noticeably stronger blur than the 3×3 which averages only 9.

### Gaussian Filter
Weighted averaging using a bell-curve distribution — pixels closer to the center contribute more than distant ones.

```
G(x,y) = (1/2πσ²) · exp(-(x²+y²)/2σ²)
```

SciPy generates the kernel automatically based on sigma. Results look softer and more natural than the Mean filter at equivalent blur strength because there's no abrupt rectangular cutoff.

### Laplacian Filter
A high-pass filter that detects edges by measuring how much each pixel differs from its four direct neighbors.

```
kernel = [[ 0, -1,  0],
          [-1,  4, -1],
          [ 0, -1,  0]]
```

Flat regions give ~0 response (dark). Edges give large response (bright). Sharpening is done by adding this back to the original:

```
sharpened = original + strength × laplacian_response
```

---

## Quantitative Results

These are the actual PSNR and MSE values measured on the test image:

| Filter | MSE | PSNR (dB) |
|--------|-----|-----------|
| Mean 3×3 | 39.79 | 32.13 |
| Mean 7×7 | 78.31 | 29.19 |
| Gaussian σ=1.5 | 54.16 | 30.79 |
| Gaussian σ=2.0 | 69.38 | 29.72 |
| Laplacian | 25419.28 | 4.08 |
| Sharpened (1.6×) | 341.14 | 22.80 |

The Laplacian's high MSE is expected — it's a high-pass filter that produces a fundamentally different output from the original, not a degraded version of it.

---

## FFT Analysis

The 2D FFT magnitude spectra show the frequency-domain explanation for why each filter behaves the way it does:

- **Center of spectrum** = low frequencies (large shapes, gradual brightness changes)
- **Outer edges of spectrum** = high frequencies (fine textures, noise, edges)
- **Mean and Gaussian** → suppress the outer edges (they remove high frequencies → blur)
- **Laplacian** → suppresses the center (removes low frequencies → edge detection)

The FFT grid makes it visually clear why the 7×7 Mean filter creates a stronger checkerboard pattern in the spectrum than the 3×3 — it cuts off a wider band of high frequencies.

---

## Problems I Ran Into

**Hardcoded image path** — The original notebook had the path set to my local machine. Fixed by switching to a relative path, then eliminated entirely by using Streamlit's file uploader.

**Laplacian looked completely black** — Matplotlib's auto-scaling compressed the tiny values into near-black. Fixed by setting `vmin=0, vmax=255` in the imshow call.

**SciPy rejected PIL Image objects** — SciPy needs NumPy arrays, not PIL images. Fixed with `np.array(img, dtype=np.float64)`. The float64 is important — integer arrays clip at 255 during convolution math and corrupt the output.

**Streamlit set_page_config crash** — It has to be the very first Streamlit command in the file. Easy to break when reorganizing code.

**Deployment dependency errors** — Streamlit Cloud needs exact library versions in requirements.txt. Solved by specifying minimum versions for all six packages.

---

## Project Structure

```
sp13-image-filters/
├── app.py          — Main Streamlit web app
├── filters.py      — apply_mean_filter(), apply_gaussian_filter(),
│                     apply_laplacian_filter(), sharpen_image(), compute_fft_magnitude()
├── utils.py        — load_image_as_grayscale(), array_to_pil(),
│                     create_comparison_figure(), create_fft_figure(), figure_to_bytes()
├── requirements.txt
└── README.md
```

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| Streamlit ≥1.32.0 | Web UI, file upload, sliders, tabs, download buttons |
| NumPy ≥1.24.0 | Kernel construction, FFT, array math |
| SciPy ≥1.10.0 | 2D convolution (`ndimage.convolve`), Gaussian filtering |
| Pillow ≥9.0.0 | Image loading and grayscale conversion |
| Matplotlib ≥3.7.0 | Comparison figures and FFT plots |
| scikit-image ≥0.21.0 | PSNR and MSE computation |

---

## Links

| Resource | Link |
|----------|------|
| Live Web App | https://sp13-image-filters.streamlit.app |
| Demo Video | https://drive.google.com/file/d/1kpxxkY1Ug3Fsf_Aj-UGr8vy7ZEXlotaF/view?usp=sharing |
| Source Code | https://github.com/Gagansharma09/sp13-image-filters/tree/main |
| Project Report | Submitted to SEEK portal — IIT Madras SP13 Final Submission |
| Google Drive (code + video) | https://drive.google.com/drive/folders/1lW5Jx4QlTN8UjkspjZIrrw8TbxI1jwTT?usp=sharing |2. Create a new repository called `sp13-image-filters`
3. Upload all 4 files: `app.py`, `filters.py`, `utils.py`, `requirements.txt`

### Step 2: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository:** `https://github.com/Gagansharma09/sp13-image-filters/edit/main/README.md`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**

### Step 3: Get Your Public URL
- app will be live at:
https://sp13-image-filters.streamlit.app/


**Cost: Completely FREE. No credit card needed.**

---

## App Features

- **6 Tabs:** Mean Filter | Gaussian Filter | Laplacian & Sharpening | Full Comparison Grid | Frequency Domain (FFT) | Quantitative Metrics
- **Interactive sliders:** Adjust kernel size, sigma, and sharpening strength in real time
- **Download buttons:** Download any filtered image or comparison figure as PNG
- **PSNR & MSE metrics:** Quantitative evaluation of filter impact
- **FFT Analysis:** See frequency domain explanation of why each filter behaves the way it does
- **Mobile friendly:** Works on phones and tablets

---

## Signal Processing Theory

### Mean Filter
- Kernel: all values = 1/N² (N = kernel size)
- Replaces each pixel with the average of its neighborhood
- Larger kernel = stronger blur (7×7 averages 49 pixels vs 9 for 3×3)

### Gaussian Filter  
- Kernel: weighted by distance using G(x,y) = (1/2πσ²)·exp(-(x²+y²)/2σ²)
- Closer pixels contribute more — smoother result than Mean filter
- Higher σ = wider kernel = more blur

### Laplacian Filter
- Kernel: [[0,-1,0],[-1,4,-1],[0,-1,0]]
- Detects edges by computing difference from 4 neighbors
- Flat regions → ~0 response | Edge regions → large response
- Sharpening: original + strength × laplacian

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.32.0 | Web UI framework |
| numpy | ≥1.24.0 | Kernel construction and array math |
| scipy | ≥1.10.0 | 2D convolution and Gaussian filtering |
| Pillow | ≥9.0.0 | Image loading and format conversion |
| matplotlib | ≥3.7.0 | Comparison figure plotting |
| scikit-image | ≥0.21.0 | PSNR and MSE metrics |
