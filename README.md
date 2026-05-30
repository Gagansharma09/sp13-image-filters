# SP13 — Image Smoothing and Sharpening
**IIT Madras BS Electronic Systems | Signal Processing Project**  
**Author:** Gagan Sharma | **Roll:** 23f3000472 | **Project Code:** SP13

---

## What This App Does

An interactive web application that applies three classical spatial filters to any uploaded grayscale image:

| Filter | Type | Effect |
|--------|------|--------|
| Mean 3×3 | Low-pass | Mild uniform blur |
| Mean 7×7 | Low-pass | Strong uniform blur |
| Gaussian σ=1 | Low-pass | Mild weighted blur (smooth) |
| Gaussian σ=2 | Low-pass | Strong weighted blur (smooth) |
| Laplacian 3×3 | High-pass | Edge detection |
| Sharpened | High-pass | Original + Laplacian edges |

---

## Project Structure

```
sp13_project/
├── app.py              ← Main Streamlit web app (run this)
├── filters.py          ← All signal processing logic
├── utils.py            ← Image loading, conversion, plotting helpers
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## How to Run Locally

### Step 1: Install Python
Make sure Python 3.8 or higher is installed.  
Download from: https://www.python.org/downloads/

### Step 2: Install Dependencies
Open terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 3: Run the App
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## How i deployed  Deploy on Streamlit Community Cloud (FREE)

### Step 1: Push to GitHub
1. Create a GitHub account at https://github.com
2. Create a new repository called `sp13-image-filters`
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
- After ~2-3 minutes, your app will be live at:
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
