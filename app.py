# =============================================================
# app.py — SP13: Image Smoothing and Sharpening
# Main Streamlit Web Application
# Author: Gagan Sharma | Roll: 23f3000472
# Project Code: SP13 | IIT Madras BS Electronic Systems
#
# HOW TO RUN LOCALLY:
#   pip install -r requirements.txt
#   streamlit run app.py
# =============================================================

import streamlit as st
import numpy as np
import io
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

# Local modules
from filters import (
    apply_mean_filter,
    apply_gaussian_filter,
    apply_laplacian_filter,
    sharpen_image,
    compute_fft_magnitude
)
from utils import (
    load_image_as_grayscale,
    array_to_pil,
    create_comparison_figure,
    create_fft_figure,
    figure_to_bytes,
    pil_to_bytes,
    get_image_stats
)

# =============================================================
# MUST BE FIRST STREAMLIT COMMAND — crashes if anything is before it
# =============================================================
st.set_page_config(
    page_title="SP13 — Image Smoothing & Sharpening",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# CUSTOM CSS — clean, professional look
# =============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 1rem;
        color: #555;
        margin-top: 0px;
        margin-bottom: 20px;
    }
    .metric-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        border-left: 4px solid #4361ee;
    }
    .filter-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #3a0ca3;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================
# HEADER
# =============================================================
st.markdown('<p class="main-title">🔬 SP13 — Image Smoothing & Sharpening</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">IIT Madras BS Electronic Systems | Signal Processing Project | Gagan Sharma (23f3000472)</p>', unsafe_allow_html=True)
st.markdown("Explore Mean, Gaussian, and Laplacian spatial filters interactively. Upload any image to begin.")
st.markdown("---")

# =============================================================
# SIDEBAR — ALL USER CONTROLS
# =============================================================
with st.sidebar:
    st.header("⚙️ Filter Controls")
    st.markdown("Adjust parameters and results update instantly.")

    st.subheader("📦 Mean Filter")
    mean_k1 = st.selectbox("Kernel Size 1", options=[3, 5, 7, 9], index=0)
    mean_k2 = st.selectbox("Kernel Size 2", options=[3, 5, 7, 9], index=2)

    st.subheader("🔔 Gaussian Filter")
    gauss_sigma1 = st.slider("Sigma 1", min_value=0.5, max_value=3.0, step=0.5, value=1.0)
    gauss_sigma2 = st.slider("Sigma 2", min_value=0.5, max_value=3.0, step=0.5, value=2.0)

    st.subheader("✏️ Sharpening")
    sharpen_str = st.slider("Strength", min_value=0.1, max_value=3.0, step=0.1, value=1.0)

    st.markdown("---")
    st.subheader("📋 Project Info")
    st.info("""
    **Project:** SP13  
    **Topic:** Image Smoothing & Sharpening  
    **Author:** Gagan Sharma  
    **Roll No:** 23f3000472  
    **Institute:** IIT Madras
    """)

# =============================================================
# IMAGE UPLOAD
# =============================================================
uploaded_file = st.file_uploader(
    "📁 Upload a JPEG or PNG image",
    type=["jpg", "jpeg", "png"],
    help="Upload any photo. It will be converted to grayscale automatically."
)

if uploaded_file is None:
    # Show a helpful placeholder when no image is uploaded
    st.info("👆 Upload an image above to start the signal processing pipeline.")
    
    st.markdown("### What this app does:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🟦 Mean Filter**  
        Replaces each pixel with the average of its neighbors.  
        Larger kernel = stronger blur.
        """)
    with col2:
        st.markdown("""
        **🔔 Gaussian Filter**  
        Weighted average — closer pixels matter more.  
        Smoother result than Mean filter.
        """)
    with col3:
        st.markdown("""
        **⚡ Laplacian Filter**  
        High-pass filter that detects edges.  
        Used for sharpening by adding back to original.
        """)
    st.stop()

# =============================================================
# LOAD IMAGE
# =============================================================
pil_image, image_array = load_image_as_grayscale(uploaded_file)
H, W = image_array.shape

# Show image info bar
info_col1, info_col2, info_col3, info_col4 = st.columns(4)
info_col1.metric("Image Width", f"{W} px")
info_col2.metric("Image Height", f"{H} px")
info_col3.metric("Total Pixels", f"{W*H:,}")
info_col4.metric("Color Mode", "Grayscale")

st.markdown("---")

# =============================================================
# APPLY ALL FILTERS (with spinner so user knows it's working)
# =============================================================
with st.spinner("⏳ Applying spatial filters..."):
    mean_res_1   = apply_mean_filter(image_array, kernel_size=mean_k1)
    mean_res_2   = apply_mean_filter(image_array, kernel_size=mean_k2)
    gauss_res_1  = apply_gaussian_filter(image_array, sigma=gauss_sigma1)
    gauss_res_2  = apply_gaussian_filter(image_array, sigma=gauss_sigma2)
    laplacian_res = apply_laplacian_filter(image_array)
    sharpened_res = sharpen_image(image_array, strength=sharpen_str)

    # For comparison grid (fixed parameters matching mid-term report)
    grid_results = {
        "Mean 3×3":      apply_mean_filter(image_array, 3),
        "Mean 7×7":      apply_mean_filter(image_array, 7),
        "Gaussian σ=1":  apply_gaussian_filter(image_array, 1.0),
        "Gaussian σ=2":  apply_gaussian_filter(image_array, 2.0),
        "Laplacian":     laplacian_res,
    }

    # For interactive tabs (uses sidebar slider values)
    interactive_results = {
        f"Mean {mean_k1}×{mean_k1}":    mean_res_1,
        f"Mean {mean_k2}×{mean_k2}":    mean_res_2,
        f"Gaussian σ={gauss_sigma1}":   gauss_res_1,
        f"Gaussian σ={gauss_sigma2}":   gauss_res_2,
        "Laplacian":                     laplacian_res,
        f"Sharpened ({sharpen_str}×)":  sharpened_res,
    }

# =============================================================
# TABS — Main Content Area
# =============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🟦 Mean Filter",
    "🔔 Gaussian Filter",
    "⚡ Laplacian & Sharpening",
    "📊 Full Comparison Grid",
    "📈 Frequency Domain (FFT)",
    "📐 Quantitative Metrics"
])

# ----------------------------------------------------------
# TAB 1: MEAN FILTER
# ----------------------------------------------------------
with tab1:
    st.subheader("Mean (Box) Filter — Uniform Low-Pass Smoothing")
    st.markdown("""
    The Mean filter replaces each pixel with the **average** of all pixels in its neighborhood.
    A 3×3 kernel averages 9 pixels. A 7×7 kernel averages 49 pixels — causing much stronger blur.
    The larger the kernel, the more high-frequency detail (edges, textures) is destroyed.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(pil_image, caption="Original Grayscale", use_column_width=True)
        st.caption(f"Size: {W}×{H}")
    with col2:
        st.image(array_to_pil(mean_res_1), caption=f"Mean {mean_k1}×{mean_k1}", use_column_width=True)
        st.caption(f"Averaging over {mean_k1*mean_k1} pixels per output pixel")
    with col3:
        st.image(array_to_pil(mean_res_2), caption=f"Mean {mean_k2}×{mean_k2}", use_column_width=True)
        st.caption(f"Averaging over {mean_k2*mean_k2} pixels per output pixel — stronger blur")

    st.markdown("**Observation:** The larger kernel removes more fine detail and produces a more blurred, 'smeared' look. Edges become soft and textures disappear.")

    # Download buttons for this tab
    st.markdown("#### Download")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            f"⬇️ Download Mean {mean_k1}×{mean_k1}",
            data=pil_to_bytes(array_to_pil(mean_res_1)),
            file_name=f"mean_{mean_k1}x{mean_k1}.png",
            mime="image/png"
        )
    with dl2:
        st.download_button(
            f"⬇️ Download Mean {mean_k2}×{mean_k2}",
            data=pil_to_bytes(array_to_pil(mean_res_2)),
            file_name=f"mean_{mean_k2}x{mean_k2}.png",
            mime="image/png"
        )

# ----------------------------------------------------------
# TAB 2: GAUSSIAN FILTER
# ----------------------------------------------------------
with tab2:
    st.subheader("Gaussian Filter — Weighted Low-Pass Smoothing")
    st.markdown("""
    The Gaussian filter uses a **bell-curve weighting** — pixels closer to the center contribute more.
    This produces a smoother, more natural-looking blur compared to the blocky Mean filter.
    Higher σ (sigma) = wider bell curve = more blurring.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(pil_image, caption="Original Grayscale", use_column_width=True)
    with col2:
        st.image(array_to_pil(gauss_res_1), caption=f"Gaussian σ={gauss_sigma1}", use_column_width=True)
        st.caption(f"Mild blur — similar strength to Mean {3 if gauss_sigma1 <= 1 else 5}×{3 if gauss_sigma1 <= 1 else 5}")
    with col3:
        st.image(array_to_pil(gauss_res_2), caption=f"Gaussian σ={gauss_sigma2}", use_column_width=True)
        st.caption("Stronger blur — but still softer edges than the equivalent Mean filter")

    st.markdown("**Observation:** The Gaussian result looks 'softer' than the Mean filter at equivalent blur strength because the weighted kernel avoids the harsh rectangular cutoff.")

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            f"⬇️ Download Gaussian σ={gauss_sigma1}",
            data=pil_to_bytes(array_to_pil(gauss_res_1)),
            file_name=f"gaussian_sigma{gauss_sigma1}.png",
            mime="image/png"
        )
    with dl2:
        st.download_button(
            f"⬇️ Download Gaussian σ={gauss_sigma2}",
            data=pil_to_bytes(array_to_pil(gauss_res_2)),
            file_name=f"gaussian_sigma{gauss_sigma2}.png",
            mime="image/png"
        )

# ----------------------------------------------------------
# TAB 3: LAPLACIAN + SHARPENING
# ----------------------------------------------------------
with tab3:
    st.subheader("Laplacian Filter — High-Pass Edge Detection & Sharpening")
    st.markdown("""
    The Laplacian is a **high-pass filter** — the exact opposite of smoothing filters.
    It detects pixels that differ sharply from their neighbors (edges).
    Flat regions give near-zero response (dark). Edge regions give large response (bright).
    
    **Sharpening** works by adding this edge response back to the original:
    `sharpened = original + strength × laplacian`
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(pil_image, caption="Original Grayscale", use_column_width=True)
    with col2:
        st.image(array_to_pil(laplacian_res), caption="Laplacian Response (Edge Map)", use_column_width=True)
        st.caption("Dark = flat region. Bright = edge. vmin/vmax forced to show detail.")
    with col3:
        st.image(array_to_pil(sharpened_res), caption=f"Sharpened Image (Strength: {sharpen_str}×)", use_column_width=True)
        st.caption("Original + Laplacian × strength. Edges are enhanced.")

    st.markdown(f"**Observation:** At strength={sharpen_str}, edges and fine details are noticeably enhanced. Very high strength values introduce 'halos' around edges — a known artifact of Laplacian sharpening.")

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button("⬇️ Download Laplacian", data=pil_to_bytes(array_to_pil(laplacian_res)), file_name="laplacian.png", mime="image/png")
    with dl2:
        st.download_button(f"⬇️ Download Sharpened ({sharpen_str}×)", data=pil_to_bytes(array_to_pil(sharpened_res)), file_name="sharpened.png", mime="image/png")

# ----------------------------------------------------------
# TAB 4: FULL COMPARISON GRID
# ----------------------------------------------------------
with tab4:
    st.subheader("Full Comparison Grid — All Filters Side by Side")
    st.markdown("This is the exact 2×3 figure from the mid-term report. Parameters are fixed (3×3, 7×7, σ=1, σ=2) to match the report format.")

    fig_comparison = create_comparison_figure(image_array, grid_results)
    st.pyplot(fig_comparison, use_container_width=True)

    st.download_button(
        label="⬇️ Download Full Comparison Grid (PNG, 150 DPI)",
        data=figure_to_bytes(fig_comparison),
        file_name="sp13_comparison_grid.png",
        mime="image/png"
    )

    st.markdown("---")
    st.markdown("#### Download Individual Filter Outputs")
    cols = st.columns(3)
    for i, (name, arr) in enumerate(interactive_results.items()):
        with cols[i % 3]:
            st.download_button(
                label=f"⬇️ {name}",
                data=pil_to_bytes(array_to_pil(arr)),
                file_name=f"{name.replace(' ', '_').replace('×','x').replace('=','_').replace('/','_')}.png",
                mime="image/png",
                key=f"dl_{i}"
            )

# ----------------------------------------------------------
# TAB 5: FFT / FREQUENCY DOMAIN
# ----------------------------------------------------------
with tab5:
    st.subheader("Frequency Domain Analysis — 2D FFT Magnitude Spectra")
    st.markdown("""
    This shows WHY each filter behaves the way it does.
    The **center** of each FFT plot = low frequencies (broad shapes, gradual brightness changes).
    The **outer edges** = high frequencies (fine textures, noise, sharp edges).
    
    - **Mean/Gaussian filters** → suppress outer edges (they remove high frequencies = blur)
    - **Laplacian filter** → suppresses the center (it removes low frequencies = edge detection)
    """)

    fig_fft = create_fft_figure(image_array, grid_results)
    st.pyplot(fig_fft, use_container_width=True)

    st.download_button(
        label="⬇️ Download FFT Comparison Figure",
        data=figure_to_bytes(fig_fft),
        file_name="sp13_fft_spectra.png",
        mime="image/png"
    )

# ----------------------------------------------------------
# TAB 6: QUANTITATIVE METRICS
# ----------------------------------------------------------
with tab6:
    st.subheader("Quantitative Metrics — PSNR and MSE")
    st.markdown("""
    These metrics measure how much each filter changes the image compared to the original.
    
    - **MSE (Mean Squared Error):** Average squared difference per pixel. Higher = more change.
    - **PSNR (Peak Signal-to-Noise Ratio):** In dB. Higher = less distortion from original. 
      Values above 30 dB are generally considered good quality.
    
    *Note: These compare filtered vs original, not filtered vs a "clean" ground truth. 
    They show filter impact, not quality per se.*
    """)

    st.markdown("---")

    # Compute metrics for all filters
    all_filter_results = {
        f"Mean {mean_k1}×{mean_k1}":    mean_res_1,
        f"Mean {mean_k2}×{mean_k2}":    mean_res_2,
        f"Gaussian σ={gauss_sigma1}":   gauss_res_1,
        f"Gaussian σ={gauss_sigma2}":   gauss_res_2,
        "Laplacian":                     laplacian_res,
        f"Sharpened ({sharpen_str}×)":  sharpened_res,
    }

    metric_data = []
    for name, arr in all_filter_results.items():
        mse = mean_squared_error(image_array, arr)
        psnr = peak_signal_noise_ratio(image_array, arr, data_range=255)
        metric_data.append((name, mse, psnr))

    # Display as a styled table
    st.markdown("#### Results Table")
    header_cols = st.columns([3, 2, 2, 3])
    header_cols[0].markdown("**Filter**")
    header_cols[1].markdown("**MSE**")
    header_cols[2].markdown("**PSNR (dB)**")
    header_cols[3].markdown("**Interpretation**")

    for name, mse, psnr in metric_data:
        row = st.columns([3, 2, 2, 3])
        row[0].write(name)
        row[1].write(f"{mse:.2f}")
        row[2].write(f"{psnr:.2f} dB")
        if "Laplacian" in name:
            row[3].write("High-pass: large deviation from original expected")
        elif "Sharpen" in name:
            row[3].write("Enhanced edges — moderate deviation")
        elif mse < 20:
            row[3].write("Mild filtering — image largely preserved")
        else:
            row[3].write("Strong filtering — significant smoothing applied")

    st.markdown("---")

    # Visual bar chart of MSE values
    st.markdown("#### MSE Bar Chart")
    import matplotlib.pyplot as plt
    fig_metrics, ax = plt.subplots(figsize=(10, 4))
    names = [m[0] for m in metric_data]
    mses = [m[1] for m in metric_data]
    bars = ax.bar(names, mses, color=['#4361ee', '#3f37c9', '#7209b7', '#560bad', '#f72585', '#b5179e'])
    ax.set_xlabel("Filter", fontsize=11)
    ax.set_ylabel("MSE (vs Original)", fontsize=11)
    ax.set_title("Filter Impact — Mean Squared Error", fontsize=13, fontweight='bold')
    plt.xticks(rotation=20, ha='right')
    for bar, val in zip(bars, mses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_metrics, use_container_width=True)

    st.download_button(
        label="⬇️ Download Metrics Chart",
        data=figure_to_bytes(fig_metrics),
        file_name="sp13_metrics_chart.png",
        mime="image/png"
    )

# =============================================================
# FOOTER
# =============================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#888; font-size:0.85rem;'>
SP13 — Image Smoothing and Sharpening | Gagan Sharma (23f3000472) | IIT Madras BS Electronic Systems<br>
Built with Python · Streamlit · SciPy · NumPy · Pillow · Matplotlib
</div>
""", unsafe_allow_html=True)
