import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import sys
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.inference.pipeline import DetectionAndClassificationPipeline
from logger import logger

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="🧠 SKIN_TELLIGENT | Dermatology Assistant",
    layout="wide",
    page_icon="🧠",
)

DEMO_IMAGE_PATH = "data/samples/ROI_detector/detection_result.png"  
# ============================================================
# STYLE <- chatgpt
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
* { font-family: 'Poppins', sans-serif; }
.stApp {
    background: linear-gradient(135deg, rgba(18,32,47,0.95) 0%, rgba(31,58,87,0.95) 100%),
                url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=1920&q=80');
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
}
.main-title {
    font-size: 48px;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #00D9FF 0%, #00ADB5 50%, #0091A0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: #E0E0E0;
    font-size: 18px;
    margin-bottom: 40px;
}
.upload-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0,173,181,0.2);
    border-radius: 20px;
    padding: 30px;
    margin: 20px 0;
}
.result-card {
    background: linear-gradient(135deg, rgba(0,173,181,0.15), rgba(0,145,160,0.15));
    border: 2px solid rgba(0,173,181,0.3);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.section-header {
    color: #00D9FF;
    font-size: 28px;
    font-weight: 600;
    margin: 30px 0 20px 0;
    text-align: center;
}
.footer {
    text-align: center;
    color: #888;
    padding: 20px;
    margin-top: 50px;
    border-top: 1px solid rgba(0,173,181,0.2);
    font-size: 14px;
}
#MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("<p class='main-title'>🧠 SKIN_TELLIGENT</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Dermatology Assistant for Smart Detection & Classification</p>", unsafe_allow_html=True)

# ============================================================
# LOAD PIPELINE
# ============================================================

@st.cache_resource
def load_pipeline():
    logger.info("[Streamlit] Loading DetectionAndClassificationPipeline...")
    return DetectionAndClassificationPipeline(config_path="src/config/pipeline_config.yaml")

pipeline = load_pipeline()

# ============================================================
# IMAGE UPLOAD + DEMO HANDLING 
# ============================================================

if "image_source" not in st.session_state:
    st.session_state.image_source = None
    st.session_state.demo_active = False

st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload a skin image for analysis", type=["jpg", "jpeg", "png"])
demo_button = st.button("🧪 Use Demo Image")
st.markdown("</div>", unsafe_allow_html=True)

# Handle upload
if uploaded_file is not None:
    st.session_state.image_source = Image.open(uploaded_file)
    st.session_state.demo_active = False

# Handle demo button
if demo_button:
    if os.path.exists(DEMO_IMAGE_PATH):
        st.session_state.image_source = Image.open(DEMO_IMAGE_PATH)
        st.session_state.demo_active = True
        st.info("🧪 Using built-in demo image for testing.")
    else:
        st.error(f"Demo image not found at {DEMO_IMAGE_PATH}")

# ============================================================
# DETECTION + CLASSIFICATION
# ============================================================

image_source = st.session_state.image_source

if image_source is not None:
    st.image(image_source, caption="📸 Input Image", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Run Detection + Classification"):
        with st.spinner("🔬 Running AI pipeline... please wait..."):
            try:
                image_cv = cv2.cvtColor(np.array(image_source.convert("RGB")), cv2.COLOR_RGB2BGR)
                det_img, crops, results = pipeline.run_image(image=image_cv, save_dir="output/streamlit_results")

                # Convert visualization for Streamlit display
                det_img_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
                st.success(f"✅ Completed! Found {len(crops)} detected region(s).")

                # Detection visualization
                st.markdown("<h3 class='section-header'>📍 Detection Results</h3>", unsafe_allow_html=True)
                st.image(det_img_rgb, caption="🎯 Detected Regions", use_container_width=True)

                # Classification
                if results:
                    st.markdown("<h3 class='section-header'>🩺 Classification Results</h3>", unsafe_allow_html=True)
                    n_cols = 4
                    for i in range(0, len(results), n_cols):
                        cols = st.columns(n_cols)
                        for j in range(n_cols):
                            idx = i + j
                            if idx < len(results):
                                res = results[idx]
                                roi = crops[idx]
                                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                                with cols[j]:
                                    st.image(roi_rgb, use_container_width=True)
                                    conf = float(res.get("confidence", 0))
                                    name = res.get("class_name", "Unknown")
                                    color = "#00FF88" if conf > 0.8 else "#FFD700" if conf > 0.6 else "#FF6B6B"
                                    st.markdown(
                                        f"<div class='result-card'>"
                                        f"<b>{name}</b><br>"
                                        f"<span style='color:{color};'>Confidence: {conf*100:.2f}%</span>"
                                        f"</div>",
                                        unsafe_allow_html=True
                                    )
                else:
                    st.warning("⚠️ No classification results found.")
            except Exception as e:
                st.error(f"❌ Error during processing: {e}")

# ============================================================
# FOOTER
# ============================================================

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div class='footer'>© 2025 SKIN_TELLIGENT | Developed by <b>Mehraj Alom Tapadar</b><br>"
    "<small>AI-Powered Dermatology Assistant for Early Skin Health Screening</small></div>",
    unsafe_allow_html=True
)
