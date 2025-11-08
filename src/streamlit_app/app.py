import streamlit as st
import requests
from PIL import Image
import os

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/detect"

#
st.set_page_config(
    page_title="🧠 SKIN_TELLIGENT | Dermatology Assistant",
    layout="wide",
    page_icon="chart_with_upwards_trend",
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, rgba(18, 32, 47, 0.95) 0%, rgba(31, 58, 87, 0.95) 100%),
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
        text-shadow: 0 0 30px rgba(0, 173, 181, 0.3);
        letter-spacing: 2px;
    }
    .subtitle {
        text-align: center;
        color: #E0E0E0;
        font-size: 18px;
        margin-bottom: 40px;
        font-weight: 300;
    }
    .upload-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 173, 181, 0.2);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        background: linear-gradient(135deg, #00D9FF 0%, #00ADB5 100%);
        color: white;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 600;
        padding: 12px 40px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 173, 181, 0.4);
        width: 100%;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #00ADB5 0%, #008C99 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 173, 181, 0.6);
    }
    .result-card {
        background: linear-gradient(135deg, rgba(0, 173, 181, 0.15), rgba(0, 145, 160, 0.15));
        backdrop-filter: blur(10px);
        border: 2px solid rgba(0, 173, 181, 0.3);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 173, 181, 0.6);
    }
    .section-header {
        color: #00D9FF;
        font-size: 28px;
        font-weight: 600;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(0, 173, 181, 0.3);
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #888;
        padding: 20px;
        margin-top: 50px;
        border-top: 1px solid rgba(0, 173, 181, 0.2);
        font-size: 14px;
    }
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 173, 181, 0.5), transparent);
        margin: 40px 0;
    }
    #MainMenu, footer, header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<p class='main-title'>🧠 SKIN_TELLIGENT</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Dermatology Assistant for Smart Detection & Classification</p>", unsafe_allow_html=True)

# --- File Upload Section ---
st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload a skin image for analysis", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.image(uploaded_file, caption="📸 Uploaded Image", use_container_width=True)

    with col2:
        image = Image.open(uploaded_file)
        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px; border: 1px solid rgba(0, 173, 181, 0.2);'>
            <p><b>File Name:</b> {uploaded_file.name}</p>
            <p><b>Image Size:</b> {image.size[0]} × {image.size[1]} px</p>
            <p><b>Format:</b> {image.format}</p>
            <p><b>Mode:</b> {image.mode}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Run Detection + Classification"):
        with st.spinner("🔬 Processing image... please wait..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            try:
                response = requests.post(API_URL, files=files)
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to FastAPI server. Please start it with `uvicorn src.api.app:app --reload`.")
                st.stop()

        # --- Handle backend responses ---
        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Detection completed successfully! Found {result['detections']} region(s).")

            # --- Display Detection Image ---
            st.markdown("<p class='section-header'>📍 Detection Results</p>", unsafe_allow_html=True)
            if os.path.exists(result["detection_image"]):
                st.image(result["detection_image"], caption="🎯 Detected Regions", use_container_width=True)
            else:
                st.warning("⚠️ Detection image not found on server.")

            # --- Classification ---
            if result["classification_results"]:
                st.markdown("<p class='section-header'>🩺 Classification Results</p>", unsafe_allow_html=True)
                n_cols = 4
                for i in range(0, len(result["classification_results"]), n_cols):
                    cols = st.columns(n_cols)
                    for j in range(n_cols):
                        idx = i + j
                        if idx < len(result["classification_results"]):
                            res = result["classification_results"][idx]
                            crop_path = result["roi_crops"][idx]
                            roi_img = Image.open(crop_path)
                            with cols[j]:
                                st.image(roi_img, use_container_width=True)
                                conf_color = "#00FF88" if res['confidence'] > 0.8 else "#FFD700" if res['confidence'] > 0.6 else "#FF6B6B"
                                st.markdown(
                                    f"<div class='result-card'><b>{res['class_name']}</b>"
                                    f"<br><span style='color:{conf_color}'>Confidence: {res['confidence']*100:.2f}%</span></div>",
                                    unsafe_allow_html=True
                                )
            else:
                st.warning("⚠️ No classification results found.")
        else:
            # --- Handle backend errors gracefully ---
            try:
                err = response.json()
                error_message = err.get("error", "Unknown error")
            except Exception:
                error_message = response.text
            st.error(f"❌ Error {response.status_code}: {error_message}")
            st.info("💡 Try re-uploading or check the FastAPI logs for details.")

else:
    st.markdown("""
    <div style='text-align: center; padding: 40px; background: rgba(255, 255, 255, 0.05);
                border-radius: 15px; border: 2px dashed rgba(0, 173, 181, 0.3); margin: 20px 0;'>
        <h3 style='color: #00D9FF;'>👆 Get Started</h3>
        <p style='color: #E0E0E0;'>Upload a skin image to begin the AI-powered detection and classification process.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div class='footer'>© 2025 SKIN_TELLIGENT | Developed by <b>Mehraj Alom Tapadar</b><br>"
    "<small>AI-Powered Dermatology Assistant for Early Skin Health Screening</small></div>",
    unsafe_allow_html=True
)
