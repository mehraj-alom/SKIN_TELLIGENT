import streamlit as st
import os
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
from io import BytesIO
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the pipeline directly (no FastAPI)
from src.inference.pipeline import DetectionAndClassificationPipeline

# ========================
# Page config
# ========================
st.set_page_config(
    page_title="🧠 SKIN_TELLIGENT | Dermatology Assistant",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# ========================
# Enhanced CSS Styling with modern gradients, animations, and neon accents
# ========================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * { 
        font-family: 'Poppins', sans-serif;
    }
    
    /* Dark theme with neon accents */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 25%, #1e2d47 50%, #162434 75%, #0d1620 100%);
        background-attachment: fixed;
        min-height: 100vh;
        position: relative;
    }
    
    /* Animated gradient background overlay */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(ellipse at 20% 50%, rgba(0, 217, 255, 0.1) 0%, transparent 50%),
                    radial-gradient(ellipse at 80% 80%, rgba(138, 43, 226, 0.08) 0%, transparent 50%);
        pointer-events: none;
        z-index: 0;
    }
    
    .stApp > * {
        position: relative;
        z-index: 1;
    }
    
    /* Main title with glow effect */
    .main-title {
        font-size: 56px;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #00D9FF 0%, #00ADB5 40%, #00FFF0 70%, #00D9FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 5px;
        text-shadow: 0 0 50px rgba(0, 217, 255, 0.5);
        letter-spacing: 3px;
        animation: glow-pulse 3s ease-in-out infinite;
    }
    
    @keyframes glow-pulse {
        0%, 100% { text-shadow: 0 0 30px rgba(0, 217, 255, 0.4); }
        50% { text-shadow: 0 0 60px rgba(0, 217, 255, 0.8); }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #E0E0E0;
        font-size: 18px;
        margin-bottom: 40px;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    /* Upload card with glassmorphism */
    .upload-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(15px);
        border: 2px solid rgba(0, 217, 255, 0.25);
        border-radius: 25px;
        padding: 35px;
        margin: 25px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(0, 217, 255, 0.5);
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.2),
                    inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    /* Button styling with gradient and hover animation */
    .stButton > button {
        background: linear-gradient(135deg, #00D9FF 0%, #00ADB5 50%, #008C99 100%);
        color: white;
        border-radius: 15px;
        font-size: 18px;
        font-weight: 600;
        padding: 14px 45px;
        border: none;
        transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4),
                    0 0 20px rgba(0, 217, 255, 0.2);
        cursor: pointer;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00FFF0 0%, #00D9FF 50%, #00ADB5 100%);
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 217, 255, 0.6),
                    0 0 30px rgba(0, 217, 255, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Result cards with neon border */
    .result-card {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.12), rgba(0, 145, 160, 0.12));
        backdrop-filter: blur(10px);
        border: 2px solid rgba(0, 217, 255, 0.35);
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px) scale(1.02);
        border-color: rgba(0, 217, 255, 0.6);
        box-shadow: 0 12px 35px rgba(0, 217, 255, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    /* High confidence */
    .conf-high {
        color: #00FF88;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
    }
    
    /* Medium confidence */
    .conf-medium {
        color: #FFD700;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.4);
    }
    
    /* Low confidence */
    .conf-low {
        color: #FF6B6B;
        font-weight: 600;
        text-shadow: 0 0 10px rgba(255, 107, 107, 0.4);
    }
    
    /* Section headers */
    .section-header {
        color: #00D9FF;
        font-size: 32px;
        font-weight: 600;
        margin: 35px 0 25px 0;
        padding-bottom: 15px;
        border-bottom: 3px solid rgba(0, 217, 255, 0.4);
        text-align: center;
        letter-spacing: 1px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        padding: 25px;
        margin-top: 50px;
        border-top: 2px solid rgba(0, 217, 255, 0.2);
        font-size: 14px;
    }
    
    /* Horizontal rule */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 217, 255, 0.5), transparent);
        margin: 45px 0;
    }
    
    /* Hide default Streamlit UI elements */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* Info and error messages */
    .stAlert {
        border-radius: 15px;
        border: 2px solid rgba(0, 217, 255, 0.3);
    }
    
    /* Spinner and loading */
    .stSpinner > div {
        border-color: rgba(0, 217, 255, 0.5);
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #E0E0E0;
        font-weight: 500;
    }
    
    </style>
""", unsafe_allow_html=True)

# ========================
# Initialize session state and pipeline (cached for performance)
# ========================
@st.cache_resource
def load_pipeline():
    """Load the DetectionAndClassificationPipeline once and cache it."""
    st.info("🔄 Initializing AI models... this may take a moment on first load.")
    try:
        pipeline = DetectionAndClassificationPipeline(config_path=Path("src/config/pipeline_config.yaml"))
        return pipeline
    except Exception as e:
        st.error(f"❌ Failed to load pipeline: {e}")
        st.stop()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = load_pipeline()

if "demo_active" not in st.session_state:
    st.session_state.demo_active = False

pipeline = st.session_state.pipeline

# ========================
# Header
# ========================
st.markdown("<p class='main-title'>🧠 SKIN_TELLIGENT</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>AI-Powered Dermatology Assistant for Smart Detection & Classification</p>",
    unsafe_allow_html=True
)

# ========================
# Sidebar: Information and controls
# ========================
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    **SKIN_TELLIGENT** uses deep learning models to:
    - 🎯 Detect skin regions of interest
    - 🩺 Classify detected regions
    - 🧭 Provide explainability via Grad-CAM++
    
    Developed for early dermatology screening.
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_explain = st.checkbox("🔍 Show Feature Attribution Analysis", value=False)
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Detector:** YOLO-based (ONNX)
    - **Classifier:** PyTorch ResNet
    - **Attribution:** Feature Importance Mapping
    """)

# ========================
# Upload / Demo Section
# ========================
st.markdown("<div class='upload-card'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    uploaded_file = st.file_uploader("📤 Upload a skin image", type=["jpg", "jpeg", "png"])
with col2:
    demo_button = st.button("🧪 Use Demo Image")
with col3:
    st.markdown("")  # Spacer

DEMO_IMAGE_PATH = "data/samples/ROI_detector/detection_result.png"

if demo_button:
    if os.path.exists(DEMO_IMAGE_PATH):
        st.session_state.demo_active = True
        st.success("✅ Demo image selected.")
    else:
        st.error(f"❌ Demo image not found at {DEMO_IMAGE_PATH}")

if uploaded_file is not None:
    st.session_state.demo_active = False

st.markdown("</div>", unsafe_allow_html=True)

# ========================
# Image Display and Analysis
# ========================
if uploaded_file or st.session_state.get("demo_active", False):
    col_display, col_info = st.columns([1, 1.5], gap="large")

    # Display uploaded or demo image
    with col_display:
        if st.session_state.demo_active:
            demo_img = Image.open(DEMO_IMAGE_PATH)
            st.image(demo_img, caption="📸 Demo Image", use_container_width=True)
            file_name = os.path.basename(DEMO_IMAGE_PATH)
        else:
            st.image(uploaded_file, caption="📸 Uploaded Image", use_container_width=True)
            file_name = uploaded_file.name

    # Display image info
    with col_info:
        if st.session_state.demo_active:
            image = Image.open(DEMO_IMAGE_PATH)
        else:
            image = Image.open(uploaded_file)

        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.07); 
                    backdrop-filter: blur(15px);
                    padding: 25px; 
                    border-radius: 15px; 
                    border: 2px solid rgba(0, 217, 255, 0.25);'>
            <p><b style='color: #00D9FF;'>File Name:</b> {file_name}</p>
            <p><b style='color: #00D9FF;'>Dimensions:</b> {image.size[0]} × {image.size[1]} px</p>
            <p><b style='color: #00D9FF;'>Format:</b> {image.format}</p>
            <p><b style='color: #00D9FF;'>Mode:</b> {image.mode}</p>
        </div>
        """, unsafe_allow_html=True)

    # ========================
    # Run Inference Button
    # ========================
    if st.button("� Analyze Image"):
        with st.spinner("⏳ Processing image... please wait..."):
            try:
                # Load image as numpy array
                if st.session_state.demo_active:
                    image_np = cv2.imread(DEMO_IMAGE_PATH)
                else:
                    image_pil = Image.open(uploaded_file)
                    image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                # Run inference pipeline locally
                output_img, crops, classification_results = pipeline.run_image(
                    image=image_np,
                    save_dir="output/streamlit_results"
                )

                # ========================
                # Display Detection Results
                # ========================
                st.markdown("<p class='section-header'>🎯 Region Analysis</p>", unsafe_allow_html=True)
                
                if output_img is not None:
                    # Convert BGR to RGB for display
                    output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                    st.image(output_img_rgb, caption="🎯 Detected Regions", use_container_width=True)
                else:
                    st.warning("⚠️ Detection image could not be generated.")

                # ========================
                # Display Classification Results
                # ========================
                if classification_results:
                    st.markdown("<p class='section-header'>🩺 Clinical Assessment</p>", unsafe_allow_html=True)
                    
                    n_cols = 4
                    for i in range(0, len(classification_results), n_cols):
                        cols = st.columns(n_cols)
                        for j in range(n_cols):
                            idx = i + j
                            if idx < len(classification_results):
                                res = classification_results[idx]

                                with cols[j]:
                                    # Display cropped ROI
                                    if idx < len(crops):
                                        crop_rgb = cv2.cvtColor(crops[idx], cv2.COLOR_BGR2RGB)
                                        st.image(crop_rgb, use_container_width=True)

                                    # Display classification result
                                    class_name = res.get('class_name', 'Unknown')
                                    confidence = res.get('confidence', 0.0)

                                    # Determine color based on confidence
                                    if confidence > 0.8:
                                        conf_class = "conf-high"
                                    elif confidence > 0.6:
                                        conf_class = "conf-medium"
                                    else:
                                        conf_class = "conf-low"

                                    st.markdown(
                                        f"""
                                        <div class='result-card'>
                                            <b style='color: #00D9FF; font-size: 18px;'>{class_name}</b>
                                            <br><br>
                                            <span class='{conf_class}'>
                                                Confidence: {confidence*100:.2f}%
                                            </span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                    # Display attribution analysis if available and requested
                                    if show_explain and res.get("gradcam"):
                                        try:
                                            if os.path.exists(res["gradcam"]):
                                                gradcam_img = cv2.imread(res["gradcam"])
                                                gradcam_rgb = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
                                                st.image(
                                                    gradcam_rgb,
                                                    caption="🔍 Feature Attribution Map",
                                                    use_container_width=True
                                                )
                                            else:
                                                st.info("💡 Attribution analysis not available.")
                                        except Exception as e:
                                            st.info(f"💡 Could not load attribution map: {e}")
                else:
                    if len(crops) == 0:
                        st.warning("⚠️ No regions detected in this image. Try a different image or check the model configuration.")
                    else:
                        st.warning("⚠️ No classification results available.")

            except Exception as e:
                st.error(f"❌ Error during inference: {str(e)}")
                st.info("💡 Please check the logs or try again with a different image.")


else:
    st.markdown("""
    <div style='text-align: center; 
                padding: 60px 40px; 
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(15px);
                border-radius: 20px; 
                border: 2px dashed rgba(0, 217, 255, 0.3); 
                margin: 30px 0;'>
        <h3 style='color: #00D9FF; font-size: 32px;'>👆 Get Started</h3>
        <p style='color: #E0E0E0; font-size: 16px;'>
            Upload a skin image or use the demo to begin the AI-powered detection and classification process.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========================
# Footer
# ========================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div class='footer'>
        © 2025 SKIN_TELLIGENT | AI-Powered Dermatology Assistant
        <br><small>Developed by <b>Mehraj Alom Tapadar</b> | For Early Skin Health Screening</small>
    </div>
    """,
    unsafe_allow_html=True
)

