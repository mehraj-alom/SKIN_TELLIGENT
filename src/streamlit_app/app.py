import streamlit as st
import os
from PIL import Image
import numpy as np
from pathlib import Path
import cv2
from io import BytesIO
import sys
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the pipeline directly (no FastAPI)
from src.inference.pipeline import DetectionAndClassificationPipeline
from src.streamlit_app.chatbot import get_chatbot_response, get_chatbot_stream, get_inference_state, InferenceState, get_abstention_message, get_uncertainty_disclaimer
from src.inference.audit import get_audit_logger
from src.config.model_registry import get_model_registry
from src.inference.contracts import InferenceResult, ClassificationResult, InferenceState as ContractState
from uuid import uuid4
import datetime

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
# Enhanced CSS Styling - Production Grade
# ========================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { 
        font-family: 'Inter', sans-serif;
    }
    
    /* Professional dark background */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #151b28 50%, #0f1419 100%);
        background-attachment: fixed;
        min-height: 100vh;
    }
    
    /* Main title - Professional */
    .main-title {
        font-size: 48px;
        font-weight: 700;
        text-align: center;
        color: #1a9fa0;
        margin-bottom: 5px;
        letter-spacing: 0.5px;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #8a92a8;
        font-size: 16px;
        margin-bottom: 40px;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* Upload card - Clean glassmorphism */
    .upload-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(26, 159, 160, 0.15);
        border-radius: 12px;
        padding: 30px;
        margin: 25px 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    
    .upload-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(26, 159, 160, 0.25);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
    }
    
    /* Button styling - Minimal and professional */
    .stButton > button {
        background: linear-gradient(135deg, #1a9fa0 0%, #157879 100%);
        color: white;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        padding: 12px 32px;
        border: none;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(26, 159, 160, 0.2);
        cursor: pointer;
        width: 100%;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #229fa0 0%, #1a8a8b 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(26, 159, 160, 0.3);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Result cards - Clean design */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(26, 159, 160, 0.2);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        transition: all 0.2s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        border-color: rgba(26, 159, 160, 0.4);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Confidence colors - Professional */
    .conf-high {
        color: #2ecc71;
        font-weight: 600;
    }
    
    .conf-medium {
        color: #f39c12;
        font-weight: 600;
    }
    
    .conf-low {
        color: #e74c3c;
        font-weight: 600;
    }
    
    /* Section headers */
    .section-header {
        color: #1a9fa0;
        font-size: 28px;
        font-weight: 600;
        margin: 35px 0 25px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(26, 159, 160, 0.25);
        text-align: center;
        letter-spacing: 0.3px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        padding: 20px;
        margin-top: 50px;
        border-top: 1px solid rgba(26, 159, 160, 0.15);
        font-size: 13px;
        line-height: 1.6;
    }
    
    /* Horizontal rule */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(26, 159, 160, 0.2), transparent);
        margin: 40px 0;
    }
    
    /* Hide default Streamlit UI elements */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 8px;
        border-left: 4px solid;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(26, 159, 160, 0.1);
    }
    
    /* Metric styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 12px;
    }
    
    /* Floating Chat Bubble Section */
    .floating-chat-container {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 9999;
    }
    
    .chat-bubble {
        background: linear-gradient(135deg, #1a9fa0 0%, #157879 100%);
        width: 180px;  /* Wider for rectangle/pill */
        height: 60px;  /* Taller */
        border-radius: 12px; /* Rectangle with rounded corners */
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 6px 16px rgba(26, 159, 160, 0.4);
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        color: white;
        font-size: 18px; /* Adjusted font size for text */
        font-weight: 600;
        border: none;
        gap: 10px; /* Space between icon and text */
        padding: 0 20px;
    }
    
    .chat-bubble:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(26, 159, 160, 0.6);
        background: linear-gradient(135deg, #229fa0 0%, #1a8a8b 100%);
    }
    
    .chat-window {
        /* Styles moved below for organization */
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-header {
        background: rgba(26, 159, 160, 0.1);
        padding: 15px 20px;
        border-bottom: 1px solid rgba(26, 159, 160, 0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 15px;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    
    .chat-input-area {
        padding: 15px;
        border-top: 1px solid rgba(26, 159, 160, 0.2);
    }
    
    /* Chat Window - Single unified box */
    .chat-window {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 400px;
        height: 550px;
        max-height: 80vh;
        background: #0f1419;
        border: 2px solid #1a9fa0;
        border-radius: 16px;
        display: flex;
        flex-direction: column;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8);
        z-index: 99999;
        overflow: hidden;
        animation: slideIn 0.3s ease-out;
    }
    
    /* Style text input inside chat window */
    .chat-window input[type="text"] {
        background: rgba(26, 159, 160, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(26, 159, 160, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Style form submit button */
    .chat-window button[kind="formSubmit"] {
        background: linear-gradient(135deg, #1a9fa0 0%, #157879 100%) !important;
        border: none !important;
        border-radius: 8px !important;
    }
    
    /* Safely target the Floating Chat Button via wrapper class */
    .floating-chat-container {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 99990; /* High but below window/input */
    }
    
    .floating-chat-container button {
        background: linear-gradient(135deg, #1a9fa0 0%, #157879 100%) !important;
        border-radius: 30px !important;
        height: 60px !important;
        width: 180px !important;
        padding: 0 20px !important;
        border: 2px solid white !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.5) !important;
    }
    
    .floating-chat-container button p {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: white !important;
    }
    
    .floating-chat-container button:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(26, 159, 160, 0.6) !important;
    }
    
    /* Safely target the Floating Chat Button via wrapper class */
    .floating-chat-container {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 99990; /* High but below window/input */
    }
    
    .floating-chat-container button {
        background: linear-gradient(135deg, #1a9fa0 0%, #157879 100%) !important;
        border-radius: 30px !important;
        height: 60px !important;
        width: 180px !important;
        padding: 0 20px !important;
        border: 2px solid white !important;
        box-shadow: 0 6px 16px rgba(0,0,0,0.5) !important;
    }
    
    .floating-chat-container button p {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: white !important;
    }
    
    .floating-chat-container button:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(26, 159, 160, 0.6) !important;
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
    # Register models on startup
    registry = get_model_registry()
    try:
        registry.register_model("detector", "models/yolov8_detector.onnx") # Update path if needed
        registry.register_model("classifier", "models/classifier_mdl.pth")
    except Exception as e:
        pass # Handle gracefully if models not found individually

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "demo_active" not in st.session_state:
    st.session_state.demo_active = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "classification_results" not in st.session_state:
    st.session_state.classification_results = None

pipeline = st.session_state.pipeline

# ========================
# Header
# ========================
st.markdown("<p class='main-title'>SKIN_TELLIGENT</p>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>AI-Powered Clinical Analysis for Dermatology Screening</p>",
    unsafe_allow_html=True
)

# ========================
# Sidebar: Information and controls
# ========================
with st.sidebar:
    st.markdown("""
    <div style='background: rgba(26, 159, 160, 0.08); 
                padding: 18px; 
                border-radius: 10px; 
                border: 1px solid rgba(26, 159, 160, 0.2);
                margin-bottom: 25px;'>
        <h3 style='color: #1a9fa0; margin-top: 0; font-size: 18px;'>SKIN_TELLIGENT</h3>
        <p style='color: #b5bcc7; font-size: 13px; line-height: 1.6; margin: 10px 0 0 0;'>
            <b>Clinical Analysis Platform</b><br><br>
            Powered by advanced deep learning for:
        </p>
        <ul style='color: #b5bcc7; font-size: 12px; margin: 10px 0;'>
            <li>Region Detection & Localization</li>
            <li>Clinical Classification</li>
            <li>Feature Attribution Analysis</li>
        </ul>
        <p style='color: #1a9fa0; font-size: 11px; margin-top: 12px; font-style: italic;'>
            Supporting early screening workflows
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <h3 style='color: #1a9fa0; margin-top: 10px; font-size: 16px;'>Quick Guide</h3>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='color: #b5bcc7; font-size: 12px; line-height: 1.8;'>
        1. Upload or select demo image<br>
        2. Click "Analyze Image"<br>
        3. Review detected regions<br>
        4. Expand "Why this decision?" for insights
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dynamic status section
    st.markdown("""
    <h3 style='color: #1a9fa0; font-size: 16px;'>System Status</h3>
    """, unsafe_allow_html=True)
    
    col_status1, col_status2 = st.columns([1, 1])
    with col_status1:
        st.metric("Detector", "Ready", label_visibility="collapsed")
    with col_status2:
        st.metric("Classifier", "Ready", label_visibility="collapsed")
    
    st.markdown("""
    <div style='background: rgba(46, 204, 113, 0.1); 
                padding: 12px; 
                border-radius: 8px; 
                border-left: 3px solid #2ecc71;
                margin-top: 10px;'>
        <p style='color: #2ecc71; font-size: 11px; margin: 0;'>
            All models initialized and ready
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <h3 style='color: #1a9fa0; font-size: 16px;'>Model Architecture</h3>
    <div style='background: rgba(255, 255, 255, 0.03); 
                padding: 14px; 
                border-radius: 8px; 
                border: 1px solid rgba(26, 159, 160, 0.15);
                font-size: 12px;'>
        <p style='color: #b5bcc7; margin: 5px 0;'>
            <b style='color: #1a9fa0;'>Region Detection</b><br>
            YOLO v8 Neural Network
        </p>
        <p style='color: #b5bcc7; margin: 8px 0 5px 0;'>
            <b style='color: #1a9fa0;'>Classification</b><br>
            Custom Trained Model
        </p>
        <p style='color: #b5bcc7; margin: 8px 0 5px 0;'>
            <b style='color: #1a9fa0;'>Explainability</b><br>
            Feature Attribution Mapping
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style='background: rgba(231, 76, 60, 0.08); 
                padding: 14px; 
                border-radius: 8px; 
                border: 1px solid rgba(231, 76, 60, 0.2);
                margin-top: 20px;'>
        <p style='color: #b5bcc7; font-size: 11px; line-height: 1.6; margin: 0;'>
            <b style='color: #e74c3c;'>Disclaimer</b><br>
            For clinical support only. Consult qualified medical professionals for diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========================
# Upload / Demo Section
# ========================
st.markdown("<div class='upload-card'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
with col2:
    demo_button = st.button("Use Demo Image")
with col3:
    st.markdown("")  # Spacer

DEMO_IMAGE_PATH = "data/samples/ROI_detector/detection_result.png"

if demo_button:
    if os.path.exists(DEMO_IMAGE_PATH):
        st.session_state.demo_active = True
        st.success("Demo image loaded")
    else:
        st.error(f"Demo image not found at {DEMO_IMAGE_PATH}")

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
            st.image(demo_img, caption="Demo Image", use_container_width=True)
            file_name = os.path.basename(DEMO_IMAGE_PATH)
        else:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            file_name = uploaded_file.name

    # Display image info
    with col_info:
        if st.session_state.demo_active:
            image = Image.open(DEMO_IMAGE_PATH)
        else:
            image = Image.open(uploaded_file)

        st.markdown(f"""
        <div style='background: rgba(255, 255, 255, 0.04); 
                    backdrop-filter: blur(10px);
                    padding: 20px; 
                    border-radius: 10px; 
                    border: 1px solid rgba(26, 159, 160, 0.15);'>
            <p style='color: #b5bcc7; margin: 4px 0;'><span style='color: #1a9fa0; font-weight: 600;'>File:</span> {file_name}</p>
            <p style='color: #b5bcc7; margin: 4px 0;'><span style='color: #1a9fa0; font-weight: 600;'>Size:</span> {image.size[0]} × {image.size[1]} px</p>
            <p style='color: #b5bcc7; margin: 4px 0;'><span style='color: #1a9fa0; font-weight: 600;'>Format:</span> {image.format}</p>
            <p style='color: #b5bcc7; margin: 4px 0;'><span style='color: #1a9fa0; font-weight: 600;'>Mode:</span> {image.mode}</p>
        </div>
        """, unsafe_allow_html=True)

    # ========================
    # Run Inference Button
    # ========================
    if st.button("Analyze Image"):
        # Create status container in sidebar for live updates
        with st.sidebar:
            status_container = st.empty()
        
        with st.spinner("Processing image..."):
            try:
                # Load image as numpy array
                if st.session_state.demo_active:
                    image_np = cv2.imread(DEMO_IMAGE_PATH)
                else:
                    image_pil = Image.open(uploaded_file)
                    image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                # Update sidebar status - Stage 1: Analyzing
                with status_container.container():
                    st.markdown("""
                    <div style='background: rgba(26, 159, 160, 0.12); 
                                padding: 12px; 
                                border-radius: 8px; 
                                border-left: 3px solid #1a9fa0;'>
                        <p style='color: #1a9fa0; font-size: 12px; margin: 0;'>
                            <b>Analyzing image...</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Run inference pipeline locally
                output_img, crops, classification_results = pipeline.run_image(
                    image=image_np,
                    save_dir="output/streamlit_results"
                )
                # Store ALL results in session state for persistence
                st.session_state.classification_results = classification_results
                st.session_state.output_img = output_img
                st.session_state.crops = crops
                st.session_state.analysis_complete = True
                
                # ========================
                # Audit Logging - Decision Governance
                # ========================
                audit_logger = get_audit_logger()
                model_registry = get_model_registry()
                model_version = model_registry.get_combined_version()
                
                # Construct InferenceResult object for logging
                inference_result = InferenceResult(
                    session_id=st.session_state.session_id,
                    model_version=model_version,
                    classifications=[]
                )
                
                for idx, res in enumerate(classification_results):
                    cls_result = ClassificationResult.from_raw(res, idx)
                    inference_result.classifications.append(cls_result)
                
                inference_result.compute_overall_state()
                
                audit_logger.log_inference(
                    session_id=st.session_state.session_id,
                    result=inference_result,
                    model_version=model_version
                )

                # Update sidebar status - Stage 2: Thinking/Processing
                with status_container.container():
                    st.markdown("""
                    <div style='background: rgba(155, 89, 182, 0.12); 
                                padding: 12px; 
                                border-radius: 8px; 
                                border-left: 3px solid #9b59b6;'>
                        <p style='color: #9b59b6; font-size: 12px; margin: 0;'>
                            <b>Classifying regions...</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Update sidebar status - Complete
                with status_container.container():
                    st.markdown("""
                    <div style='background: rgba(46, 204, 113, 0.12); 
                                padding: 12px; 
                                border-radius: 8px; 
                                border-left: 3px solid #2ecc71;'>
                        <p style='color: #2ecc71; font-size: 12px; margin: 0;'>
                            <b>Analysis Complete</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # ========================
                # Display Detection Results
                # ========================
                st.markdown("<p class='section-header'>Region Analysis</p>", unsafe_allow_html=True)
                
                if output_img is not None:
                    # Convert BGR to RGB for display
                    output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                    st.image(output_img_rgb, caption="Detected Regions", use_container_width=True)
                else:
                    st.warning("⚠️ Detection image could not be generated.")

                # ========================
                # Display Classification Results
                # ========================
                if classification_results:
                    st.markdown("<p class='section-header'>Clinical Assessment</p>", unsafe_allow_html=True)
                    
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
                                            <b style='color: #1a9fa0; font-size: 18px;'>{class_name}</b>
                                            <br><br>
                                            <span class='{conf_class}'>
                                                Confidence: {confidence*100:.2f}%
                                            </span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                    # Expandable attribution analysis
                                    if res.get("gradcam"):
                                        with st.expander(f"Why this decision? (Region {idx + 1})"):
                                            st.markdown("""
                                            <p style='color: #b5bcc7; font-size: 12px;'>
                                                <i>Feature Attribution Map - Shows which areas influenced the classification.</i>
                                            </p>
                                            """, unsafe_allow_html=True)
                                            try:
                                                if os.path.exists(res["gradcam"]):
                                                    gradcam_img = cv2.imread(res["gradcam"])
                                                    gradcam_rgb = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
                                                    st.image(
                                                        gradcam_rgb,
                                                        caption="Feature Attribution - Red indicates influential areas",
                                                        use_container_width=True
                                                    )
                                                else:
                                                    st.info("Attribution analysis not available for this region.")
                                            except Exception as e:
                                                st.info(f"Could not load attribution map: {e}")
                else:
                    if len(crops) == 0:
                        st.warning("No regions detected in this image. Try a different image.")
                    else:
                        st.warning("No classification results available.")

            except Exception as e:
                with status_container.container():
                    st.markdown(f"""
                    <div style='background: rgba(231, 76, 60, 0.12); 
                                padding: 12px; 
                                border-radius: 8px; 
                                border-left: 3px solid #e74c3c;'>
                        <p style='color: #e74c3c; font-size: 12px; margin: 0;'>
                            <b>Analysis Failed</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                st.error(f"Error during inference: {str(e)}")
                st.info("Please check the logs or try again with a different image.")

    # ========================
    # Display Persisted Analysis Results (outside button block)
    # ========================
    if st.session_state.get("analysis_complete", False):
        output_img = st.session_state.get("output_img")
        crops = st.session_state.get("crops", [])
        classification_results = st.session_state.get("classification_results", [])
        
        st.markdown("<p class='section-header'>Region Analysis</p>", unsafe_allow_html=True)
        
        if output_img is not None:
            output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            st.image(output_img_rgb, caption="Detected Regions", use_container_width=True)
        
        if classification_results:
            st.markdown("<p class='section-header'>Clinical Assessment</p>", unsafe_allow_html=True)
            
            n_cols = 4
            for i in range(0, len(classification_results), n_cols):
                cols = st.columns(n_cols)
                for j in range(n_cols):
                    idx = i + j
                    if idx < len(classification_results):
                        res = classification_results[idx]
                        with cols[j]:
                            if idx < len(crops):
                                crop_rgb = cv2.cvtColor(crops[idx], cv2.COLOR_BGR2RGB)
                                st.image(crop_rgb, use_container_width=True)
                            
                            class_name = res.get('class_name', 'Unknown')
                            confidence = res.get('confidence', 0.0)
                            
                            if confidence > 0.8:
                                conf_class = "conf-high"
                            elif confidence > 0.6:
                                conf_class = "conf-medium"
                            else:
                                conf_class = "conf-low"
                            
                            st.markdown(
                                f"""<div class='result-card'>
                                    <b style='color: #1a9fa0; font-size: 18px;'>{class_name}</b>
                                    <br><br>
                                    <span class='{conf_class}'>Confidence: {confidence*100:.2f}%</span>
                                </div>""",
                                unsafe_allow_html=True
                            )
                            
                            if res.get("gradcam") and os.path.exists(res["gradcam"]):
                                with st.expander(f"Why this decision? (Region {idx + 1})"):
                                    gradcam_img = cv2.imread(res["gradcam"])
                                    gradcam_rgb = cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB)
                                    st.image(gradcam_rgb, caption="Feature Attribution", use_container_width=True)


else:
    st.markdown("""
    <div style='text-align: center; 
                padding: 60px 40px; 
                background: rgba(255, 255, 255, 0.04);
                backdrop-filter: blur(10px);
                border-radius: 12px; 
                border: 1px dashed rgba(26, 159, 160, 0.2); 
                margin: 30px 0;'>
        <h3 style='color: #1a9fa0; font-size: 28px; margin-top: 0;'>Get Started</h3>
        <p style='color: #b5bcc7; font-size: 15px;'>
            Upload a skin image or use the demo to begin analysis.
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
        © 2025 SKIN_TELLIGENT | Clinical Dermatology Analysis<br>
        <small>Developed by Mehraj Alom Tapadar</small><br>
        <small style='color: #f39c12;'>For educational purposes only - Not for medical diagnosis.</small>
    </div>
    """,
    unsafe_allow_html=True
)

# ========================
# Sidebar Chatbot UI (Reliable rendering)
# ========================
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <h3 style='color: #1a9fa0; font-size: 16px;'>🧬 Knowledge-Restricted Assistant</h3>
    """, unsafe_allow_html=True)
    
    # Determine inference state from classification results
    results = st.session_state.get("classification_results", [])
    max_confidence = 0.0
    if results:
        max_confidence = max(r.get('confidence', 0.0) for r in results)
    
    inference_state = get_inference_state(max_confidence)
    
    # Display inference state indicator
    if st.session_state.get("analysis_complete", False):
        if inference_state == InferenceState.HIGH_CONFIDENCE:
            st.success(f"✅ High Confidence ({max_confidence*100:.0f}%) - Full assistance available")
        elif inference_state == InferenceState.UNCERTAIN:
            st.warning(f"⚠️ Uncertain ({max_confidence*100:.0f}%) - Limited assistance")
        else:
            st.error(f"❌ Low Confidence ({max_confidence*100:.0f}%) - Abstaining")
    
    # Message Display Area in sidebar
    chat_container = st.container(height=280)
    with chat_container:
        if not st.session_state.chat_history:
            if not st.session_state.get("analysis_complete", False):
                st.info("👋 Upload and analyze an image to begin.")
            elif inference_state == InferenceState.ABSTAIN:
                st.warning(get_abstention_message())
            elif inference_state == InferenceState.UNCERTAIN:
                st.info("👋 I can provide limited educational information. How can I help?")
            else:
                st.info("👋 Analysis complete. Ask me about the detected condition(s).")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Context Preparation
    if results:
        context = "The AI detected the following: " + ", ".join([f"{r.get('class_name', 'Unknown')} with {r.get('confidence', 0.0)*100:.1f}% confidence" for r in results])
    else:
        context = "No analysis has been performed yet."
    
    # Chat Input - Only enable if analysis is complete
    if st.session_state.get("analysis_complete", False):
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask a question...", key="chat_text_input", label_visibility="collapsed")
            submit_button = st.form_submit_button("➤ Send", use_container_width=True)
            
            if submit_button and user_input:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Display user message immediately
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Stream assistant response
                with st.chat_message("assistant"):
                    try:
                        stream = get_chatbot_stream(
                            user_input, 
                            context, 
                            st.session_state.chat_history[:-1],
                            inference_state=inference_state.value
                        )
                        response = st.write_stream(stream)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        
                        # Log chat
                        get_audit_logger().log_chat(
                            session_id=st.session_state.get("session_id", "unknown"),
                            user_query=user_input,
                            assistant_response=response,
                            inference_state=inference_state.value
                        )
                        
                    except Exception as e:
                        st.error(f"Chat error: {e}")
                        get_audit_logger().log_error(
                            session_id=st.session_state.get("session_id", "unknown"),
                            error_type="CHAT_ERROR",
                            error_message=str(e)
                        )
                # No rerun needed if we update UI in place, but good for consistency
                # st.rerun() 
    else:
        st.info("ℹ️ Analyze an image to enable the assistant.")
