# SKIN_TELLIGENT
Smart dermatology assistant that identifies affected regions and classifies skin conditions using deep learning built for accessible, early skin health screening

## SKIN_TELLIGENT — Detection + Classification Pipeline
A modular, config-driven YOLO-based detection pipeline for dermatological image analysis.
Performs object detection, ROI extraction, and visualization of skin regions using OpenCV’s DNN. now extended with a classification modeule that analyses each detected ROI to identify specific skin diseases .

## Features
- Configurable via YAML
- Modular classes: Detector, ROIExtractor, Visualizer , Classifier
- Centralized logging and structured outputs
- Automatic ROI cropping & saving
- Integrated classification for detected skin regions using PyTorch models (.pth)
- Ready for extension (MLflow, AWS)

## Recent Updates
- Added Classifier module — runs on cropped regions from YOLO detections
- Enhanced pipeline to perform end-to-end Detection + Disease Classification
- FastAPI integration for real-time inference
- Future updates: improved inference speed, better UI, and health-risk scoring , Masking for more detailed , VIT (in place of classfier ) , ensemble method , multi modal inference


## Note

This project is under active development. The final README, detailed model documentation, and project motivation will be added soon.

### More Will be added later

## 🧭 SKIN_TELLIGENT Pipeline Overview

<p align="center">
  <img src="https://i.ibb.co/hJfvCpD3/pipeline-overview.png" 
       alt="Detection and Classification Pipeline" width="750">
</p>

