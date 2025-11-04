from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
import cv2
import numpy as np
import os
from src.inference.pipeline import DetectionPipeline
from logger import logger

logger.info("Starting SKINTELLIGENT API...")
app = FastAPI(title="SKINTELLIGENT API", description="detection Pipeline")


pipeline = DetectionPipeline(config_path=Path("src/config/pipeline_config.yaml"))


@app.get("/")
def root():
    return {",essage": "SKINTELLIGENT detection API!"}


@app.post("/detect")
async def detect_skin_disease(file: UploadFile = File(...)):
    """
    Upload an image for skin disease detection.
    Returns JSON with detection info and file paths.
    """
    try:
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        
        output_img, crops = pipeline.run_image(image=image, save_dir="output/api_results")
    # jsom response with file paths
        crop_paths = [f"output/api_results/crop_{i}.jpg" for i in range(len(crops))]
        result_path = "output/api_results/detections.jpg"

        response = {
            "message": "Detection completed successfully.",
            "detections": len(crops),
            "detection_image": result_path,
            "roi_crops": crop_paths
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error during API detection: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/download/{filename}")
def download_file(filename: str):
    """
    Download processed image or crop by filename.
    """
    file_path = Path("output/api_results") / filename
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(file_path)
