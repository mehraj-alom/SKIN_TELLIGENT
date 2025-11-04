import os
import cv2
from box import Box
from pathlib import Path
from src.detection.detector import Detector
from src.detection.postProcessing.ROI_extractor import ROIExtractor
from src.detection.visualize import DetectionVisualizer
from utils.global_utils import read_yaml 
from logger import logger
from src.config.detector_config import DetectorConfig


class DetectionPipeline:
    def __init__(self, config_path: str):
        """
        Initialize Detection Pipeline from YAML config
        """
        logger.info(f"Initializing DetectionPipeline using config: {config_path}")
        self.config = read_yaml(Path(config_path))

        # --- Build DetectorConfig from YAML params ---
        detector_config = DetectorConfig(
            confidence_threshold=self.config.params.confidence_threshold,
            prob_threshold=self.config.params.prob_threshold,
            nms_threshold=self.config.params.nms_threshold,
            input_width=self.config.params.input_width,
            input_height=self.config.params.input_height,
        )

        # --- Initialize Detector ---
        self.detector = Detector(
            model_path=self.config.model.model_path,
            yaml_path=self.config.model.yaml_path,
            config=detector_config
        )

        # --- Initialize ROI Extractor ---
        self.roi_extractor = ROIExtractor()

        # --- Initialize Visualizer (pass labels from detector) ---
        self.visualizer = DetectionVisualizer(labels=self.detector.labels)

        logger.info("DetectionPipeline initialized successfully.")

    def run(self):
        """
        Run full detection pipeline from YAML config
        Steps:
            1. Load image
            2. Run detector
            3. Extract ROIs
            4. Visualize detections
            5. Save outputs
        """
        # --- Paths from YAML ---
        image_path = self.config.inference.input_image
        output_dir = self.config.inference.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # --- Step 1: Load Image ---
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image from {image_path}")
            return None, []

        logger.info(f"Running detection on image: {image_path}")

        # --- Step 2: Object Detection ---
        boxes, confidences, classes, x_factor, y_factor = self.detector.detect(image)
        if not boxes:
            logger.warning("No detections found.")
            return image, []

        # --- Step 3: Extract Cropped ROIs ---
        cropped_regions = self.roi_extractor.crop_rois(image, boxes, x_factor, y_factor)
        logger.info(f"{len(cropped_regions)} ROIs extracted.")

        # --- Step 4: Visualization ---
        visualized_img = self.visualizer.draw_boxes(
            image.copy(), boxes, confidences, classes, x_factor, y_factor
        )

        # --- Step 5: Save Outputs ---
        det_img_path = os.path.join(output_dir, "detections.jpg")
        cv2.imwrite(det_img_path, visualized_img)
        logger.info(f"Detections saved at: {det_img_path}")

        for i, crop in enumerate(cropped_regions):
            crop_path = os.path.join(output_dir, f"crop_{i}.jpg")
            cv2.imwrite(crop_path, crop)
            logger.info(f"Saved ROI: {crop_path}")

        logger.info("Pipeline execution completed successfully.")
        return visualized_img, cropped_regions
