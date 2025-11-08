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
from src.config.classifier_config import PredictionConfig
from src.classification.classifier import Classifier
import yaml
from yaml import SafeLoader

class DetectionAndClassificationPipeline:
    def __init__(self, config_path: str):
        """
        
        """
        logger.info(f"Initializing DetectionAndClassificationPipeline using config: {config_path}")
        self.config = read_yaml(Path(config_path))

        # DetectorConfig from YAML params 
        detector_config = DetectorConfig(
            confidence_threshold=self.config.params.confidence_threshold,
            prob_threshold=self.config.params.prob_threshold,
            nms_threshold=self.config.params.nms_threshold,
            input_width=self.config.params.input_width,
            input_height=self.config.params.input_height,
        )

        #  Detector 
        self.detector = Detector(
            model_path=self.config.model.model_path,
            yaml_path=self.config.model.yaml_path,
            config=detector_config
        )
          #Initialize Classifier ---
        Classifier_config = PredictionConfig(
            model_path=Path(self.config.model.classifier_model_path),
            device="cpu"  
        )

        # 
        #Initialize ROI Extractor ---
        self.roi_extractor = ROIExtractor()
        # classifier initialized
        self.classifier = Classifier(config=Classifier_config)

        # Initialize Visualizer (pass labels from detector) ---
        self.visualizer = DetectionVisualizer(labels=self.detector.labels)

        self.class_labels = self.load_class_labels(self.config.model.classifier_labels_path)

        logger.info("DetectionAndClassificationPipeline initialized successfully.")
    

    def load_class_labels(self, labels_path: str):
        """Load disease class names from YAML file."""

        if not os.path.exists(labels_path):
            logger.warning(f"Labels YAML not found at {labels_path}")
            return []

        with open(labels_path, "r") as f:
            data = yaml.load(f, Loader=SafeLoader)
        return [data["names"][i] for i in sorted(data["names"].keys())]
    


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

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image from {image_path}")
            return None, []

        logger.info(f"Running detection on image: {image_path}")

        #      Detection
        boxes, confidences, classes, x_factor, y_factor = self.detector.detect(image)
        if not boxes:
            logger.warning("No detections found.")
            logger.warning("sending the input image for classification.")
            classification_result = self.classifier.predict(image)
            return image, [], classification_result

        # extract Cropped ROIs
        cropped_regions = self.roi_extractor.crop_rois(image, boxes, x_factor, y_factor)
        logger.info(f"{len(cropped_regions)} ROIs extracted.")

        # classification on each cropped ROI
        classifier_results = []

        for i, roi in enumerate(cropped_regions):
            preds = self.classifier.predict(roi)
            if preds is not None:
                classifier_results.append(preds)
                logger.info(f"ROI {i}: Predicted disease  with confidence")
            else:
                logger.warning(f"ROI {i}: Classification failed.")    
    

 
        visualized_img = self.visualizer.draw_boxes(
            image.copy(), boxes, confidences, classes, x_factor, y_factor
        )

        det_img_path = os.path.join(output_dir, "detections.jpg")
        cv2.imwrite(det_img_path, visualized_img)
        logger.info(f"Detections saved at: {det_img_path}")

        for i, crop in enumerate(cropped_regions):
            crop_path = os.path.join(output_dir, f"crop_{i}.jpg")
            cv2.imwrite(crop_path, crop)
            logger.info(f"Saved ROI: {crop_path}")
            logger.info(f"Classifier result for ROI {i}: {classifier_results[i]}")

        logger.info(f"pipeline  ( Detection + Classification) completed.")
        return visualized_img, cropped_regions , classifier_results
    
     # Fastapi routes 
    def run_image(self, image, save_dir="output/api_results"):
        """
        Run detection pipeline on a given image and save results.
        Args:
            image: Input image as a numpy array.
            save_dir: Directory to save output images.
        Returns:
            visualized_img: Image with detections visualized.
            cropped_regions: List of cropped ROI images.    
            
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        boxes, confidences, classes, x_factor, y_factor = self.detector.detect(image)
        if not boxes:
            logger.warning("no detections found.")        # send the input image for classification. neeed to be corrected the logui heree 
            return image, []

        cropped_regions = self.roi_extractor.crop_rois(image, boxes, x_factor, y_factor)
        
        classification_results = []
        for i, crop in enumerate(cropped_regions):
            preds = self.classifier.predict(crop)
            logger.info(f"Classifier output for ROI {i}: {preds}")
            if preds and isinstance(preds, dict):
                preds["roi_index"] = i
                classification_results.append(preds)
            else:
                logger.warning(f"ROI {i}: No valid prediction returned.")


        visualized_img = self.visualizer.draw_boxes(image.copy(), 
                                                    boxes, 
                                                    confidences, 
                                                    classes, 
                                                    x_factor,
                                                    y_factor)
        
        result_path = os.path.join(save_dir, "detections.jpg")
        cv2.imwrite(result_path, visualized_img)
        for i, crop in enumerate(cropped_regions):
            cv2.imwrite(os.path.join(save_dir, f"crop_{i}.jpg"), crop)

        return visualized_img, cropped_regions, classification_results
