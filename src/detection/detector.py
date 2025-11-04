import cv2 
import numpy as np
from src.config.detector_config import(
    DetectorConfig
) 
from logger import logger
import yaml
from yaml.loader import SafeLoader

class Detector:
        
    def __init__(self, model_path: str, yaml_path: str, config: DetectorConfig = None):
        """
        Initialize detector
        
        Args:
            model_path: Path to ONNX model file
            yaml_path: Path to data.yaml file containing labels
            config: DetectorConfig object with detection parameters
        """
        self.config = config if config else DetectorConfig(confidence_threshold=0.6,
                                                           prob_threshold=0.5,
                                                         nms_threshold=0.5,
                                                           input_width=640,
                                                           input_height=640)
        self.labels = self.load_labels(yaml_path)
        self.model = self.load_model(model_path)



    def load_labels(self, labels_file_path: str):
        """ "Load class labels from YAML file"
        args:
            labels_file_path (str): path to data.yaml file
        returns:
            list : list of class labels
        """
        try:
            with open(labels_file_path, "r") as f:
                data_yaml = yaml.load(f, Loader=SafeLoader)
                logger.info(f"YAML file {labels_file_path} loaded successfully.")
            return data_yaml["names"]
        except Exception as e:
            logger.error(f"Error loading YAML file from {labels_file_path}: {e}")
            raise e
    
    def load_model(self, file_path: str):
        """Load detector model from ONNX file
        args:
            file_path (str): path to onnx model file
        returns:
            cv2.dnn_Net : DNN model
        """
        try:
            detector_model = cv2.dnn.readNetFromONNX(file_path)
            detector_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            detector_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            return detector_model
        except Exception as e:
            logger.error(f"Error loading model from {file_path}: {e}")
            raise e
    def prepare_image(self, image: np.ndarray):
       
        """
        Prepare image for YOLO input
        args:
            image (np.ndarray): input image
        
        Returns:
            blob, input_img, x_factor, y_factor
        """
        img = image.copy()
        if image is not None:
            height, width , c = img.shape
        else:
            logger.info("Found Input image is None. while preparing image for detector input.")
        max_wh = max(width, height)
        input_img = np.zeros((max_wh, max_wh, c), dtype=np.uint8)
        input_img[0:height, 0:width] = img
        blob = cv2.dnn.blobFromImage(image = input_img,
                                     scalefactor = 1/255.0, 
                                     size = (self.config.input_width, self.config.input_height), 
                                     swapRB=True,
                                     crop=False)

        x_factor = max_wh / self.config.input_width
        y_factor = max_wh / self.config.input_height

        return blob, input_img, x_factor, y_factor
    
    def apply_nms(self, detections: np.ndarray):
        """
        Apply Non-Maximum Suppression (NMS) to filter detections
        args:
            detections (np.ndarray): raw detections from the model
        returns:
            boxes, confidences, classes     
        """
        boxes = []
        confidences = []
        classes = []
        
        for i in range(len(detections)):
            row = detections[i]
            confidence = row[4]  # confidence of detection on object
            
            if confidence > self.config.confidence_threshold:
                class_score = row[5:].max()  # it givivin maximum probability from objects
                class_id = row[5:].argmax()  # to get the index position at which max probability occur
                
                if class_score > self.config.prob_threshold:
                    cx, cy, w, h = row[0:4]
                    #bndbox
                    # Left,top, width and  height
                    left = int(cx - 0.5*w)
                    top = int(cy - 0.5*h)
                    width = int(w)
                    height = int(h)
                    
                    box = np.array([left, top, width, height])
                    
                    confidences.append(confidence)
                    boxes.append(box)
                    classes.append(class_id)
        
        return boxes, confidences, classes
    

    def detect(self, image: np.ndarray):
        """
        Perform object detection on the input image
        args:
            image (np.ndarray): input image
        returns:
            boxes, confidences, classes, x_factor, y_factor 
        """
        logger.info("Starting detection process.")

        blob, input_img, x_factor, y_factor = self.prepare_image(image)
        
        self.model.setInput(blob)
        preds = self.model.forward()
        
        if preds.ndim == 2:
            preds = np.expand_dims(preds, axis=0)
        
        detections = preds[0]
        
        #NMS filtering
        boxes, confidences, classes = self.apply_nms(detections)
        
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            self.config.prob_threshold, 
            self.config.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_boxes = [boxes[i] for i in indices]
            filtered_confidences = [confidences[i] for i in indices]
            filtered_classes = [classes[i] for i in indices]
            
            return filtered_boxes, filtered_confidences, filtered_classes, x_factor, y_factor
        else:
            return [], [], [], x_factor, y_factor

    
