import cv2
import numpy as np

class DetectionVisualizer:
    """Class for visualizing detection results"""
    
    def __init__(self, labels: list, box_color=(0, 255, 0), text_color=(0, 255, 0)):
        """
        Initialize visualizer
        
        Args:
            labels: List of class labels
            box_color: BGR color tuple for bounding boxes
            text_color: BGR color tuple for text
        """
        self.labels = labels
        self.box_color = box_color
        self.text_color = text_color
    
    def draw_boxes(self, image: np.ndarray, boxes, confidences, classes, x_factor, y_factor):
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            boxes: List of bounding boxes
            confidences: List of confidence scores
            classes: List of class IDs
            x_factor: X-axis scaling factor
            y_factor: Y-axis scaling factor
            
        Returns:
            Image with drawn bounding boxes
        """
        result_img = image.copy()
        
        for i in range(len(boxes)):
            box = boxes[i]
            left = int(box[0] * x_factor)
            top = int(box[1] * y_factor)
            width = int(box[2] * x_factor)
            height = int(box[3] * y_factor)
            
            cv2.rectangle(result_img, (left, top), (left + width, top + height), 
                         self.box_color, 2)
            label = f"{self.labels[classes[i]]}: {confidences[i]:.2f}"
            cv2.putText(result_img, label, (left, top - 10),cv2.FONT_HERSHEY_SIMPLEX,0.5,self.text_color, 2)

        return result_img
    
    def crop_rois(self, image: np.ndarray, boxes, x_factor, y_factor):
        """
        Crop regions of interest (ROIs) from image
        
        Args:
            image: Input image
            boxes: List of bounding boxes
            x_factor: X-axis scaling factor
            y_factor: Y-axis scaling factor
            
        Returns:
            List of cropped ROI images
        """
        rois = []
        
        for box in boxes:
            left = int(box[0] * x_factor)
            top = int(box[1] * y_factor)
            width = int(box[2] * x_factor)
            height = int(box[3] * y_factor)
            
            roi = image[top:top + height, left:left + width]
            rois.append(roi)
        
        return rois
    