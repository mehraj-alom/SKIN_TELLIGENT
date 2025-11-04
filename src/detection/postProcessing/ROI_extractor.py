import numpy as np

class ROIExtractor:

    def __init__(self):
        pass

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
    