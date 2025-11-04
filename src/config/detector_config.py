from logger import logger

class DetectorConfig:
    """ "Configuration class for YOLO model parameters"
    param :
         confidence_threshold (float)  : Confidence threshold for detections.[default = 0.4]
        prob_threshold (float): Probability threshold for class predictions.[default = 0.25]
        nms_threshold (float): Non-maximum suppression threshold.[default = 0.45]
        input_width (int)  Width of the input image for the model.[default = 640]
        input_height (int): Height of the input image for the model.[default = 640]
    """
    logger.info("Initializing DetectorConfig with default/provided parameters.")
    def __init__(self, 
                 confidence_threshold=0.4, 
                 prob_threshold=0.25,
                 nms_threshold=0.45,
                 input_width=640,
                 input_height=640):
        self.confidence_threshold = confidence_threshold
        self.prob_threshold = prob_threshold
        self.nms_threshold = nms_threshold
        self.input_width = input_width
        self.input_height = input_height