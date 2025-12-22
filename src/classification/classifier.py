from src.config.classifier_config import PredictionConfig
from src.classification.preprocess import Transform
import torch
from pathlib import Path    
from logger import logger
import yaml
import os
from typing import Union, IO
import torch.nn as nn
from torch import load as torch_load
import numpy as np
from PIL import Image
from yaml.loader import SafeLoader
import cv2
from src.explainabaility.gradcampp import GradCAMPlusPlus, apply_heatmap_on_image

class Classifier:
    def __init__(self, config: PredictionConfig):
        """Initialize PyTorch classifier"""

        if not isinstance(config, PredictionConfig):
            raise TypeError(f" expected config to be PredictionConfig, got {type(config)}")

        self.config = config
        self.device = config.device
        self.model_path = config.model_path
        self.labels_path = getattr(config, "labels_path", "src/classification/classifier_labels.yaml")


        self.model = self.load_model()
        if self.model is None:
            logger.warning("M9del not loaded in classifier")

        
        self.class_labels = self.load_labels(self.labels_path)
        if not self.class_labels:
            logger.warning("No class labels loaded in classifier")

        try:
            self.transform = Transform()
        except AttributeError:
            logger.error("Transform not found.")
            self.transform = None         

    def load_labels(self, labels_path: str):
        """Load class labels from YAML file."""
        if not os.path.exists(labels_path):
            logger.error(f"Labels file not found: {labels_path}")
            return []
        try:
            with open(labels_path, "r") as f:
                data = yaml.load(f, Loader=SafeLoader)
            labels = [data["names"][i] for i in sorted(data["names"].keys())]
            logger.info(f"loaded {len(labels)} labels from {labels_path}")
            return labels
        except Exception as e:
            logger.error(f"Error loading labels: {e}")
            return []

    def load_model(self) -> Union[torch.nn.Module, None]:
        """Load PyTorch model"""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return None
        try:
            model = torch_load(self.model_path, map_location=self.device, weights_only=False)
            if hasattr(model, "to"):
                model = model.to(self.device)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logger.exception(f"Failed to load model (classifier): {e}")
            return None

    def predict(self, image_input: Union[str, Path, IO, Image.Image, np.ndarray]) -> Union[dict, None]:
        """Predict disease class for a given ROI image."""
        if self.model is None:
            logger.error("model is not loaded. Prediction aborted.")
            return None
        if self.transform is None:
            logger.error("transform function not available. Prediction aborted.")
            return None

        try:
            if isinstance(image_input, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
                image_tensor = self.transform.preprocess_image(image).unsqueeze(0).to(self.device)

            elif isinstance(image_input, (str, Path)):
                if not os.path.exists(image_input):
                    logger.error(f"Image file not found: {image_input}")
                    return None
                image_tensor = self.transform.preprocess_image(str(image_input)).unsqueeze(0).to(self.device)

            elif isinstance(image_input, Image.Image):
                image_tensor = self.transform.preprocess_image(image_input).unsqueeze(0).to(self.device)
            # file like for Fastapi UploadFile
            else:
                image = Image.open(image_input)
                image_tensor = self.transform.preprocess_image(image).unsqueeze(0).to(self.device)

        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            return None

        # Inference
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = nn.Softmax(dim=1)(output)
                probs = probabilities.cpu().numpy().squeeze()

                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])
                class_name = self.class_labels[top_idx] if self.class_labels else f"class_{top_idx}"
                if confidence < 0.6:
                    class_name = "unknown - better visit a specialist"
                    confidence = 0.0
                logger.info(f"Predicted {class_name} ({confidence:.4f})")
                return {"class_name": class_name, "confidence": confidence, "class_idx": top_idx}

        except Exception as e:
            logger.exception(f"Model inference failed: {e}")
            return None

    def explain(self, image_input: Union[str, Path, IO, Image.Image, np.ndarray], save_path: str = None) -> Union[str, None]:
        """Generate Grad-CAM++ overlay for the top predicted class and save to `save_path`.

        Returns the path to the saved image or None on failure.
        """
        if self.model is None or self.transform is None:
            logger.error("Model or transform not available for explainability")
            return None

        try:
            # Prepare numpy image for overlay size
            if isinstance(image_input, np.ndarray):
                # BGR -> RGB
                orig_np = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(orig_np)
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
                orig_np = np.array(pil_img)
            else:
                # path or file-like
                pil_img = Image.open(image_input).convert("RGB")
                orig_np = np.array(pil_img)

            up_h, up_w = orig_np.shape[0], orig_np.shape[1]

            # Preprocess and get tensor
            input_tensor = self.transform.preprocess_image(pil_img).unsqueeze(0)

            cam_tool = GradCAMPlusPlus(self.model)
            heatmap = cam_tool.generate_cam(input_tensor, target_class=None, upsample_size=(up_h, up_w))

            overlay = apply_heatmap_on_image(orig_np, heatmap, alpha=0.4)

            if save_path is None:
                # default temp path
                save_path = os.path.join("output", "explainability")
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, "gradcam.jpg")
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # overlay is RGB numpy array; save with cv2 (convert RGB->BGR)
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            logger.info(f"Saved Grad-CAM++ overlay at: {save_path}")
            return save_path
        except Exception as e:
            logger.exception(f"Explainability generation failed: {e}")
            return None
