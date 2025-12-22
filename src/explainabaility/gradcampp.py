import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import cv2

class GradCAMPlusPlus:
    """A lightweight Grad-CAM++ implementation for PyTorch models.

    Usage:
        cam = GradCAMPlusPlus(model, target_layer)
        heatmap = cam.generate_cam(input_tensor, target_class)
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        # find a default target conv layer if not provided
        if target_layer is None:
            target_layer = self._find_target_layer()
        self.target_layer = target_layer
        # register hooks
        self._register_hooks()

    def _find_target_layer(self):
        # pick the last Conv2d layer in the model
        target = None
        for name, module in self.model.named_modules():

            if isinstance(module, nn.Conv2d):
                target = module
        if target is None:
            raise RuntimeError("No Conv2d layer found in model for Grad-CAM")
        return target

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple; take first
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class=None, upsample_size=None):
        """Generate Grad-CAM++ heatmap for an input tensor.

        Args:
            input_tensor: torch.Tensor shape (1,C,H,W)
            target_class: int index of target class. If None uses argmax.
            upsample_size: tuple (H,W) to resize heatmap to. If None uses activations size.

        Returns:
            heatmap: numpy array HxW normalized to [0,1]
        """
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        # FOorward
        output = self.model(input_tensor)
        if isinstance(output, tuple) or isinstance(output, list):
            logits = output[0]
        else:
            logits = output

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        # one-hot and backward
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits, device=device)
        one_hot[0, target_class] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Hooks didn't capture activations/gradients")

        # A: activations (N, C, H, W), grads (N, C, H, W)
        activations = self.activations[0]
        grads = self.gradients[0]

        # Grad-CAM++ weights (approxiMation)
        grads_power_2 = grads.pow(2)
        grads_power_3 = grads.pow(3)

        # Sum over spatial dimen
        sum_activations = torch.sum(activations, dim=(1, 2), keepdim=True)
        eps = 1e-8
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + activations * grads_power_3.sum(dim=(1, 2), keepdim=True)
        alpha = alpha_num / (alpha_denom + eps)

        # avoid keeping gradients for cam computation
        score = logits[0, target_class].detach()
        positive_grads = F.relu(torch.exp(score) * grads)
        weights = (alpha * positive_grads).sum(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        # normalize (detach to prevent grad tensors)
        cam_np = cam.detach().cpu().numpy()
        cam_np -= cam_np.min()
        if cam_np.max() != 0:
            cam_np = cam_np / (cam_np.max() + eps)

        # upsample if Req
        if upsample_size is not None:

            cam_np = cv2.resize(cam_np, (upsample_size[1], upsample_size[0]))

        return cam_np


def apply_heatmap_on_image(image_np, heatmap, colormap=None, alpha=0.5):
    

    if colormap is None:
        colormap = cv2.COLORMAP_JET

    heatmap_uint8 = (heatmap * 255).astype('uint8')
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    if image_np.dtype != 'uint8':
        image_uint8 = (image_np * 255).astype('uint8')
    else:
        image_uint8 = image_np
    overlay = cv2.addWeighted(image_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
