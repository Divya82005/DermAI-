
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image


class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer
        self.activations = None

        # Save the output of the selected layer
        self.forward_handle = (
            target_layer.register_forward_hook(
                self._save_activation
            )
        )

    def _save_activation(self, module, inputs, output):

        self.activations = output

    def generate(self, image_tensor, class_index=None):

        self.model.zero_grad(set_to_none=True)
        self.activations = None

        # Grad-CAM requires gradients even if the caller
        # has disabled gradients elsewhere.
        with torch.enable_grad():

            outputs = self.model(image_tensor)

            class_logits = outputs["class_logits"]

            if class_index is None:
                class_index = torch.argmax(
                    class_logits,
                    dim=1
                ).item()

            target_score = class_logits[0, class_index]

            if self.activations is None:
                raise RuntimeError(
                    "Grad-CAM could not capture target-layer activations."
                )

            # Calculate gradients of the class score
            # with respect to the target-layer activations.
            gradients = torch.autograd.grad(
                outputs=target_score,
                inputs=self.activations,
                retain_graph=False,
                create_graph=False
            )[0]

            activations = self.activations

            # Average gradients across spatial dimensions
            weights = gradients.mean(
                dim=(2, 3),
                keepdim=True
            )

            # Weighted combination of feature maps
            cam = (weights * activations).sum(
                dim=1,
                keepdim=True
            )

            # Keep positive contributions
            cam = F.relu(cam)

            # Resize to model input dimensions
            cam = F.interpolate(
                cam,
                size=image_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            cam = cam[0, 0]

            # Normalize safely to 0–1
            cam_min = cam.min()
            cam_max = cam.max()

            if (cam_max - cam_min).item() > 1e-8:

                cam = (
                    (cam - cam_min)
                    / (cam_max - cam_min)
                )

            else:

                cam = torch.zeros_like(cam)

            return cam.detach().cpu().numpy()

    def close(self):

        self.forward_handle.remove()


def create_heatmap(
    cam,
    original_image,
    lesion_mask,
    alpha=0.45
):
    """
    Create a Grad-CAM overlay restricted to the
    lesion identified by the segmentation mask.

    Outside the segmented lesion, the original image
    remains unchanged.
    """

    # Convert inputs to consistent formats
    original_image = original_image.convert("RGB")
    lesion_mask = lesion_mask.convert("L")

    original_array = np.asarray(
        original_image,
        dtype=np.uint8
    )

    width, height = original_image.size

    # Convert CAM to a floating-point array
    cam = np.asarray(
        cam,
        dtype=np.float32
    )

    # Handle invalid values safely
    cam = np.nan_to_num(
        cam,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    cam = np.clip(cam, 0.0, 1.0)

    # Resize CAM to the original image dimensions
    cam = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

    # Create a color heatmap
    cam_uint8 = np.clip(
        cam * 255,
        0,
        255
    ).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Resize segmentation mask using nearest-neighbor
    # interpolation to preserve mask boundaries.
    lesion_mask = lesion_mask.resize(
        (width, height),
        Image.Resampling.NEAREST
    )

    mask_array = (
        np.asarray(lesion_mask) > 127
    )

    # Start with the original image
    result = original_array.copy()

    # Blend Grad-CAM with the original image
    blended = cv2.addWeighted(
        original_array,
        1.0 - alpha,
        heatmap_rgb,
        alpha,
        0
    )

    # Apply the heatmap only within the lesion mask
    result[mask_array] = blended[mask_array]

    return Image.fromarray(result, mode="RGB")