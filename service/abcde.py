
import math

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------
# Helper: Convert PIL image to NumPy RGB image
# ---------------------------------------------------------

def pil_to_rgb_array(image):
    return np.array(image.convert("RGB"))


# ---------------------------------------------------------
# Helper: Convert segmentation mask to binary NumPy mask
# ---------------------------------------------------------

def mask_to_binary(mask):

    # Convert mask to grayscale
    if isinstance(mask, Image.Image):
        mask_array = np.array(mask.convert("L"))
    else:
        mask_array = np.array(mask)

        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]

    # Support both probability masks [0, 1]
    # and 8-bit masks [0, 255]
    if mask_array.size == 0:
        return np.zeros(mask_array.shape[:2], dtype=np.uint8)

    if np.issubdtype(mask_array.dtype, np.floating):
        threshold = 0.5 if mask_array.max() <= 1.0 else 127.0
    else:
        threshold = 127

    binary_mask = (
        mask_array > threshold
    ).astype(np.uint8)

    return binary_mask


# ---------------------------------------------------------
# Helper: Match mask dimensions to original image
# ---------------------------------------------------------

def resize_mask_to_image(binary_mask, image):

    image_array = pil_to_rgb_array(image)

    height, width = image_array.shape[:2]

    # Resize only when dimensions differ
    if binary_mask.shape[:2] != (height, width):

        binary_mask = cv2.resize(
            binary_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    binary_mask = (
        binary_mask > 0
    ).astype(np.uint8)

    return binary_mask


# ---------------------------------------------------------
# A — Asymmetry
# ---------------------------------------------------------

def calculate_asymmetry(binary_mask):

    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0:
        return 0.0

    lesion_area = int(np.sum(binary_mask))

    if lesion_area == 0:
        return 0.0

    # Find the lesion bounding box
    min_x = xs.min()
    max_x = xs.max()
    min_y = ys.min()
    max_y = ys.max()

    # Crop the lesion region so reflections are
    # calculated around the lesion bounding box.
    lesion_crop = binary_mask[
        min_y:max_y + 1,
        min_x:max_x + 1
    ]

    # Reflection across the vertical axis
    flipped_vertical = cv2.flip(lesion_crop, 1)

    vertical_difference = np.sum(
        np.abs(
            lesion_crop.astype(np.int16)
            - flipped_vertical.astype(np.int16)
        )
    )

    # Reflection across the horizontal axis
    flipped_horizontal = cv2.flip(lesion_crop, 0)

    horizontal_difference = np.sum(
        np.abs(
            lesion_crop.astype(np.int16)
            - flipped_horizontal.astype(np.int16)
        )
    )

    crop_area = np.sum(lesion_crop)

    asymmetry = (
        vertical_difference + horizontal_difference
    ) / (2.0 * crop_area)

    return float(
        min(max(asymmetry, 0.0), 1.0)
    )


# ---------------------------------------------------------
# B — Border irregularity
# ---------------------------------------------------------

def calculate_border_irregularity(binary_mask):

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if area <= 0 or perimeter <= 0:
        return 0.0

    # Circularity = 4πA / P²
    # A perfect circle has circularity close to 1.
    circularity = (
        4.0 * math.pi * area
    ) / (
        perimeter ** 2
    )

    circularity = min(
        max(circularity, 0.0),
        1.0
    )

    # Higher values indicate a less circular boundary.
    return float(1.0 - circularity)


# ---------------------------------------------------------
# C — Color variation
# ---------------------------------------------------------

def calculate_color_variation(image, binary_mask):

    image_array = pil_to_rgb_array(image)

    # Safety check: image and mask must have equal dimensions
    if binary_mask.shape != image_array.shape[:2]:
        raise ValueError(
            "Color analysis requires matching image and mask "
            f"dimensions. Image: {image_array.shape[:2]}, "
            f"mask: {binary_mask.shape}"
        )

    lesion_region = binary_mask > 0

    if not np.any(lesion_region):
        return 0.0

    # Convert original RGB image to HSV
    hsv_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2HSV
    )

    lesion_hsv = hsv_image[lesion_region]

    # Standard deviation of HSV channels
    hue_std = np.std(lesion_hsv[:, 0])
    saturation_std = np.std(lesion_hsv[:, 1])
    value_std = np.std(lesion_hsv[:, 2])

    # Normalize channel variations
    hue_score = hue_std / 90.0
    saturation_score = saturation_std / 128.0
    value_score = value_std / 128.0

    color_variation = (
        hue_score
        + saturation_score
        + value_score
    ) / 3.0

    return float(
        min(max(color_variation, 0.0), 1.0)
    )


# ---------------------------------------------------------
# D — Diameter
# ---------------------------------------------------------

def calculate_diameter(binary_mask):

    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0:
        return 0.0

    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1

    # Approximate diameter using the longest bounding-box axis
    diameter_pixels = max(width, height)

    return float(diameter_pixels)


# ---------------------------------------------------------
# Calculate all ABCD measurements
# ---------------------------------------------------------

def calculate_abcde(image, segmentation_mask):

    # Convert original image to RGB
    image = image.convert("RGB")

    # Convert mask to binary
    binary_mask = mask_to_binary(segmentation_mask)

    # CRITICAL FIX:
    # Resize the predicted mask from 288 x 288
    # to the original image dimensions when needed.
    binary_mask = resize_mask_to_image(
        binary_mask,
        image
    )

    # Calculate lesion area in the resized mask
    lesion_area = int(np.sum(binary_mask))

    if lesion_area == 0:
        return {
            "A_asymmetry": 0.0,
            "B_border_irregularity": 0.0,
            "C_color_variation": 0.0,
            "D_diameter_pixels": 0.0,
            "lesion_area_pixels": 0
        }

    # A — Asymmetry
    asymmetry = calculate_asymmetry(
        binary_mask
    )

    # B — Border irregularity
    border_irregularity = calculate_border_irregularity(
        binary_mask
    )

    # C — Color variation
    color_variation = calculate_color_variation(
        image,
        binary_mask
    )

    # D — Diameter
    diameter = calculate_diameter(
        binary_mask
    )

    return {
        "A_asymmetry": asymmetry,
        "B_border_irregularity": border_irregularity,
        "C_color_variation": color_variation,
        "D_diameter_pixels": diameter,
        "lesion_area_pixels": lesion_area
    }