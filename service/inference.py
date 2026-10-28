
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from torchvision import transforms

from models.ca_ynet import CAYNet
from service.grad_cam import GradCAM, create_heatmap
from service.abcde import calculate_abcde


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "training"
    / "dermai_best_model.pth"
)

IMAGE_SIZE = (288, 288)

CLASS_NAMES = {
    0: "Nevus",
    1: "Melanoma",
    2: "Seborrheic Keratosis"
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------
# Image transformation
# ---------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# Load trained CA-YNet
# ---------------------------------------------------------

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}\n\n"
            "Please check whether training has completed."
        )

    print("Loading CA-YNet model...")

    model = CAYNet(
        num_classes=3,
        pretrained_encoder=False
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["student_model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    print("CA-YNet model loaded successfully.")
    print("Device:", DEVICE)

    return model


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict(model, image_path):

    # ---------------------------------------------
    # 1. Load original image
    # ---------------------------------------------

    with Image.open(image_path) as opened_image:
        image = opened_image.convert("RGB")

    original_width, original_height = image.size

    # ---------------------------------------------
    # 2. Prepare model input
    # ---------------------------------------------

    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    # ---------------------------------------------
    # 3. Classification and segmentation
    # ---------------------------------------------

    # Gradients are not needed for this initial pass.
    with torch.no_grad():

        outputs = model(image_tensor)

        class_logits = outputs["class_logits"]

        probabilities = F.softmax(
            class_logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

        disease = CLASS_NAMES[predicted_class]

        segmentation_logits = outputs["seg_logits"]

        segmentation_probability = torch.sigmoid(
            segmentation_logits
        )

        segmentation_mask = (
            segmentation_probability > 0.5
        ).float()

    # ---------------------------------------------
    # 4. Create segmentation mask
    # ---------------------------------------------

    mask = (
        segmentation_mask[0, 0]
        .cpu()
        .numpy()
        * 255
    ).astype(np.uint8)

    # The mask initially has size 288 x 288.
    mask_image = Image.fromarray(
        mask,
        mode="L"
    )

    lesion_pixels = int(
        segmentation_mask.sum().item()
    )

    # ---------------------------------------------
    # 5. ABCDE analysis
    # ---------------------------------------------

    abcde = calculate_abcde(
        image=image,
        segmentation_mask=mask_image
    )

    # ---------------------------------------------
    # 6. Generate Grad-CAM
    # ---------------------------------------------

    target_layer = model.encoder.layer4[-1].conv2

    grad_cam = GradCAM(
        model,
        target_layer
    )

    try:

        cam = grad_cam.generate(
            image_tensor,
            class_index=predicted_class
        )

        # Resize the CAM and overlay it only inside
        # the lesion identified by segmentation.
        heatmap = create_heatmap(
            cam=cam,
            original_image=image,
            lesion_mask=mask_image,
            alpha=0.45
        )

    finally:

        grad_cam.close()

    # ---------------------------------------------
    # 7. Return results
    # ---------------------------------------------

    return {
        "disease": disease,
        "class_id": predicted_class,
        "confidence": confidence,

        "original_image_size": [
            original_width,
            original_height
        ],

        "lesion_pixels": lesion_pixels,
        "abcde": abcde,

        "segmentation_mask": mask_image,
        "gradcam_heatmap": heatmap
    }


# ---------------------------------------------------------
# Direct execution
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("DERMAI+ ML INFERENCE")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Model path:", MODEL_PATH)

    if not MODEL_PATH.exists():

        print("Model checkpoint does not exist.")
        print("Please check your training output.")

    else:

        model = load_model()

        print()
        print("Model is ready for inference.")