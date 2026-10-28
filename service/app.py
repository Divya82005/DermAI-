from pathlib import Path
import os
import tempfile

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse

from service.inference import load_model, predict


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="DermAI+ ML Service",
    description="Machine learning inference service for DermAI+",
    version="1.0.0"
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = None


# ---------------------------------------------------------
# Output directory
# ---------------------------------------------------------

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "UP",
        "service": "DermAI+ ML Service"
    }


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...)
):

    global model

    # ---------------------------------------------
    # Validate file
    # ---------------------------------------------

    if not file.filename:

        return JSONResponse(
            status_code=400,
            content={
                "error": "No image file provided."
            }
        )

    # ---------------------------------------------
    # Load model
    # ---------------------------------------------

    try:

        if model is None:

            model = load_model()

    except FileNotFoundError as error:

        return JSONResponse(
            status_code=503,
            content={
                "error": str(error)
            }
        )

    # ---------------------------------------------
    # Temporary uploaded image
    # ---------------------------------------------

    suffix = Path(
        file.filename
    ).suffix

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_path = temp_file.name

    try:

        contents = await file.read()

        temp_file.write(contents)

        temp_file.close()

        # -----------------------------------------
        # Run inference
        # -----------------------------------------

        result = predict(
            model,
            temp_path
        )

        # -----------------------------------------
        # Create output filenames
        # -----------------------------------------

        image_name = Path(
            file.filename
        ).stem

        segmentation_filename = (
            image_name
            + "_segmentation.png"
        )

        gradcam_filename = (
            image_name
            + "_gradcam.png"
        )

        # -----------------------------------------
        # Output paths
        # -----------------------------------------

        segmentation_path = (
            OUTPUT_DIR
            / segmentation_filename
        )

        gradcam_path = (
            OUTPUT_DIR
            / gradcam_filename
        )

        # -----------------------------------------
        # Save segmentation
        # -----------------------------------------

        result[
            "segmentation_mask"
        ].save(
            segmentation_path
        )

        # -----------------------------------------
        # Save Grad-CAM
        # -----------------------------------------

        result[
            "gradcam_heatmap"
        ].save(
            gradcam_path
        )

        # -----------------------------------------
        # Remove image objects
        # -----------------------------------------

        result.pop(
            "segmentation_mask"
        )

        result.pop(
            "gradcam_heatmap"
        )

        # -----------------------------------------
        # Return complete response
        # -----------------------------------------

        return {

            "filename": file.filename,

            "prediction": result,

            "segmentation_mask_url": (
                f"/segmentation/"
                f"{segmentation_filename}"
            ),

            "gradcam_url": (
                f"/gradcam/"
                f"{gradcam_filename}"
            )
        }

    except Exception as error:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(error)
            }
        )

    finally:

        if os.path.exists(temp_path):

            os.remove(temp_path)


# ---------------------------------------------------------
# Segmentation endpoint
# ---------------------------------------------------------

@app.get("/segmentation/{filename}")
def get_segmentation(
    filename: str
):

    file_path = (
        OUTPUT_DIR
        / filename
    )

    if not file_path.exists():

        return JSONResponse(
            status_code=404,
            content={
                "error": "Segmentation mask not found."
            }
        )

    return FileResponse(
        file_path,
        media_type="image/png"
    )


# ---------------------------------------------------------
# Grad-CAM endpoint
# ---------------------------------------------------------

@app.get("/gradcam/{filename}")
def get_gradcam(
    filename: str
):

    file_path = (
        OUTPUT_DIR
        / filename
    )

    if not file_path.exists():

        return JSONResponse(
            status_code=404,
            content={
                "error": "Grad-CAM heatmap not found."
            }
        )

    return FileResponse(
        file_path,
        media_type="image/png"
    )