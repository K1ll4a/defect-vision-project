from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field
from torchvision.transforms import functional as F

from defect_detection.models import load_checkpoint
from defect_detection.utils import get_device

API_DESCRIPTION = """
DefectVision exposes a trained Faster R-CNN model as an HTTP inference service.

Upload an RGB image to `/predict` and the API returns detected visual defects as
bounding boxes with class names and confidence scores.

Runtime configuration:

* `DEFECTVISION_WEIGHTS` - checkpoint path, defaults to `runs/defect_faster_rcnn/best.pt`
* `DEFECTVISION_DEVICE` - `auto`, `cpu`, or `cuda`
* `DEFECTVISION_SCORE_THRESHOLD` - default confidence threshold for detections
"""

OPENAPI_TAGS = [
    {
        "name": "service",
        "description": "Service metadata and readiness checks.",
    },
    {
        "name": "inference",
        "description": "Computer vision inference endpoints.",
    },
]

app = FastAPI(
    title="DefectVision API",
    summary="Visual defect detection inference API",
    description=API_DESCRIPTION,
    version="0.1.0",
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={"displayRequestDuration": True},
)

WEIGHTS_PATH = os.getenv("DEFECTVISION_WEIGHTS", "runs/defect_faster_rcnn/best.pt")
DEVICE = os.getenv("DEFECTVISION_DEVICE", "auto")
SCORE_THRESHOLD = float(os.getenv("DEFECTVISION_SCORE_THRESHOLD", "0.4"))

_model = None
_class_names = None
_device = None


class ServiceInfoResponse(BaseModel):
    name: str = Field(description="API service name.")
    version: str = Field(description="API version.")
    docs_url: str = Field(description="Swagger UI documentation URL.")
    redoc_url: str = Field(description="ReDoc documentation URL.")
    openapi_url: str = Field(description="Raw OpenAPI schema URL.")
    endpoints: dict[str, str] = Field(description="Main API endpoints.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "DefectVision API",
                "version": "0.1.0",
                "docs_url": "/docs",
                "redoc_url": "/redoc",
                "openapi_url": "/openapi.json",
                "endpoints": {
                    "health": "/health",
                    "predict": "/predict",
                },
            }
        }
    }


class HealthResponse(BaseModel):
    status: str = Field(description="Service status.")
    weights: str = Field(description="Checkpoint path used by the API.")
    weights_exists: bool = Field(description="Whether the checkpoint file exists.")
    device: str = Field(description="Configured inference device.")
    score_threshold: float = Field(description="Default detection confidence threshold.", ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "weights": "runs/defect_faster_rcnn/best.pt",
                "weights_exists": True,
                "device": "auto",
                "score_threshold": 0.4,
            }
        }
    }


class Detection(BaseModel):
    class_id: int = Field(description="Numeric class id predicted by the model.", examples=[2])
    class_name: str = Field(description="Human-readable defect class name.", examples=["scratch"])
    score: float = Field(description="Model confidence score.", ge=0.0, le=1.0, examples=[0.9668])
    box_xyxy: list[float] = Field(
        description="Bounding box coordinates in [x_min, y_min, x_max, y_max] format.",
        min_length=4,
        max_length=4,
        examples=[[410.3, 32.9, 512.0, 46.7]],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "class_id": 2,
                "class_name": "scratch",
                "score": 0.9668,
                "box_xyxy": [410.3, 32.9, 512.0, 46.7],
            }
        }
    }


class PredictionResponse(BaseModel):
    filename: str | None = Field(description="Original uploaded filename.", examples=["val_0000.png"])
    width: int = Field(description="Input image width in pixels.", examples=[512])
    height: int = Field(description="Input image height in pixels.", examples=[512])
    score_threshold: float = Field(description="Confidence threshold used for this request.", ge=0.0, le=1.0)
    detections: list[Detection] = Field(description="Detected defects after score filtering.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "filename": "val_0000.png",
                "width": 512,
                "height": 512,
                "score_threshold": 0.4,
                "detections": [
                    {
                        "class_id": 2,
                        "class_name": "scratch",
                        "score": 0.9668,
                        "box_xyxy": [410.3, 32.9, 512.0, 46.7],
                    }
                ],
            }
        }
    }


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error description.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Uploaded file is not a valid image.",
            }
        }
    }


def get_model():
    global _model, _class_names, _device
    if _model is None:
        if not Path(WEIGHTS_PATH).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model weights were not found at {WEIGHTS_PATH!r}. Set DEFECTVISION_WEIGHTS or train the model first.",
            )
        _device = get_device(DEVICE)
        try:
            _model, _class_names, _ = load_checkpoint(WEIGHTS_PATH, device=_device)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to load model checkpoint: {exc}") from exc
    return _model, _class_names, _device


@app.get(
    "/",
    response_model=ServiceInfoResponse,
    tags=["service"],
    summary="Show API entrypoint",
    description="Returns the main API documentation links and endpoint paths.",
)
def root():
    return {
        "name": app.title,
        "version": app.version,
        "docs_url": app.docs_url,
        "redoc_url": app.redoc_url,
        "openapi_url": app.openapi_url,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["service"],
    summary="Check service health",
    description="Returns service configuration and whether the configured model checkpoint exists.",
)
def health():
    return {
        "status": "ok",
        "weights": WEIGHTS_PATH,
        "weights_exists": Path(WEIGHTS_PATH).exists(),
        "device": DEVICE,
        "score_threshold": SCORE_THRESHOLD,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Detect defects in an image",
    description=(
        "Accepts a JPEG or PNG image as multipart form-data and returns detected "
        "defect bounding boxes filtered by confidence score."
    ),
    responses={
        400: {
            "model": ErrorResponse,
            "description": "The uploaded file cannot be decoded as an image.",
        },
        500: {
            "model": ErrorResponse,
            "description": "The model checkpoint exists but could not be loaded.",
        },
        503: {
            "model": ErrorResponse,
            "description": "The configured model checkpoint is missing.",
        },
    },
)
async def predict(
    file: UploadFile = File(..., description="Input image file. Supported formats include PNG and JPEG."),
    score_threshold: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Optional confidence threshold for this request. "
            "If omitted, DEFECTVISION_SCORE_THRESHOLD is used."
        ),
        examples=[0.4],
    ),
):
    model, class_names, device = get_model()
    content = await file.read()
    try:
        image = Image.open(BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    threshold = SCORE_THRESHOLD if score_threshold is None else score_threshold
    tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    detections = []
    for label, score, box in zip(output["labels"], output["scores"], output["boxes"]):
        score_value = float(score.detach().cpu().item())
        if score_value < threshold:
            continue
        label_idx = int(label.detach().cpu().item())
        detections.append({
            "class_id": label_idx,
            "class_name": class_names[label_idx] if class_names and label_idx < len(class_names) else str(label_idx),
            "score": score_value,
            "box_xyxy": [float(x) for x in box.detach().cpu().tolist()],
        })

    return {
        "filename": file.filename,
        "width": image.width,
        "height": image.height,
        "score_threshold": threshold,
        "detections": detections,
    }
