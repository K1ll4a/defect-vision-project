from __future__ import annotations

import os
from io import BytesIO

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision.transforms import functional as F

from defect_detection.model import load_checkpoint
from defect_detection.utils import get_device

app = FastAPI(title="DefectVision API", version="0.1.0")

WEIGHTS_PATH = os.getenv("DEFECTVISION_WEIGHTS", "runs/defect_faster_rcnn/best.pt")
DEVICE = os.getenv("DEFECTVISION_DEVICE", "auto")
SCORE_THRESHOLD = float(os.getenv("DEFECTVISION_SCORE_THRESHOLD", "0.4"))

_model = None
_class_names = None
_device = None


def get_model():
    global _model, _class_names, _device
    if _model is None:
        _device = get_device(DEVICE)
        _model, _class_names, _ = load_checkpoint(WEIGHTS_PATH, device=_device)
    return _model, _class_names, _device


@app.get("/health")
def health():
    return {"status": "ok", "weights": WEIGHTS_PATH, "device": DEVICE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model, class_names, device = get_model()
    content = await file.read()
    image = Image.open(BytesIO(content)).convert("RGB")
    tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        output = model([tensor])[0]

    detections = []
    for label, score, box in zip(output["labels"], output["scores"], output["boxes"]):
        score_value = float(score.detach().cpu().item())
        if score_value < SCORE_THRESHOLD:
            continue
        label_idx = int(label.detach().cpu().item())
        detections.append({
            "class_id": label_idx,
            "class_name": class_names[label_idx] if class_names and label_idx < len(class_names) else str(label_idx),
            "score": score_value,
            "box_xyxy": [float(x) for x in box.detach().cpu().tolist()],
        })

    return {"detections": detections}
