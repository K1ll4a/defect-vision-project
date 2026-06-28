from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F

from defect_detection.models import load_checkpoint
from defect_detection.utils import ensure_dir, get_device


def draw_predictions(
    image: Image.Image,
    predictions: dict[str, torch.Tensor],
    class_names: list[str] | None,
    score_threshold: float = 0.4,
) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    boxes = predictions["boxes"].detach().cpu()
    labels = predictions["labels"].detach().cpu()
    scores = predictions["scores"].detach().cpu()

    for box, label, score in zip(boxes, labels, scores):
        if float(score) < score_threshold:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        label_idx = int(label)
        label_name = class_names[label_idx] if class_names and label_idx < len(class_names) else str(label_idx)
        text = f"{label_name}: {float(score):.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((x1, y1), text, fill="white", font=font)

    return image


def predict_image(weights: str, image_path: str, output_path: str | None = None, score_threshold: float = 0.4, device_str: str = "auto"):
    device = get_device(device_str)
    model, class_names, _ = load_checkpoint(weights, device=device)

    image = Image.open(image_path).convert("RGB")
    tensor = F.to_tensor(image).to(device)

    with torch.no_grad():
        prediction = model([tensor])[0]

    result = draw_predictions(image.copy(), prediction, class_names, score_threshold=score_threshold)

    if output_path is not None:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        result.save(output_path)
        print(f"Saved: {output_path}")

    return result, prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/prediction.png")
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    predict_image(
        weights=args.weights,
        image_path=args.image,
        output_path=args.output,
        score_threshold=args.score_threshold,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
