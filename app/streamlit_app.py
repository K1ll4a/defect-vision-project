from __future__ import annotations

import argparse
from io import BytesIO

import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as F

from defect_detection.model import load_checkpoint
from defect_detection.predict import draw_predictions
from defect_detection.utils import get_device


@st.cache_resource
def load_model_cached(weights: str, device_str: str):
    device = get_device(device_str)
    model, class_names, _ = load_checkpoint(weights, device=device)
    return model, class_names, device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/defect_faster_rcnn/best.pt")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="DefectVision", layout="wide")
    st.title("DefectVision — PyTorch defect detection")
    st.write("Upload an image and detect visual defects with Faster R-CNN.")

    score_threshold = st.sidebar.slider("Score threshold", 0.0, 1.0, 0.4, 0.05)
    weights = st.sidebar.text_input("Checkpoint path", args.weights)
    device_str = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)

    uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to run detection.")
        return

    image = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)
    col1.subheader("Input")
    col1.image(image, use_container_width=True)

    model, class_names, device = load_model_cached(weights, device_str)
    tensor = F.to_tensor(image).to(device)
    with torch.no_grad():
        prediction = model([tensor])[0]

    result = draw_predictions(image.copy(), prediction, class_names, score_threshold=score_threshold)
    col2.subheader("Prediction")
    col2.image(result, use_container_width=True)

    rows = []
    for label, score, box in zip(prediction["labels"], prediction["scores"], prediction["boxes"]):
        if float(score) < score_threshold:
            continue
        label_idx = int(label)
        rows.append({
            "class": class_names[label_idx] if class_names and label_idx < len(class_names) else str(label_idx),
            "score": round(float(score), 4),
            "box_xyxy": [round(float(x), 1) for x in box.detach().cpu().tolist()],
        })

    st.subheader("Detections")
    st.dataframe(rows, use_container_width=True)

    buf = BytesIO()
    result.save(buf, format="PNG")
    st.download_button("Download prediction", buf.getvalue(), file_name="prediction.png", mime="image/png")


if __name__ == "__main__":
    main()
