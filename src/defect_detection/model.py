from __future__ import annotations

import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int | None = 3,
) -> torch.nn.Module:
    """Build Faster R-CNN with a replaced classification head.

    Args:
        num_classes: Number of classes including background.
        pretrained: Use COCO-pretrained weights if True.
        trainable_backbone_layers: Number of trainable ResNet backbone layers.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def load_checkpoint(weights_path: str, device: torch.device):
    checkpoint = torch.load(weights_path, map_location=device)
    class_names = checkpoint.get("class_names")
    config = checkpoint.get("config", {})
    num_classes = len(class_names) if class_names else config.get("num_classes")
    if num_classes is None:
        raise ValueError("Cannot infer num_classes from checkpoint. Save class_names or config in checkpoint.")
    model = build_faster_rcnn(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names, config
