from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


class COCODefectDataset(Dataset):
    """COCO-style object detection dataset.

    Expected bbox format in annotations: [x, y, width, height].
    Returned bbox format for torchvision detection models: [x_min, y_min, x_max, y_max].
    """

    def __init__(self, images_dir: str | Path, annotations_path: str | Path, transforms=None):
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.transforms = transforms

        with open(self.annotations_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = sorted(coco["images"], key=lambda x: x["id"])
        self.categories = sorted(coco.get("categories", []), key=lambda x: x["id"])
        self.cat_id_to_label = {cat["id"]: idx + 1 for idx, cat in enumerate(self.categories)}
        self.label_to_name = {idx + 1: cat["name"] for idx, cat in enumerate(self.categories)}
        self.label_to_name[0] = "__background__"

        self.annotations_by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in coco.get("annotations", []):
            if ann.get("iscrowd", 0) == 1:
                continue
            self.annotations_by_image_id[ann["image_id"]].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    @property
    def class_names(self) -> list[str]:
        names = ["__background__"]
        names.extend([cat["name"] for cat in self.categories])
        return names

    def __getitem__(self, idx: int):
        image_info = self.images[idx]
        image_id = int(image_info["id"])
        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in self.annotations_by_image_id.get(image_id, []):
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[int(ann["category_id"])])
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id], dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if target["boxes"].numel() == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
