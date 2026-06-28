from __future__ import annotations

import random
from typing import Callable

import torch
from PIL import Image
from torchvision.transforms import functional as F


class DetectionToTensor:
    """Convert PIL image to float tensor in [0, 1]."""

    def __call__(self, image: Image.Image, target: dict):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    """Horizontal flip with box coordinate update."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, target: dict):
        if random.random() >= self.p:
            return image, target

        width, _ = image.size
        image = F.hflip(image)
        boxes = target["boxes"].clone()
        if boxes.numel() > 0:
            x_min = width - boxes[:, 2]
            x_max = width - boxes[:, 0]
            boxes[:, 0] = x_min
            boxes[:, 2] = x_max
            target["boxes"] = boxes
        return image, target


class Compose:
    def __init__(self, transforms: list[Callable]):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: dict):
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


def get_train_transforms() -> Compose:
    return Compose([
        RandomHorizontalFlip(p=0.5),
        DetectionToTensor(),
    ])


def get_val_transforms() -> Compose:
    return Compose([
        DetectionToTensor(),
    ])
