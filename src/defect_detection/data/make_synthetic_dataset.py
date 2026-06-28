from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

CATEGORIES = [
    {"id": 1, "name": "crack"},
    {"id": 2, "name": "scratch"},
    {"id": 3, "name": "stain"},
]


def random_bg(size: int) -> Image.Image:
    base = random.randint(170, 220)
    image = Image.new("RGB", (size, size), (base, base, base))
    draw = ImageDraw.Draw(image)
    # subtle industrial-like texture
    for _ in range(1200):
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        delta = random.randint(-18, 18)
        val = max(0, min(255, base + delta))
        draw.point((x, y), fill=(val, val, val))
    return image.filter(ImageFilter.GaussianBlur(radius=0.25))


def clamp_box(x1, y1, x2, y2, size):
    x1 = max(0, min(size - 2, x1))
    y1 = max(0, min(size - 2, y1))
    x2 = max(x1 + 2, min(size - 1, x2))
    y2 = max(y1 + 2, min(size - 1, y2))
    return [x1, y1, x2, y2]


def draw_crack(draw: ImageDraw.ImageDraw, size: int):
    margin = max(8, size // 12)
    x = random.randint(margin, max(margin, size - margin - 1))
    y = random.randint(margin, max(margin, size - margin - 1))
    points = [(x, y)]
    step = max(8, size // 16)
    for _ in range(random.randint(4, 8)):
        x += random.randint(-step, int(step * 1.5))
        y += random.randint(-step, int(step * 1.2))
        points.append((max(0, min(size - 1, x)), max(0, min(size - 1, y))))
    width = random.randint(2, max(2, size // 100))
    color = tuple(random.randint(20, 60) for _ in range(3))
    draw.line(points, fill=color, width=width)
    xs, ys = zip(*points)
    pad = width + 4
    return clamp_box(min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad, size), 1


def draw_scratch(draw: ImageDraw.ImageDraw, size: int):
    margin = max(8, size // 12)
    x1 = random.randint(margin, max(margin, size - margin - 1))
    y1 = random.randint(margin, max(margin, size - margin - 1))
    length = random.randint(max(10, size // 8), max(12, size // 3))
    dx = random.randint(-max(4, size // 16), max(4, size // 16))
    x2 = max(0, min(size - 1, x1 + length))
    y2 = max(0, min(size - 1, y1 + dx))
    width = random.randint(1, 3)
    color = tuple(random.randint(80, 130) for _ in range(3))
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    pad = width + 3
    return clamp_box(min(x1, x2) - pad, min(y1, y2) - pad, max(x1, x2) + pad, max(y1, y2) + pad, size), 2


def draw_stain(draw: ImageDraw.ImageDraw, size: int):
    margin = max(8, size // 14)
    w = random.randint(max(8, size // 20), max(12, size // 5))
    h = random.randint(max(8, size // 22), max(12, size // 6))
    x1 = random.randint(margin, max(margin, size - w - margin))
    y1 = random.randint(margin, max(margin, size - h - margin))
    x2 = x1 + w
    y2 = y1 + h
    color = (
        random.randint(95, 150),
        random.randint(85, 135),
        random.randint(65, 110),
    )
    draw.ellipse([x1, y1, x2, y2], fill=color)
    return clamp_box(x1, y1, x2, y2, size), 3


def make_split(output: Path, split: str, n_images: int, image_size: int, start_ann_id: int = 1):
    images_dir = output / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = start_ann_id

    for i in range(n_images):
        image_id = i + 1
        file_name = f"{split}_{i:04d}.png"
        image = random_bg(image_size)
        draw = ImageDraw.Draw(image, "RGB")

        n_defects = random.randint(1, 5)
        for _ in range(n_defects):
            defect_type = random.choice(["crack", "scratch", "stain"])
            if defect_type == "crack":
                box, cat_id = draw_crack(draw, image_size)
            elif defect_type == "scratch":
                box, cat_id = draw_scratch(draw, image_size)
            else:
                box, cat_id = draw_stain(draw, image_size)

            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
            })
            ann_id += 1

        image.save(images_dir / file_name)
        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": image_size,
            "height": image_size,
        })

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    with open(output / split / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    return ann_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/synthetic")
    parser.add_argument("--train-size", type=int, default=120)
    parser.add_argument("--val-size", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    next_ann = make_split(output, "train", args.train_size, args.image_size, start_ann_id=1)
    make_split(output, "val", args.val_size, args.image_size, start_ann_id=next_ann)

    print(f"Synthetic dataset created at: {output}")
    print(f"Train images: {args.train_size} | Val images: {args.val_size}")


if __name__ == "__main__":
    main()
