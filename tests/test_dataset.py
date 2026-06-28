from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from defect_detection.dataset import COCODefectDataset
from defect_detection.transforms import get_train_transforms


def test_synthetic_dataset_loads(tmp_path: Path):
    data_dir = tmp_path / "synthetic"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_synthetic_dataset.py",
            "--output",
            str(data_dir),
            "--train-size",
            "3",
            "--val-size",
            "1",
            "--image-size",
            "128",
        ],
        check=True,
    )

    ds = COCODefectDataset(
        images_dir=data_dir / "train" / "images",
        annotations_path=data_dir / "train" / "annotations.json",
        transforms=get_train_transforms(),
    )
    image, target = ds[0]
    assert image.shape[0] == 3
    assert target["boxes"].ndim == 2
    assert target["boxes"].shape[1] == 4
    assert target["labels"].ndim == 1
