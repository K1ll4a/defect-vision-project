from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

cache_dir = Path(os.getenv("TMPDIR", "/tmp")) / "defect_vision_matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
(cache_dir / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


LOSS_SERIES = [
    ("train_loss", "Train loss"),
    ("val_loss", "Validation loss"),
    ("test_loss", "Test loss"),
]


def read_metrics(metrics_path: str | Path) -> list[dict[str, Any]]:
    path = Path(metrics_path)
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def plot_loss_curves(metrics_path: str | Path, output_path: str | Path) -> Path:
    records = read_metrics(metrics_path)
    if not records:
        raise ValueError(f"No metric records found in {metrics_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [int(record["epoch"]) for record in records]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)

    plotted = 0
    for key, label in LOSS_SERIES:
        values = [record.get(key) for record in records]
        points = [(epoch, value) for epoch, value in zip(epochs, values) if value is not None]
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
        plotted += 1

    if plotted == 0:
        raise ValueError(f"No loss fields found in {metrics_path}. Expected one of: train_loss, val_loss, test_loss")

    ax.set_title("Training Progress")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    ax.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, default="runs/defect_faster_rcnn/metrics.jsonl")
    parser.add_argument("--output", type=str, default="runs/defect_faster_rcnn/loss_curve.png")
    args = parser.parse_args()

    output_path = plot_loss_curves(args.metrics, args.output)
    print(f"Saved loss curve: {output_path}")


if __name__ == "__main__":
    main()
