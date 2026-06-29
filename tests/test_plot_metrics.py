from __future__ import annotations

import json
from pathlib import Path

from defect_detection.training.plot_metrics import plot_loss_curves


def test_plot_loss_curves_creates_png(tmp_path: Path):
    metrics_path = tmp_path / "metrics.jsonl"
    output_path = tmp_path / "loss_curve.png"
    records = [
        {"epoch": 1, "train_loss": 0.5, "test_loss": 0.6},
        {"epoch": 2, "train_loss": 0.3, "test_loss": 0.4},
    ]
    with metrics_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    result = plot_loss_curves(metrics_path, output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
