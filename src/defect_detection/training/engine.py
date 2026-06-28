from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from defect_detection.data import COCODefectDataset, get_train_transforms, get_val_transforms
from defect_detection.models import build_faster_rcnn
from defect_detection.training.evaluate import evaluate_detector, evaluate_loss
from defect_detection.training.plot_metrics import plot_loss_artifacts
from defect_detection.utils import collate_fn, ensure_dir, get_device, load_yaml, move_targets_to_device, save_jsonl, set_seed


def train_one_epoch(model, data_loader, optimizer, device, scaler=None, amp: bool = True) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    progress = tqdm(data_loader, desc="train", leave=False)
    for images, targets in progress:
        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        use_amp = amp and device.type == "cuda"
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        if scaler is not None and use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        loss_value = float(losses.detach().cpu().item())
        running_loss += loss_value
        n_batches += 1
        progress.set_postfix(loss=f"{loss_value:.4f}")

    return running_loss / max(n_batches, 1)


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, metrics: dict, config: dict, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config,
            "class_names": class_names,
        },
        path,
    )


def main(config_path: str) -> None:
    config = load_yaml(config_path)
    set_seed(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    output_dir = ensure_dir(config.get("output_dir", "runs/defect_faster_rcnn"))

    train_ds = COCODefectDataset(
        images_dir=config["data"]["train_images"],
        annotations_path=config["data"]["train_annotations"],
        transforms=get_train_transforms(),
    )
    val_ds = COCODefectDataset(
        images_dir=config["data"]["val_images"],
        annotations_path=config["data"]["val_annotations"],
        transforms=get_val_transforms(),
    )
    test_images = config["data"].get("test_images")
    test_annotations = config["data"].get("test_annotations")
    if test_images and test_annotations and Path(test_images).exists() and Path(test_annotations).exists():
        test_ds = COCODefectDataset(
            images_dir=test_images,
            annotations_path=test_annotations,
            transforms=get_val_transforms(),
        )
        test_split_name = "test"
    else:
        test_ds = val_ds
        test_split_name = "val"

    class_names = config.get("class_names") or train_ds.class_names
    num_classes = int(config.get("num_classes", len(class_names)))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 2)),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 2)),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 2)),
        collate_fn=collate_fn,
    )

    model = build_faster_rcnn(
        num_classes=num_classes,
        pretrained=bool(config["model"].get("pretrained", True)),
        trainable_backbone_layers=config["model"].get("trainable_backbone_layers", 3),
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(config["train"]["lr"]),
        momentum=float(config["train"].get("momentum", 0.9)),
        weight_decay=float(config["train"].get("weight_decay", 0.0005)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(config["train"].get("lr_step_size", 3)),
        gamma=float(config["train"].get("lr_gamma", 0.1)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and bool(config["train"].get("amp", True))))

    best_map50 = -1.0
    epochs = int(config["train"]["epochs"])
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Train images: {len(train_ds)} | Val images: {len(val_ds)} | Test images: {len(test_ds)} ({test_split_name})")

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            amp=bool(config["train"].get("amp", True)),
        )
        scheduler.step()

        test_loss = evaluate_loss(
            model,
            test_loader,
            device=device,
            amp=bool(config["train"].get("amp", True)),
        )
        metrics = evaluate_detector(
            model,
            val_loader,
            device=device,
            score_threshold=float(config["eval"].get("score_threshold", 0.35)),
            iou_threshold=float(config["eval"].get("iou_threshold", 0.5)),
            num_classes=num_classes,
        )
        metrics.update({
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "test_split": test_split_name,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - start, 2),
        })
        save_jsonl(metrics_path, metrics)
        plot_loss_artifacts(metrics_path, output_dir)

        print(json.dumps(metrics, indent=2))

        save_checkpoint(output_dir / "last.pt", model, optimizer, scheduler, epoch, metrics, config, class_names)
        if metrics["map50"] > best_map50:
            best_map50 = metrics["map50"]
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler, epoch, metrics, config, class_names)
            print(f"Saved new best checkpoint: mAP@0.5={best_map50:.4f}")

    print(f"Done. Best mAP@0.5: {best_map50:.4f}")
    print(f"Artifacts: {output_dir}")
    print(f"Loss plots: {output_dir / 'train_loss_curve.png'} | {output_dir / 'test_loss_curve.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/faster_rcnn.yaml")
    args = parser.parse_args()
    main(args.config)
