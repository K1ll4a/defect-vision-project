from __future__ import annotations

import torch
from tqdm import tqdm

from defect_detection.utils import move_targets_to_device


def evaluate_loss(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    amp: bool = True,
) -> float:
    """Compute validation loss for torchvision detection models.

    Torchvision detection models return losses only in training mode when targets
    are supplied. Gradients stay disabled here, so model weights are not updated.
    """
    was_training = model.training
    model.train()

    running_loss = 0.0
    n_batches = 0
    use_amp = amp and device.type == "cuda"

    try:
        with torch.no_grad():
            for images, targets in tqdm(data_loader, desc="test_loss", leave=False):
                images = [img.to(device) for img in images]
                targets = move_targets_to_device(targets, device)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                running_loss += float(losses.detach().cpu().item())
                n_batches += 1
    finally:
        model.train(was_training)

    return running_loss / max(n_batches, 1)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def evaluate_detector(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    score_threshold: float = 0.35,
    iou_threshold: float = 0.5,
    num_classes: int | None = None,
) -> dict[str, float]:
    """Lightweight detection evaluation.

    Calculates class-averaged AP at IoU=0.5 using greedy matching.
    This is not a full COCO evaluator, but it is enough for a clean pet-project MVP.
    """
    model.eval()
    if num_classes is None:
        num_classes = 100

    predictions_by_class = {c: [] for c in range(1, num_classes)}
    gt_by_class = {c: {} for c in range(1, num_classes)}

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="eval", leave=False):
            images = [img.to(device) for img in images]
            targets_device = move_targets_to_device(targets, device)
            outputs = model(images)

            for target, output in zip(targets_device, outputs):
                image_id = int(target["image_id"].item())

                gt_boxes = target["boxes"]
                gt_labels = target["labels"]
                for cls in range(1, num_classes):
                    cls_boxes = gt_boxes[gt_labels == cls]
                    gt_by_class[cls][image_id] = {
                        "boxes": cls_boxes,
                        "matched": torch.zeros(len(cls_boxes), dtype=torch.bool, device=device),
                    }

                keep = output["scores"] >= score_threshold
                pred_boxes = output["boxes"][keep]
                pred_labels = output["labels"][keep]
                pred_scores = output["scores"][keep]

                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    cls = int(label.item())
                    if cls in predictions_by_class:
                        predictions_by_class[cls].append((image_id, float(score.item()), box))

    aps = []
    total_tp = total_fp = total_fn = 0

    for cls in range(1, num_classes):
        preds = sorted(predictions_by_class[cls], key=lambda x: x[1], reverse=True)
        n_gt = sum(len(v["boxes"]) for v in gt_by_class[cls].values())
        if n_gt == 0:
            continue

        tp = torch.zeros(len(preds))
        fp = torch.zeros(len(preds))

        for i, (image_id, score, pred_box) in enumerate(preds):
            gt_info = gt_by_class[cls].get(image_id)
            if gt_info is None or len(gt_info["boxes"]) == 0:
                fp[i] = 1
                continue

            ious = box_iou(pred_box.unsqueeze(0), gt_info["boxes"]).squeeze(0)
            best_iou, best_idx = (ious.max(0) if len(ious) else (torch.tensor(0.0, device=pred_box.device), torch.tensor(0, device=pred_box.device)))

            if best_iou >= iou_threshold and not gt_info["matched"][best_idx]:
                tp[i] = 1
                gt_info["matched"][best_idx] = True
            else:
                fp[i] = 1

        total_tp += int(tp.sum().item())
        total_fp += int(fp.sum().item())
        total_fn += int(n_gt - tp.sum().item())

        if len(preds) == 0:
            aps.append(0.0)
            continue

        cum_tp = torch.cumsum(tp, dim=0)
        cum_fp = torch.cumsum(fp, dim=0)
        recall = cum_tp / max(n_gt, 1)
        precision = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-6)

        # 11-point interpolated AP@0.5
        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            if torch.any(recall >= t):
                ap += torch.max(precision[recall >= t]).item()
        ap /= 11.0
        aps.append(ap)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    map50 = sum(aps) / max(len(aps), 1)

    return {
        "map50": float(map50),
        "precision": float(precision),
        "recall": float(recall),
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
    }
