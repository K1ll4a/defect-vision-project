from __future__ import annotations

import argparse
from pathlib import Path

import torch

from defect_detection.models import load_checkpoint
from defect_detection.utils import ensure_dir, get_device


def export_torchscript(weights: str, output: str, image_size: int, device_str: str = "cpu") -> None:
    device = get_device(device_str)
    model, _, _ = load_checkpoint(weights, device)
    model.eval()
    example = [torch.rand(3, image_size, image_size, device=device)]
    traced = torch.jit.trace(model, example, strict=False)
    ensure_dir(Path(output).parent)
    traced.save(output)
    print(f"TorchScript saved: {output}")


def export_onnx(weights: str, output: str, image_size: int, device_str: str = "cpu") -> None:
    """Best-effort ONNX export.

    Detection models have dynamic list/dict outputs, so ONNX export can require extra work
    in production. This script keeps the export command isolated for experimentation.
    """
    device = get_device(device_str)
    model, _, _ = load_checkpoint(weights, device)
    model.eval()
    example = [torch.rand(3, image_size, image_size, device=device)]
    ensure_dir(Path(output).parent)
    torch.onnx.export(
        model,
        (example,),
        output,
        opset_version=17,
        dynamo=True,
    )
    print(f"ONNX saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/model.ts")
    parser.add_argument("--format", type=str, choices=["torchscript", "onnx"], default="torchscript")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.format == "torchscript":
        export_torchscript(args.weights, args.output, args.image_size, args.device)
    else:
        export_onnx(args.weights, args.output, args.image_size, args.device)


if __name__ == "__main__":
    main()
