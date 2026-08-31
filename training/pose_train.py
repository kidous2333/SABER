#!/usr/bin/env python
"""pose_train.py — YOLO pose model training for mouse body/tail keypoints.

Usage:
  python pose_train.py --model yolo11n-pose.yaml --data coco8-pose_mouse.yaml --epochs 30
  python pose_train.py --model yolo11n-pose.yaml --data coco8-pose_tail.yaml --epochs 50 --batch 32
"""


import sys
from pathlib import Path
# Allow running as a script from a subdirectory (e.g. `python mining/discovery.py`)
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO pose model for mouse keypoints")
    parser.add_argument("--model", type=str, default="yolo11n-pose.yaml",
                        help="Model YAML config (default: yolo11n-pose.yaml)")
    parser.add_argument("--data", type=str, required=True,
                        help="Dataset YAML path (e.g. coco8-pose_mouse.yaml)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device (empty=auto, 0=gpu0, cpu)")
    parser.add_argument("--project", type=str, default="runs/train", help="Output project dir")
    parser.add_argument("--name", type=str, default="pose_exp", help="Experiment name")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Pretrained weights path (optional)")

    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"Error: dataset not found: {args.data}")
        return

    from src.pose_trainer import train_tmp_model

    print(f"Training pose model: {args.model}")
    print(f"  Dataset:  {args.data}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Imgsz:    {args.imgsz}")
    print(f"  Batch:    {args.batch}")

    metrics = train_tmp_model(
        model_yaml=args.model,
        dataset_path=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        pretrained_path=args.pretrained,
    )

    print(f"\nTraining complete.")
    if metrics:
        print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"  Weights: {Path(args.project) / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
