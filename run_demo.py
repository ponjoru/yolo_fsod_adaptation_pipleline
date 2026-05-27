#!/usr/bin/env python3
"""
Standalone demo inference — re-run video predictions with any confidence threshold.

Usage:
    python run_demo.py \
        --weights runs/run_<id>/weights/<run_name>/weights/best.pt \
        --demo-dir /path/to/dataset/demo \
        --conf 0.35 \
        --output-dir runs/run_<id>/demo_predictions_conf0.35

Optional — pull imgsz/device defaults from config:
    python run_demo.py --weights ... --demo-dir ... --ml-config config_ml.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from pipeline.demo import run_demo_inference
from pipeline.trainer import suppress_ultralytics_console


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run demo video inference with a trained YOLO model."
    )
    parser.add_argument("--weights",   required=True,       help="Path to .pt weights file")
    parser.add_argument("--demo-dir",  required=True,       help="Directory containing demo videos")
    parser.add_argument("--output-dir", default=None,       help="Output directory (default: demo_predictions/ next to weights)")
    parser.add_argument("--conf",      type=float, default=None, help="Confidence threshold (default: from config or 0.25)")
    parser.add_argument("--imgsz",     type=int,   default=None, help="Inference image size (default: from config or 640)")
    parser.add_argument("--device",    default=None,        help="Device — cuda index or 'cpu' (default: from config or '0')")
    parser.add_argument("--tracker",    default=None,        help="Tracker config: 'bytetrack.yaml' or 'botsort.yaml' (default: no tracking)")
    parser.add_argument("--ml-config",  default=None,       help="Optional: config_ml.yaml to pull imgsz/device/conf defaults")
    parser.add_argument("--user-config", default=None,      help="Optional: config_user.yaml overrides")
    args = parser.parse_args()

    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )

    # Pull defaults from config if provided
    conf_threshold = 0.25
    imgsz = 640
    device = "0"
    tracker = None

    if args.ml_config:
        from pipeline.config import load_config
        cfg = load_config(args.ml_config, args.user_config or "config_user.yaml")
        conf_threshold = cfg.get("demo", {}).get("conf_threshold", 0.25)
        imgsz = cfg["compute"]["imgsz"]
        device = cfg["compute"]["device"]
        tracker = cfg.get("demo", {}).get("tracker", None)

    # CLI args override config
    if args.conf    is not None: conf_threshold = args.conf
    if args.imgsz   is not None: imgsz = args.imgsz
    if args.device  is not None: device = args.device
    if args.tracker is not None: tracker = args.tracker

    output_dir = args.output_dir or str(Path(args.weights).parent.parent.parent / "demo_predictions")

    if not Path(args.weights).exists():
        logger.error(f"Weights not found: {args.weights}")
        sys.exit(1)

    if not Path(args.demo_dir).exists():
        logger.error(f"Demo directory not found: {args.demo_dir}")
        sys.exit(1)

    logger.info(f"Weights   : {args.weights}")
    logger.info(f"Demo dir  : {args.demo_dir}")
    logger.info(f"Output    : {output_dir}")
    logger.info(f"Conf      : {conf_threshold}  |  imgsz: {imgsz}  |  device: {device}  |  tracker: {tracker or 'none'}")

    suppress_ultralytics_console()

    output_paths = run_demo_inference(
        weights_path=args.weights,
        demo_dir=args.demo_dir,
        output_dir=output_dir,
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        device=device,
        tracker=tracker,
    )

    if output_paths:
        logger.info(f"Done — {len(output_paths)} video(s) saved to {output_dir}")
    else:
        logger.warning("No videos processed.")


if __name__ == "__main__":
    main()
