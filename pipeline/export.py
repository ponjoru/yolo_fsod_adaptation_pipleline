"""Export final trained YOLO model to ONNX."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def export_onnx(
    weights_path: str,
    output_dir: str,
    imgsz: int = 640,
    run_id: str = "final",
) -> str:
    """Export best.pt to ONNX, copy to output_dir. Returns path to .onnx file."""
    from ultralytics import YOLO

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights_path)
    export_path = model.export(format="onnx", imgsz=imgsz)

    if not export_path or not Path(export_path).exists():
        raise RuntimeError(f"ONNX export failed — output not found: {export_path}")

    dest = out_dir / f"{run_id}.onnx"
    shutil.copy2(export_path, dest)
    logger.info(f"ONNX model saved to {dest}")
    return str(dest)
