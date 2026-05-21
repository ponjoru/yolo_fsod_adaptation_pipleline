"""Demo video inference — run the trained model over demo videos and save annotated output."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import cv2

from .trainer import suppress_ultralytics_console

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _find_videos(demo_dir: str) -> List[Path]:
    root = Path(demo_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def run_demo_inference(
    weights_path: str,
    demo_dir: str,
    output_dir: str,
    conf_threshold: float,
    imgsz: int,
    device: str,
) -> List[str]:
    """Run inference on all videos in demo_dir, save annotated mp4s to output_dir.

    Returns list of output file paths.
    """
    from ultralytics import YOLO

    suppress_ultralytics_console()

    videos = _find_videos(demo_dir)
    if not videos:
        logger.info(f"No videos found in {demo_dir} — skipping demo inference.")
        return []

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights_path)
    output_paths: List[str] = []

    for video_path in videos:
        logger.info(f"  Demo: {video_path.name}  conf={conf_threshold}")
        out_path = out_root / f"{video_path.stem}.mp4"

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        try:
            for frame_idx, result in enumerate(
                model.predict(
                    source=str(video_path),
                    imgsz=imgsz,
                    conf=conf_threshold,
                    device=device,
                    stream=True,
                    verbose=False,
                    save=False,
                )
            ):
                writer.write(result.plot())
                if total_frames and (frame_idx + 1) % 200 == 0:
                    logger.debug(f"    {frame_idx + 1}/{total_frames} frames")
        finally:
            writer.release()

        logger.info(f"  Saved: {out_path}")
        output_paths.append(str(out_path))

    return output_paths
