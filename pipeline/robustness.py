"""Synthetic robustness probes: perturb val images, re-evaluate, return mAP.

Debug visualizations:
  - Phase 1 (first robustness call): saves raw perturbed image mosaics.
  - Final (after full retraining): saves mosaics with predictions overlaid.
One JPEG per perturbation type is written to the supplied mosaic_dir.
"""

from __future__ import annotations

import logging
import math
import random as _random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def _perspective_warp(img: np.ndarray, strength: float = 0.05) -> np.ndarray:
    h, w = img.shape[:2]
    dx = int(w * strength)
    dy = int(h * strength)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [dx, dy], [w - dx, dy],
        [w - dx // 2, h - dy // 2], [dx // 2, h - dy // 2],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h))


def _crop(img: np.ndarray, crop_frac: float = 0.85) -> np.ndarray:
    h, w = img.shape[:2]
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    cropped = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _blur(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _compression(img: np.ndarray, quality: int = 30) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _brightness_shift(img: np.ndarray, delta: int = 60) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _scale_perturbation(img: np.ndarray, scale: float = 0.75) -> np.ndarray:
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros_like(img)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = small
    return out


_PERTURBATIONS: Dict[str, callable] = {
    "perspective_warp": _perspective_warp,
    "crop": _crop,
    "blur": _blur,
    "compression": _compression,
    "brightness_shift": _brightness_shift,
    "scale_perturbation": _scale_perturbation,
}

_MOSAIC_CELL_SIZE = 320  # px per cell in debug mosaics


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RobustnessEvaluator:
    """Apply each enabled perturbation to val images and return mean mAP.

    Optionally saves debug mosaic images when mosaic_dir is provided to
    evaluate(). Sampling is deterministic (seed-based) so Phase 1 and
    final mosaics always show the same source images.
    """

    def __init__(self, cfg: Dict[str, Any]):
        probes_cfg = cfg.get("robustness_probes", {})
        # Separate the boolean probe flags from non-probe keys
        self.probes_cfg: Dict[str, bool] = {
            k: v for k, v in probes_cfg.items() if k in _PERTURBATIONS
        }
        self.metric: str = cfg["scoring"]["metric"]
        self.compute_cfg = cfg["compute"]
        self.n_mosaic_samples: int = int(probes_cfg.get("n_mosaic_samples", 4))
        self._seed: int = cfg["compute"]["seed"]

    # ------------------------------------------------------------------
    # Mosaic helpers
    # ------------------------------------------------------------------

    def _select_samples(self, images: List[str]) -> List[str]:
        """Deterministically select up to n_mosaic_samples images."""
        rng = _random.Random(self._seed)
        pool = sorted(images)   # sort for filesystem-order independence
        return rng.sample(pool, min(self.n_mosaic_samples, len(pool)))

    def _build_mosaic(self, cells: List[np.ndarray]) -> np.ndarray:
        """Tile cell images into a square-ish grid."""
        n = len(cells)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        s = _MOSAIC_CELL_SIZE
        blank = np.zeros((s, s, 3), dtype=np.uint8)
        resized = [cv2.resize(c, (s, s)) for c in cells]
        # Pad to fill the grid rectangle
        resized += [blank] * (cols * rows - n)
        grid_rows = [
            np.hstack(resized[r * cols: (r + 1) * cols])
            for r in range(rows)
        ]
        return np.vstack(grid_rows)

    def _draw_predictions(
        self,
        img: np.ndarray,
        model,
        class_names: List[str],
        conf: float = 0.25,
    ) -> np.ndarray:
        """Run inference and draw predicted boxes onto img."""
        results = model.predict(img, conf=conf, verbose=False, save=False)
        out = img.copy()
        if results and len(results[0].boxes):
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].cpu().numpy())
                conf_val = float(box.conf[0].cpu().numpy())
                label = f"{class_names[cls_id] if cls_id < len(class_names) else cls_id} {conf_val:.2f}"
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(out, label, (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
                            cv2.LINE_AA)
        return out

    def _save_mosaics(
        self,
        sample_images: List[str],
        mosaic_dir: str,
        model=None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Save one mosaic JPEG per enabled probe type to mosaic_dir."""
        Path(mosaic_dir).mkdir(parents=True, exist_ok=True)
        enabled = [k for k in self.probes_cfg if self.probes_cfg[k]]

        for probe_name in enabled:
            perturb_fn = _PERTURBATIONS[probe_name]
            cells: List[np.ndarray] = []

            for img_path in sample_images:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                cell = perturb_fn(img)
                if model is not None and class_names:
                    cell = self._draw_predictions(cell, model, class_names)
                cells.append(cell)

            if not cells:
                continue

            mosaic = self._build_mosaic(cells)
            out_path = Path(mosaic_dir) / f"{probe_name}.jpg"
            cv2.imwrite(str(out_path), mosaic)
            logger.info(f"  Probe mosaic saved: {out_path}")

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def _apply_perturbations_to_dir(
        self,
        val_images: List[str],
        perturb_fn: callable,
        tmp_img_dir: Path,
    ) -> List[str]:
        tmp_img_dir.mkdir(parents=True, exist_ok=True)
        out_paths = []
        for img_path in val_images:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"Could not read image: {img_path}")
                continue
            out_path = tmp_img_dir / Path(img_path).name
            cv2.imwrite(str(out_path), perturb_fn(img))
            out_paths.append(str(out_path))
        return out_paths

    def _write_probe_data_yaml(
        self,
        tmp_dir: Path,
        perturbed_images: List[str],
        val_labels: List[str],
        nc: int,
        class_names: List[str],
    ) -> str:
        img_list = tmp_dir / "probe_val.txt"
        with open(img_list, "w") as f:
            for p in perturbed_images:
                f.write(p + "\n")

        label_dir = tmp_dir / "labels"
        label_dir.mkdir(exist_ok=True)
        for lp in val_labels:
            src = Path(lp)
            if src.exists():
                dst = label_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)

        data = {
            "path": str(tmp_dir),
            "train": str(img_list),
            "val": str(img_list),
            "nc": nc,
            "names": class_names,
        }
        yaml_path = tmp_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)
        return str(yaml_path)

    def evaluate(
        self,
        weights_path: str,
        val_images: List[str],
        val_labels: List[str],
        nc: int,
        class_names: List[str],
        mosaic_dir: Optional[str] = None,
        overlay_predictions: bool = False,
    ) -> float:
        """Run all enabled probes, return mean mAP.

        If mosaic_dir is set, also saves one debug mosaic per probe type.
        overlay_predictions=True draws model predictions on the mosaic cells.
        """
        from ultralytics import YOLO

        enabled = [k for k, v in self.probes_cfg.items() if v]
        if not enabled:
            logger.info("No robustness probes enabled.")
            return 0.0

        if not Path(weights_path).exists():
            logger.warning(f"Weights not found for robustness eval: {weights_path}")
            return 0.0

        model = YOLO(weights_path)
        metric_key = "metrics/mAP50(B)" if self.metric == "map50" else "metrics/mAP50-95(B)"
        scores: List[float] = []
        tmp_root = Path(tempfile.mkdtemp(prefix="robustness_"))

        try:
            for probe_name in enabled:
                perturb_fn = _PERTURBATIONS.get(probe_name)
                if perturb_fn is None:
                    logger.warning(f"Unknown probe: {probe_name}")
                    continue

                probe_dir = tmp_root / probe_name
                img_dir = probe_dir / "images"
                perturbed = self._apply_perturbations_to_dir(val_images, perturb_fn, img_dir)

                if not perturbed:
                    logger.warning(f"No images produced for probe {probe_name}")
                    continue

                data_yaml = self._write_probe_data_yaml(
                    probe_dir, perturbed, val_labels, nc, class_names
                )

                try:
                    results = model.val(
                        data=data_yaml,
                        device=self.compute_cfg["device"],
                        workers=self.compute_cfg["workers"],
                        imgsz=self.compute_cfg["imgsz"],
                        verbose=False,
                        plots=False,
                        save=False,
                    )
                    metrics = results.results_dict if hasattr(results, "results_dict") else {}
                    score = float(metrics.get(metric_key, 0.0))
                    logger.info(f"  Probe [{probe_name}]: {self.metric} = {score:.4f}")
                    scores.append(score)
                except Exception as e:
                    logger.warning(f"Probe {probe_name} failed: {e}")
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        # Debug mosaics — saved after metric loop so the model is already loaded
        if mosaic_dir:
            sample_images = self._select_samples(val_images)
            viz_model = model if overlay_predictions else None
            viz_classes = class_names if overlay_predictions else None
            try:
                self._save_mosaics(sample_images, mosaic_dir, viz_model, viz_classes)
            except Exception as e:
                logger.warning(f"Mosaic save failed: {e}")

        return float(np.mean(scores)) if scores else 0.0
