"""Temporal split construction and YOLO data.yaml generation for each CV fold."""

from __future__ import annotations

import re
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml


_FRAME_RE = re.compile(r"_(\d+)\.[^.]+$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".PNG", ".JPEG"}


def resolve_freeze_bn(cfg: dict, n_train_images: int) -> bool:
    """Resolve freeze_bn config value to a concrete bool.

    Accepts true | false | auto. When auto, returns True if
    n_train_images < freeze_bn_auto_threshold (BN batch statistics
    are unreliable on very small datasets).
    """
    val = cfg["grid_search"]["freeze_bn"]
    if val == "auto":
        threshold = cfg["grid_search"].get("freeze_bn_auto_threshold", 100)
        return n_train_images < threshold
    return bool(val)


def parse_frame_index(filename: str) -> int:
    """Parse frame index from '{video_id}_{frame_id}.ext' filenames."""
    m = _FRAME_RE.search(Path(filename).name)
    if not m:
        raise ValueError(f"Cannot parse frame index from filename: {filename!r}")
    return int(m.group(1))


@dataclass
class FoldPaths:
    """Image/label path lists for one CV fold."""
    fold_idx: int
    train_images: List[str]
    val_images: List[str]
    # corresponding label paths (derived from image paths)
    train_labels: List[str]
    val_labels: List[str]
    # temp dir owning the data.yaml (caller must clean up)
    _temp_dir: str = field(default="", repr=False)
    data_yaml: str = field(default="", repr=False)


class TemporalSplitter:
    """Split images into n_folds consecutive temporal blocks.

    Enforces min_gap_frames at every train/val boundary to prevent
    near-identical frame leakage.
    """

    def __init__(self, n_folds: int = 3, min_gap_frames: int = 5):
        self.n_folds = n_folds
        self.min_gap_frames = min_gap_frames

    def split(self, image_paths: List[str]) -> List[Tuple[List[str], List[str]]]:
        """Return list of (train_paths, val_paths) tuples, one per fold."""
        sorted_paths = sorted(image_paths, key=lambda p: parse_frame_index(p))
        n = len(sorted_paths)
        block_size = n // self.n_folds
        if block_size == 0:
            raise ValueError(
                f"Too few images ({n}) for {self.n_folds} folds. "
                "Need at least n_folds images."
            )

        # Build block boundaries
        blocks: List[List[str]] = []
        for i in range(self.n_folds):
            start = i * block_size
            end = (i + 1) * block_size if i < self.n_folds - 1 else n
            blocks.append(sorted_paths[start:end])

        folds = []
        for val_block_idx in range(self.n_folds):
            val_block = blocks[val_block_idx]
            val_indices = set(
                parse_frame_index(p) for p in val_block
            )

            train_paths: List[str] = []
            for block_idx, block in enumerate(blocks):
                if block_idx == val_block_idx:
                    continue
                for p in block:
                    frame_idx = parse_frame_index(p)
                    # Check min gap against all val frames at block boundaries
                    min_gap = min(abs(frame_idx - vi) for vi in val_indices)
                    if min_gap >= self.min_gap_frames:
                        train_paths.append(p)

            folds.append((train_paths, val_block))

        return folds


class DatasetBuilder:
    """Build per-fold YOLO datasets (data.yaml + image lists) in temp directories."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.train_images_dir = Path(cfg["data"]["_train_images_dir"])
        self.neg_images_dir = Path(cfg["data"]["_neg_images_dir"])
        self.labels_dir = Path(cfg["data"]["_labels_dir"])
        self.negative_ratio: float = cfg["data"]["negative_ratio"]
        self.min_gap_frames: int = cfg["data"]["min_gap_frames"]
        self.n_folds: int = cfg["cv"]["n_folds"]

        # Read class names from the dataset's data.yaml
        base_data_yaml = Path(cfg["data"]["_data_yaml"])
        with open(base_data_yaml) as f:
            base_yaml = yaml.safe_load(f)
        self.class_names: List[str] = base_yaml["names"]
        self.nc: int = len(self.class_names)

    def _label_path(self, image_path: str) -> str:
        img = Path(image_path)
        return str(self.labels_dir / (img.stem + ".txt"))

    def _collect_negatives(self, n_positives: int) -> List[str]:
        if not self.neg_images_dir.exists():
            return []
        neg_imgs = sorted(
            p for p in self.neg_images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        neg_imgs = [str(p) for p in neg_imgs]
        target_count = int(n_positives * self.negative_ratio)
        if len(neg_imgs) < target_count:
            actual_ratio = len(neg_imgs) / max(n_positives, 1)
            warnings.warn(
                f"Negative images available: {len(neg_imgs)}, target: {target_count} "
                f"(ratio {actual_ratio:.2f} vs target {self.negative_ratio:.2f}). "
                f"Add more images to {self.neg_images_dir} to meet the target ratio.",
                UserWarning,
                stacklevel=2,
            )
            return neg_imgs
        return neg_imgs[:target_count]

    def _ensure_neg_labels(self, neg_images: List[str], tmp_label_dir: Path) -> None:
        """Create empty label files for negative images if they don't exist."""
        tmp_label_dir.mkdir(parents=True, exist_ok=True)
        for img_path in neg_images:
            label_name = Path(img_path).stem + ".txt"
            label_file = tmp_label_dir / label_name
            if not label_file.exists():
                label_file.touch()

    def _write_data_yaml(
        self,
        tmp_dir: Path,
        train_txt: str,
        val_txt: str,
    ) -> str:
        data = {
            "path": str(tmp_dir),
            "train": train_txt,
            "val": val_txt,
            "nc": self.nc,
            "names": self.class_names,
        }
        yaml_path = tmp_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        return str(yaml_path)

    def _write_image_list(self, path: Path, image_paths: List[str]) -> None:
        with open(path, "w") as f:
            for p in image_paths:
                f.write(p + "\n")

    def build_folds(self) -> List[FoldPaths]:
        all_images = sorted(
            p for p in self.train_images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        all_images = [str(p) for p in all_images]

        if not all_images:
            raise FileNotFoundError(
                f"No images found in {self.train_images_dir}"
            )

        splitter = TemporalSplitter(
            n_folds=self.n_folds,
            min_gap_frames=self.min_gap_frames,
        )
        raw_folds = splitter.split(all_images)

        fold_paths: List[FoldPaths] = []
        for fold_idx, (train_imgs, val_imgs) in enumerate(raw_folds):
            neg_imgs = self._collect_negatives(len(train_imgs))
            train_all = train_imgs + neg_imgs

            tmp_dir = Path(tempfile.mkdtemp(prefix=f"fold{fold_idx}_"))
            neg_label_dir = tmp_dir / "neg_labels"
            self._ensure_neg_labels(neg_imgs, neg_label_dir)

            train_txt = tmp_dir / "train.txt"
            val_txt = tmp_dir / "val.txt"
            self._write_image_list(train_txt, train_all)
            self._write_image_list(val_txt, val_imgs)

            data_yaml = self._write_data_yaml(tmp_dir, str(train_txt), str(val_txt))

            train_labels = [self._label_path(p) for p in train_imgs] + \
                           [str(neg_label_dir / (Path(p).stem + ".txt")) for p in neg_imgs]
            val_labels = [self._label_path(p) for p in val_imgs]

            fold_paths.append(FoldPaths(
                fold_idx=fold_idx,
                train_images=train_all,
                val_images=val_imgs,
                train_labels=train_labels,
                val_labels=val_labels,
                _temp_dir=str(tmp_dir),
                data_yaml=data_yaml,
            ))

        return fold_paths

    def build_full_train(self) -> FoldPaths:
        """Build a dataset using all available images (no val split) for final retraining."""
        all_images = sorted(
            p for p in self.train_images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        all_images = [str(p) for p in all_images]
        neg_imgs = self._collect_negatives(len(all_images))
        all_with_negs = all_images + neg_imgs

        tmp_dir = Path(tempfile.mkdtemp(prefix="final_train_"))
        neg_label_dir = tmp_dir / "neg_labels"
        self._ensure_neg_labels(neg_imgs, neg_label_dir)

        train_txt = tmp_dir / "train.txt"
        self._write_image_list(train_txt, all_with_negs)
        # Use same images for val in final training (val metrics are informational only)
        data_yaml = self._write_data_yaml(tmp_dir, str(train_txt), str(train_txt))

        return FoldPaths(
            fold_idx=-1,
            train_images=all_with_negs,
            val_images=all_images,
            train_labels=[],
            val_labels=[],
            _temp_dir=str(tmp_dir),
            data_yaml=data_yaml,
        )

    @staticmethod
    def cleanup_fold(fold: FoldPaths) -> None:
        if fold._temp_dir and Path(fold._temp_dir).exists():
            shutil.rmtree(fold._temp_dir, ignore_errors=True)
