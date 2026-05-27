"""COCO head weight transfer for mapped classes.

After Ultralytics reinitializes the detection head (due to nc mismatch),
the HeadInitializer callback copies pretrained COCO classification weights
into the target class slots for any class that has a COCO mapping.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# COCO 80-class list (index matches COCO class_id)
COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

_COCO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(COCO_CLASSES)}


def get_coco_index(class_name: str) -> Optional[int]:
    return _COCO_INDEX.get(class_name.lower())


def _get_last_conv(module: nn.Module) -> Optional[nn.Conv2d]:
    """Return the last Conv2d layer in a Sequential or Module."""
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


class HeadInitializer:
    """Extracts COCO detection head weights and produces a training callback
    that copies them into the new-nc model for mapped classes.

    Weights are extracted lazily per model name and cached, so a single
    HeadInitializer instance works correctly across architectures of different
    channel widths (yolo11s/m/l).
    """

    def __init__(
        self,
        target_classes: List[str],
        class_mapping: Dict[str, object],  # target_name -> coco_name | [coco_names]
    ):
        self.target_classes = target_classes
        self.class_mapping = {k.lower(): v for k, v in class_mapping.items()}
        self._cache: Dict[str, Optional[List]] = {}

    def _extract_coco_head(self, weights_path: str) -> Optional[List]:
        """Load COCO model and store cv3 (classification branch) weights per scale."""
        try:
            from ultralytics import YOLO
            model = YOLO(weights_path)
            detect = model.model.model[-1]  # Detect module

            scale_weights = []
            for seq in detect.cv3:
                conv = _get_last_conv(seq)
                if conv is None:
                    return None
                scale_weights.append({
                    "weight": conv.weight.data.clone(),   # [80, C, 1, 1]
                    "bias": conv.bias.data.clone() if conv.bias is not None else None,
                })
            return scale_weights
        except Exception as e:
            logger.warning(f"Could not extract COCO head weights from {weights_path!r}: {e}. "
                           "Mapped classes will use random initialization.")
            return None

    def _get_weights(self, model_name: str) -> Optional[List]:
        if model_name not in self._cache:
            self._cache[model_name] = self._extract_coco_head(f"{model_name}.pt")
        return self._cache[model_name]

    def _resolve_coco_indices(self, target_class: str) -> List[int]:
        raw = self.class_mapping.get(target_class.lower())
        if raw is None:
            return []
        if isinstance(raw, str):
            raw = [raw]
        indices = []
        for name in raw:
            idx = get_coco_index(name)
            if idx is None:
                logger.warning(f"COCO class {name!r} not found; skipping.")
            else:
                indices.append(idx)
        return indices

    def make_callback(self, model_name: str) -> callable:
        """Return an on_train_start callback that copies COCO head weights for model_name."""
        coco_weights = self._get_weights(model_name)
        target_classes = self.target_classes

        def on_train_start(trainer):
            if coco_weights is None:
                return

            try:
                detect = trainer.model.model[-1]
            except (AttributeError, IndexError):
                logger.warning("Could not access Detect module for head init; skipping.")
                return

            n_scales = len(detect.cv3)
            if n_scales != len(coco_weights):
                logger.warning(
                    f"Scale count mismatch: model has {n_scales}, "
                    f"COCO cache has {len(coco_weights)}. Skipping head init."
                )
                return

            transferred = 0
            for scale_idx, (seq, coco_scale) in enumerate(zip(detect.cv3, coco_weights)):
                conv = _get_last_conv(seq)
                if conv is None:
                    continue

                coco_w = coco_scale["weight"]   # [80, C, 1, 1]
                coco_b = coco_scale["bias"]     # [80] or None
                nc_target = conv.weight.shape[0]

                with torch.no_grad():
                    for tgt_idx, tgt_class in enumerate(target_classes):
                        if tgt_idx >= nc_target:
                            break
                        coco_indices = self._resolve_coco_indices(tgt_class)
                        if not coco_indices:
                            continue  # novel class — keep random init

                        # Guard against shape mismatch (different backbone size)
                        if coco_w.shape[1] != conv.weight.shape[1]:
                            logger.warning(
                                f"Channel mismatch at scale {scale_idx}: "
                                f"COCO {coco_w.shape[1]} vs model {conv.weight.shape[1]}. "
                                "Skipping weight copy for this scale."
                            )
                            break

                        # Average COCO weights across all mapped source classes
                        src_w = coco_w[coco_indices].mean(dim=0, keepdim=True)  # [1,C,1,1]
                        conv.weight.data[tgt_idx : tgt_idx + 1].copy_(src_w)

                        if coco_b is not None and conv.bias is not None:
                            src_b = coco_b[coco_indices].mean()
                            conv.bias.data[tgt_idx] = src_b

                        transferred += 1
                        logger.info(
                            f"  [{tgt_class}] <- COCO {[COCO_CLASSES[i] for i in coco_indices]} "
                            f"(scale {scale_idx})"
                        )

            logger.info(f"Head init: transferred weights for {transferred} class×scale slots.")

        return on_train_start
