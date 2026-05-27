"""Single-run YOLO training wrapper."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.utils as uu
      
logger = logging.getLogger(__name__)

# Freeze strategy -> number of layers to freeze (YOLO11 backbone ≈ 10 layers)
FREEZE_STRATEGIES: Dict[str, int] = {
    "full_backbone": 10,
    "partial_backbone": 6,
    "early_only": 3,
    "full_finetune": 0,
}


@dataclass
class TrainResult:
    fold_idx: int
    model_name: str
    epochs: int
    freeze: str
    freeze_bn: bool
    map50: float
    map: float
    weights_path: str
    save_dir: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def suppress_ultralytics_console() -> None:
    """Strip all uu.LOGGER handlers and disable tqdm/rich bars at startup.

    Must be called once after ultralytics is imported and before any YOLO call.
    _redirect_ultralytics_to_file() re-applies VERBOSE=False per training run,
    but this covers the window before the first run (e.g. HeadInitializer).
    """
    for handler in uu.LOGGER.handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        uu.LOGGER.removeHandler(handler)
    uu.LOGGER.propagate = False
    uu.VERBOSE = False


def _redirect_ultralytics_to_file(log_path: str) -> None:
    """Redirect all Ultralytics logging to a per-run file, away from the console."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    ul_logger = uu.LOGGER
    for handler in ul_logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        ul_logger.removeHandler(handler)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    ul_logger.addHandler(fh)
    ul_logger.setLevel(logging.DEBUG)
    ul_logger.propagate = False
    uu.VERBOSE = False


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark=False is required for determinism; trades ~10% throughput for reproducibility
    torch.backends.cudnn.benchmark = False


def _make_freeze_bn_callback():
    def on_train_epoch_start(trainer):
        # Keep BN layers in eval mode so they use running stats, not batch stats
        for module in trainer.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    return on_train_epoch_start


def run_training(
    *,
    model_name: str,
    data_yaml: str,
    fold_idx: int,
    epochs: int,
    freeze: str,
    freeze_bn: bool,
    lr0: float = 0.01,
    augment_params: Dict[str, Any],
    run_dir: str,
    cfg: Dict[str, Any],
    head_init_callback: Optional[callable] = None,
    run_name_prefix: str = "",
) -> TrainResult:
    """Train one YOLO model for one fold. Returns TrainResult with metrics."""

    verbose = cfg["logging"]["verbose"]
    seed = cfg["compute"]["seed"]
    base_name = f"{model_name}_ep{epochs}_{freeze}_fbn{int(freeze_bn)}"
    run_name = f"{run_name_prefix}_{base_name}" if run_name_prefix else base_name
    log_path = str(Path(run_dir) / run_name / f"fold{fold_idx}.log")
    _redirect_ultralytics_to_file(log_path)
    _set_global_seeds(seed)

    n_freeze = FREEZE_STRATEGIES.get(freeze, 0)
    device = cfg["compute"]["device"]
    workers = cfg["compute"]["workers"]
    batch = cfg["compute"]["batch"]
    imgsz = cfg["compute"]["imgsz"]

    weights_file = f"{model_name}.pt"
    model = YOLO(weights_file)

    if head_init_callback is not None:
        model.add_callback("on_train_start", head_init_callback)

    if freeze_bn:
        model.add_callback("on_train_epoch_start", _make_freeze_bn_callback())

    # Build augmentation kwargs — start from ML config defaults then apply overrides
    aug_cfg = cfg.get("augmentations", {})
    train_kwargs = {
        "data": data_yaml,
        "epochs": epochs,
        "freeze": n_freeze,
        "device": device,
        "workers": workers,
        "batch": batch,
        "imgsz": imgsz,
        "name": run_name,
        "exist_ok": True,
        "verbose": verbose,
        "patience": 0,           # disable early stopping in search phase
        "seed": seed,
        "lr0": lr0,
        "optimizer": 'AdamW',
        "save": True,
        "plots": False,
        # Augmentation defaults from ML config
        "mosaic": aug_cfg.get("mosaic", 0.0),
        "mixup": aug_cfg.get("mixup", 0.0),
        "copy_paste": aug_cfg.get("copy_paste", 0.0),
        "fliplr": aug_cfg.get("fliplr", 0.5),
        "flipud": aug_cfg.get("flipud", 0.0),
    }

    # Geometry aug overrides (from Bayesian search or defaults)
    for key in ("perspective", "scale", "translate", "degrees"):
        if key in augment_params:
            train_kwargs[key] = augment_params[key]

    args_path = Path(run_dir) / run_name / "args.json"
    args_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args_path, "w") as f:
        json.dump(train_kwargs, f, indent=2)

    try:
        with open(log_path, "a") as _stderr_sink, contextlib.redirect_stderr(_stderr_sink):
            results = model.train(**train_kwargs)
    except Exception as e:
        logger.error(f"Training failed (fold={fold_idx}, model={model_name}): {e}")
        return TrainResult(
            fold_idx=fold_idx,
            model_name=model_name,
            epochs=epochs,
            freeze=freeze,
            freeze_bn=freeze_bn,
            map50=0.0,
            map=0.0,
            weights_path="",
            save_dir="",
        )

    # Extract metrics from results
    metrics = results.results_dict if hasattr(results, "results_dict") else {}
    map50 = float(metrics.get("metrics/mAP50(B)", 0.0))
    map_val = float(metrics.get("metrics/mAP50-95(B)", 0.0))

    # Find best weights
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("runs") / "detect" / run_name
    best_weights = str(save_dir / "weights" / "best.pt")
    if not Path(best_weights).exists():
        best_weights = str(save_dir / "weights" / "last.pt")

    return TrainResult(
        fold_idx=fold_idx,
        model_name=model_name,
        epochs=epochs,
        freeze=freeze,
        freeze_bn=freeze_bn,
        map50=map50,
        map=map_val,
        weights_path=best_weights,
        save_dir=str(save_dir),
    )
