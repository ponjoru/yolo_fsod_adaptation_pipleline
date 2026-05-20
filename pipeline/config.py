"""Config loading with deep-merge: config_ml.yaml is base, config_user.yaml overrides."""

from __future__ import annotations

import copy
import yaml
from pathlib import Path
from typing import Any


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(
    ml_config_path: str | Path = "config_ml.yaml",
    user_config_path: str | Path | None = "config_user.yaml",
) -> dict[str, Any]:
    cfg = _load_yaml(ml_config_path)
    if user_config_path is not None and Path(user_config_path).exists():
        user_cfg = _load_yaml(user_config_path)
        cfg = _deep_merge(cfg, user_cfg)

    dataset_dir = cfg["data"].get("dataset_dir", "")
    if not dataset_dir:
        raise ValueError("data.dataset_dir must be set in config_user.yaml")

    # Derive standard sub-paths from dataset_dir
    root = Path(dataset_dir)
    cfg["data"]["_train_images_dir"] = str(root / "images" / "train")
    cfg["data"]["_neg_images_dir"] = str(root / "images" / "negatives")
    cfg["data"]["_new_view_dir"] = str(root / "images" / "new_view_train")
    cfg["data"]["_labels_dir"] = str(root / "labels" / "train")
    cfg["data"]["_data_yaml"] = str(root / "data.yaml")

    return cfg
