"""Phase 1 — Model selection via fixed baseline hyperparameters.

Each candidate model (yolo11s/m/l) is trained with the same fixed
baseline (epochs, freeze, lr0) across all CV folds. The best model
is selected by composite score and passed to Phase 2.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset import FoldPaths, resolve_freeze_bn
from .head_init import HeadInitializer
from .robustness import RobustnessEvaluator
from .scoring import compute_composite_score
from .trainer import run_training
from .utils import TopKWeightsTracker, append_csv_row, cleanup_run_artifacts

logger = logging.getLogger(__name__)


@dataclass
class ModelSelectionResult:
    model_name: str
    freeze_bn: bool          # resolved value — carried forward to Phase 2 and final training
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    robustness_score: float
    composite_score: float


class GridSearcher:
    """Phase 1: select best model architecture using fixed baseline hyperparameters."""

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        self.cfg = cfg
        self.gs_cfg = cfg["grid_search"]
        self.metric_key = "map50" if cfg["scoring"]["metric"] == "map50" else "map"
        self.run_dir = run_dir
        self.checkpoint_path = str(Path(run_dir) / "results.jsonl")

    def _load_checkpoint(self) -> List[Dict]:
        p = Path(self.checkpoint_path)
        if not p.exists():
            return []
        with open(p) as f:
            return [json.loads(l) for l in f if l.strip()]

    def _append_checkpoint(self, record: Dict) -> None:
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def run(
        self,
        folds: List[FoldPaths],
        head_initializer: Optional[HeadInitializer] = None,
        class_names: Optional[List[str]] = None,
        nc: Optional[int] = None,
        probe_mosaic_dir: Optional[str] = None,
        csv_path: Optional[str] = None,
        weights_tracker: Optional[TopKWeightsTracker] = None,
    ) -> ModelSelectionResult:
        baseline = self.gs_cfg["baseline"]
        baseline_epochs: int = baseline["epochs"]
        baseline_freeze: str = baseline["freeze"]
        baseline_lr0: float = float(baseline["lr0"])

        n_train_images = sum(len(f.val_images) for f in folds)
        freeze_bn = resolve_freeze_bn(self.cfg, n_train_images)
        logger.info(
            f"Phase 1 baseline: epochs={baseline_epochs}, freeze={baseline_freeze}, "
            f"lr0={baseline_lr0}, freeze_bn={freeze_bn} "
            f"(config={self.gs_cfg['freeze_bn']!r}, n_train_images={n_train_images})"
        )

        done_records = self._load_checkpoint()
        done_keys = {r["run_key"] for r in done_records}

        per_model_scores: Dict[str, List[float]] = {}
        per_model_rob: Dict[str, List[float]] = {}
        for r in done_records:
            per_model_scores.setdefault(r["model_name"], []).append(r["score"])
            per_model_rob.setdefault(r["model_name"], []).append(r["robustness_score"])

        models = self.gs_cfg["models"]
        total = len(models) * len(folds)
        done_count = len(done_keys)
        logger.info(f"Grid search: {total} total runs, {done_count} already done.")

        robustness_evaluator = RobustnessEvaluator(self.cfg)
        _phase1_mosaic_saved = False

        run_idx = 0
        for model_name in models:
            for fold in folds:
                run_key = f"{model_name}|fold{fold.fold_idx}"
                run_idx += 1

                if run_key in done_keys:
                    logger.debug(f"[{run_idx}/{total}] Skip (cached): {run_key}")
                    continue

                logger.info(f"[{run_idx}/{total}] Training: {run_key}")
                head_cb = head_initializer.make_callback() if head_initializer else None

                result = run_training(
                    model_name=model_name,
                    data_yaml=fold.data_yaml,
                    fold_idx=fold.fold_idx,
                    epochs=baseline_epochs,
                    freeze=baseline_freeze,
                    freeze_bn=freeze_bn,
                    lr0=baseline_lr0,
                    augment_params={},
                    run_dir=self.run_dir,
                    cfg=self.cfg,
                    head_init_callback=head_cb,
                )

                score = result.map50 if self.metric_key == "map50" else result.map

                rob = 0.0
                if class_names and nc and result.weights_path:
                    mosaic_dir = probe_mosaic_dir if (probe_mosaic_dir and not _phase1_mosaic_saved) else None
                    try:
                        rob = robustness_evaluator.evaluate(
                            weights_path=result.weights_path,
                            val_images=fold.val_images,
                            val_labels=fold.val_labels,
                            nc=nc,
                            class_names=class_names,
                            mosaic_dir=mosaic_dir,
                            overlay_predictions=False,
                        )
                        if mosaic_dir:
                            _phase1_mosaic_saved = True
                    except Exception as e:
                        logger.warning(f"Robustness eval failed for {run_key}: {e}")

                # Top-k tracking then cleanup (robustness eval already consumed weights)
                if weights_tracker and result.save_dir:
                    weights_tracker.consider(score, Path(result.save_dir).name, result.save_dir)
                if result.save_dir:
                    cleanup_run_artifacts(result.save_dir)

                record = {
                    "run_key": run_key,
                    "model_name": model_name,
                    "fold_idx": fold.fold_idx,
                    "map50": result.map50,
                    "map": result.map,
                    "score": score,
                    "robustness_score": rob,
                    "weights_path": result.weights_path,
                }
                self._append_checkpoint(record)
                per_model_scores.setdefault(model_name, []).append(score)
                per_model_rob.setdefault(model_name, []).append(rob)

            # Write one CSV row per model after all its folds complete
            if csv_path:
                fold_scores = per_model_scores.get(model_name, [])
                rob_scores = per_model_rob.get(model_name, [])
                mean_rob = statistics.mean(rob_scores) if rob_scores else 0.0
                composite = compute_composite_score(fold_scores, mean_rob, self.cfg)
                cv_mean = statistics.mean(fold_scores) if fold_scores else 0.0
                cv_std = statistics.stdev(fold_scores) if len(fold_scores) > 1 else 0.0
                append_csv_row(csv_path, [
                    model_name, "phase1",
                    f"{composite:.6f}", f"{cv_mean:.6f}", f"{cv_std:.6f}", f"{mean_rob:.6f}",
                ])

        return self._select_best(per_model_scores, per_model_rob, freeze_bn)

    def _select_best(
        self,
        per_model_scores: Dict[str, List[float]],
        per_model_rob: Dict[str, List[float]],
        freeze_bn: bool,
    ) -> ModelSelectionResult:
        best: Optional[ModelSelectionResult] = None
        best_score = float("-inf")

        for model_name, fold_scores in per_model_scores.items():
            rob_list = per_model_rob.get(model_name, [])
            mean_rob = statistics.mean(rob_list) if rob_list else 0.0
            composite = compute_composite_score(fold_scores, mean_rob, self.cfg)

            if composite > best_score:
                best_score = composite
                cv_std = statistics.stdev(fold_scores) if len(fold_scores) > 1 else 0.0
                best = ModelSelectionResult(
                    model_name=model_name,
                    freeze_bn=freeze_bn,
                    cv_scores=fold_scores,
                    cv_mean=statistics.mean(fold_scores),
                    cv_std=cv_std,
                    robustness_score=mean_rob,
                    composite_score=composite,
                )

        if best is None:
            raise RuntimeError("Grid search produced no results.")

        logger.info(
            f"Best model: {best.model_name} | "
            f"score={best.composite_score:.4f} "
            f"(cv_mean={best.cv_mean:.4f}, cv_std={best.cv_std:.4f}, "
            f"rob={best.robustness_score:.4f})"
        )
        return best
