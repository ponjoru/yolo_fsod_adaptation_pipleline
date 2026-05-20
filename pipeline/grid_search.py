"""Phase 1 — Exhaustive grid search over (model, epochs, freeze, freeze_bn)."""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset import DatasetBuilder, FoldPaths
from .head_init import HeadInitializer
from .robustness import RobustnessEvaluator
from .scoring import compute_composite_score
from .trainer import TrainResult, run_training

logger = logging.getLogger(__name__)


@dataclass
class RecipeResult:
    model_name: str
    epochs: int
    freeze: str
    freeze_bn: bool
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    robustness_score: float
    composite_score: float


class GridSearcher:
    """Runs all (model × epochs × freeze × freeze_bn) combos across all CV folds."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.gs_cfg = cfg["grid_search"]
        self.scoring_cfg = cfg["scoring"]
        self.metric_key = (
            "map50" if cfg["scoring"]["metric"] == "map50" else "map"
        )
        self.run_dir = str(Path(cfg["logging"]["save_dir"]) / "grid_search")
        self.checkpoint_path = str(Path(self.run_dir) / "results.jsonl")

    def _load_checkpoint(self) -> List[Dict]:
        """Load previously completed results (for resume on crash)."""
        results = []
        p = Path(self.checkpoint_path)
        if p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
        return results

    def _append_checkpoint(self, record: Dict) -> None:
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _make_run_key(self, model: str, epochs: int, freeze: str, freeze_bn: bool, fold: int) -> str:
        return f"{model}|ep{epochs}|{freeze}|fbn{int(freeze_bn)}|fold{fold}"

    def run(
        self,
        folds: List[FoldPaths],
        head_initializer: Optional[HeadInitializer] = None,
        class_names: Optional[List[str]] = None,
        nc: Optional[int] = None,
    ) -> RecipeResult:
        """Run full grid search, return best RecipeResult."""
        done_keys = {r["run_key"] for r in self._load_checkpoint()}
        per_fold_results: Dict[str, List[float]] = {}   # recipe_key -> [fold_scores]
        rob_scores: Dict[str, List[float]] = {}          # recipe_key -> [rob_scores]

        # Reload existing checkpoint data
        for r in self._load_checkpoint():
            rk = r["recipe_key"]
            per_fold_results.setdefault(rk, []).append(r["score"])
            rob_scores.setdefault(rk, []).append(r["robustness_score"])

        models = self.gs_cfg["models"]
        epochs_list = self.gs_cfg["epochs"]
        freeze_list = self.gs_cfg["freeze"]
        freeze_bn_list = self.gs_cfg["freeze_bn"]

        total = len(models) * len(epochs_list) * len(freeze_list) * len(freeze_bn_list) * len(folds)
        done_count = len(done_keys)
        logger.info(f"Grid search: {total} total runs, {done_count} already done.")

        robustness_evaluator = RobustnessEvaluator(self.cfg)

        run_idx = 0
        for model_name, epochs, freeze, freeze_bn, fold in itertools.product(
            models, epochs_list, freeze_list, freeze_bn_list, folds
        ):
            run_key = self._make_run_key(model_name, epochs, freeze, freeze_bn, fold.fold_idx)
            recipe_key = f"{model_name}|ep{epochs}|{freeze}|fbn{int(freeze_bn)}"
            run_idx += 1

            if run_key in done_keys:
                logger.debug(f"[{run_idx}/{total}] Skip (cached): {run_key}")
                continue

            logger.info(f"[{run_idx}/{total}] Training: {run_key}")

            head_cb = head_initializer.make_callback() if head_initializer else None

            result: TrainResult = run_training(
                model_name=model_name,
                data_yaml=fold.data_yaml,
                fold_idx=fold.fold_idx,
                epochs=epochs,
                freeze=freeze,
                freeze_bn=freeze_bn,
                augment_params={},   # phase 1 uses default augmentation
                run_dir=self.run_dir,
                cfg=self.cfg,
                head_init_callback=head_cb,
            )

            score = result.map50 if self.metric_key == "map50" else result.map

            # Robustness probe for this fold
            rob = 0.0
            if class_names and nc and result.weights_path:
                try:
                    rob = robustness_evaluator.evaluate(
                        weights_path=result.weights_path,
                        val_images=fold.val_images,
                        val_labels=fold.val_labels,
                        nc=nc,
                        class_names=class_names,
                    )
                except Exception as e:
                    logger.warning(f"Robustness eval failed for {run_key}: {e}")

            record = {
                "run_key": run_key,
                "recipe_key": recipe_key,
                "model_name": model_name,
                "epochs": epochs,
                "freeze": freeze,
                "freeze_bn": freeze_bn,
                "fold_idx": fold.fold_idx,
                "map50": result.map50,
                "map": result.map,
                "score": score,
                "robustness_score": rob,
                "weights_path": result.weights_path,
            }
            self._append_checkpoint(record)
            per_fold_results.setdefault(recipe_key, []).append(score)
            rob_scores.setdefault(recipe_key, []).append(rob)

        return self._select_best(per_fold_results, rob_scores)

    def _select_best(
        self,
        per_fold_results: Dict[str, List[float]],
        rob_scores: Dict[str, List[float]],
    ) -> RecipeResult:
        import statistics

        best_recipe: Optional[RecipeResult] = None
        best_score = float("-inf")

        for recipe_key, fold_scores in per_fold_results.items():
            rob_list = rob_scores.get(recipe_key, [])
            mean_rob = statistics.mean(rob_list) if rob_list else 0.0
            composite = compute_composite_score(fold_scores, mean_rob, self.cfg)

            if composite > best_score:
                best_score = composite
                parts = recipe_key.split("|")
                model_name = parts[0]
                epochs = int(parts[1].replace("ep", ""))
                freeze = parts[2]
                freeze_bn = parts[3] == "fbn1"
                cv_std = statistics.stdev(fold_scores) if len(fold_scores) > 1 else 0.0

                best_recipe = RecipeResult(
                    model_name=model_name,
                    epochs=epochs,
                    freeze=freeze,
                    freeze_bn=freeze_bn,
                    cv_scores=fold_scores,
                    cv_mean=statistics.mean(fold_scores),
                    cv_std=cv_std,
                    robustness_score=mean_rob,
                    composite_score=composite,
                )

        if best_recipe is None:
            raise RuntimeError("Grid search produced no results.")

        logger.info(
            f"Best recipe: {best_recipe.model_name}, ep={best_recipe.epochs}, "
            f"freeze={best_recipe.freeze}, freeze_bn={best_recipe.freeze_bn} | "
            f"score={best_recipe.composite_score:.4f} "
            f"(cv_mean={best_recipe.cv_mean:.4f}, cv_std={best_recipe.cv_std:.4f}, "
            f"rob={best_recipe.robustness_score:.4f})"
        )
        return best_recipe
