"""Phase 2 — Bayesian optimisation over geometry augmentation hyperparameters (Optuna)."""

from __future__ import annotations
import optuna
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset import FoldPaths
from .grid_search import RecipeResult
from .head_init import HeadInitializer
from .robustness import RobustnessEvaluator
from .scoring import compute_composite_score
from .trainer import run_training

logger = logging.getLogger(__name__)


@dataclass
class BayesianResult:
    perspective: float
    scale: float
    translate: float
    degrees: float
    best_score: float


class BayesianSearcher:
    """Optimise geometry augmentation params with Optuna, fixing phase-1 recipe."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.bs_cfg = cfg["bayesian_search"]
        self.n_trials: int = self.bs_cfg["n_trials"]
        self.geo_bounds: Dict = self.bs_cfg["geometry"]
        self.metric_key = "map50" if cfg["scoring"]["metric"] == "map50" else "map"
        self.run_dir = str(Path(cfg["logging"]["save_dir"]) / "bayesian_search")

    def run(
        self,
        best_recipe: RecipeResult,
        folds: List[FoldPaths],
        head_initializer: Optional[HeadInitializer] = None,
        class_names: Optional[List[str]] = None,
        nc: Optional[int] = None,
    ) -> BayesianResult:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        robustness_evaluator = RobustnessEvaluator(self.cfg)

        def objective(trial: optuna.Trial) -> float:
            augment_params = {
                "perspective": trial.suggest_float(
                    "perspective",
                    self.geo_bounds["perspective"][0],
                    self.geo_bounds["perspective"][1],
                ),
                "scale": trial.suggest_float(
                    "scale",
                    self.geo_bounds["scale"][0],
                    self.geo_bounds["scale"][1],
                ),
                "translate": trial.suggest_float(
                    "translate",
                    self.geo_bounds["translate"][0],
                    self.geo_bounds["translate"][1],
                ),
                "degrees": trial.suggest_float(
                    "rotation",
                    self.geo_bounds["rotation"][0],
                    self.geo_bounds["rotation"][1],
                ),
            }

            cv_scores: List[float] = []
            rob_scores: List[float] = []

            for fold in folds:
                head_cb = head_initializer.make_callback() if head_initializer else None

                result = run_training(
                    model_name=best_recipe.model_name,
                    data_yaml=fold.data_yaml,
                    fold_idx=fold.fold_idx,
                    epochs=best_recipe.epochs,
                    freeze=best_recipe.freeze,
                    freeze_bn=best_recipe.freeze_bn,
                    augment_params=augment_params,
                    run_dir=self.run_dir,
                    cfg=self.cfg,
                    head_init_callback=head_cb,
                )

                score = result.map50 if self.metric_key == "map50" else result.map
                cv_scores.append(score)

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
                        logger.warning(f"Robustness eval failed in Bayesian trial: {e}")
                rob_scores.append(rob)

            mean_rob = statistics.mean(rob_scores) if rob_scores else 0.0
            return compute_composite_score(cv_scores, mean_rob, self.cfg)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name="bayesian_aug_search",
            storage=None,
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        best = study.best_params
        logger.info(
            f"Bayesian search best: perspective={best['perspective']:.5f}, "
            f"scale={best['scale']:.3f}, translate={best['translate']:.3f}, "
            f"degrees={best['rotation']:.2f} | score={study.best_value:.4f}"
        )

        return BayesianResult(
            perspective=best["perspective"],
            scale=best["scale"],
            translate=best["translate"],
            degrees=best["rotation"],
            best_score=study.best_value,
        )
