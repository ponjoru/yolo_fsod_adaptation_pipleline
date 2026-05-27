"""Phase 2 — Joint Bayesian optimisation over the full training recipe.

Fixes the model architecture selected in Phase 1 and jointly searches:
  epochs       [int]         — training length
  freeze       [categorical] — freeze strategy
  lr0          [log-float]   — initial learning rate
  perspective  [float]       — geometry augmentation
  scale        [float]       — geometry augmentation
  translate    [float]       — geometry augmentation
  degrees      [float]       — geometry augmentation (rotation)

Optuna SQLite storage is used so a crashed run can be resumed cleanly
by re-running the pipeline with the same run_id / config.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna

from .dataset import FoldPaths
from .grid_search import ModelSelectionResult
from .head_init import HeadInitializer
from .robustness import RobustnessEvaluator
from .scoring import compute_composite_score
from .trainer import run_training
from .utils import append_csv_row, delete_run_dir

logger = logging.getLogger(__name__)


@dataclass
class BayesianResult:
    # Optimisation recipe
    epochs: int
    freeze: str
    lr0: float
    # Geometry augmentations
    perspective: float
    scale: float
    translate: float
    degrees: float
    best_score: float


class BayesianSearcher:
    """Phase 2: joint Bayesian search over epochs, freeze, lr0, and geometry augs."""

    def __init__(self, cfg: Dict[str, Any], run_dir: str):
        self.cfg = cfg
        self.bs_cfg = cfg["bayesian_search"]
        self.n_trials: int = self.bs_cfg["n_trials"]
        self.metric_key = "map50" if cfg["scoring"]["metric"] == "map50" else "map"
        self.run_dir = run_dir

    def run(
        self,
        model_result: ModelSelectionResult,
        folds: List[FoldPaths],
        head_initializer: Optional[HeadInitializer] = None,
        class_names: Optional[List[str]] = None,
        nc: Optional[int] = None,
        csv_path: Optional[str] = None,
    ) -> BayesianResult:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        model_name = model_result.model_name
        freeze_bn = model_result.freeze_bn
        robustness_evaluator = RobustnessEvaluator(self.cfg)

        epochs_min, epochs_max = self.bs_cfg["epochs"]
        freeze_choices: List[str] = self.bs_cfg["freeze"]
        lr0_min, lr0_max = self.bs_cfg["lr0"]
        geo = self.bs_cfg["geometry"]

        def objective(trial: optuna.Trial) -> float:
            epochs = trial.suggest_int("epochs", int(epochs_min), int(epochs_max))
            freeze = trial.suggest_categorical("freeze", freeze_choices)
            lr0 = trial.suggest_float("lr0", lr0_min, lr0_max, log=True)
            perspective = trial.suggest_float(
                "perspective", geo["perspective"][0], geo["perspective"][1]
            )
            scale = trial.suggest_float("scale", geo["scale"][0], geo["scale"][1])
            translate = trial.suggest_float(
                "translate", geo["translate"][0], geo["translate"][1]
            )
            degrees = trial.suggest_float(
                "degrees", geo["rotation"][0], geo["rotation"][1]
            )

            cv_scores: List[float] = []
            rob_scores: List[float] = []

            for fold in folds:
                head_cb = head_initializer.make_callback(model_name) if head_initializer else None

                result = run_training(
                    model_name=model_name,
                    data_yaml=fold.data_yaml,
                    fold_idx=fold.fold_idx,
                    epochs=epochs,
                    freeze=freeze,
                    freeze_bn=freeze_bn,
                    lr0=lr0,
                    augment_params={
                        "perspective": perspective,
                        "scale": scale,
                        "translate": translate,
                        "degrees": degrees,
                    },
                    run_dir=self.run_dir,
                    cfg=self.cfg,
                    head_init_callback=head_cb,
                    run_name_prefix=f"{trial.number + 1:03d}",
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
                        logger.warning(f"Robustness eval failed in trial {trial.number}: {e}")
                rob_scores.append(rob)

                if result.save_dir:
                    delete_run_dir(result.save_dir)

            mean_rob = statistics.mean(rob_scores) if rob_scores else 0.0
            composite = compute_composite_score(cv_scores, mean_rob, self.cfg)

            trial.set_user_attr("cv_mean", statistics.mean(cv_scores) if cv_scores else 0.0)
            trial.set_user_attr("cv_std", statistics.stdev(cv_scores) if len(cv_scores) > 1 else 0.0)
            trial.set_user_attr("robustness", mean_rob)

            return composite

        def on_trial_complete(
            study: optuna.Study, trial: optuna.trial.FrozenTrial
        ) -> None:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                return
            cv_mean = trial.user_attrs.get("cv_mean", 0.0)
            cv_std  = trial.user_attrs.get("cv_std", 0.0)
            rob     = trial.user_attrs.get("robustness", 0.0)
            logger.info(
                f"Trial {trial.number + 1:>4d}/{self.n_trials} | "
                f"score={trial.value:.4f} | "
                f"{self.metric_key}={cv_mean:.4f} ± {cv_std:.4f} | "
                f"rob={rob:.4f}"
            )
            if csv_path is None:
                return
            append_csv_row(csv_path, [
                f"trial_{trial.number:04d}", "phase2",
                f"{trial.value:.6f}",
                f"{cv_mean:.6f}",
                f"{cv_std:.6f}",
                f"{rob:.6f}",
            ])

        Path(self.run_dir).mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{self.run_dir}/optuna_study.db"

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.cfg["compute"]["seed"]),
            study_name="bayesian_search",
            storage=storage,
            load_if_exists=True,
        )

        remaining = max(0, self.n_trials - len(study.trials))
        if remaining == 0:
            logger.info("Bayesian search already complete (loaded from storage).")
        else:
            logger.info(
                f"Bayesian search: {remaining} trials remaining "
                f"({len(study.trials)} already done)."
            )
            study.optimize(
                objective,
                n_trials=remaining,
                callbacks=[on_trial_complete],
                show_progress_bar=False,
            )

        best = study.best_params
        logger.info(
            f"Bayesian search best: epochs={best['epochs']}, freeze={best['freeze']}, "
            f"lr0={best['lr0']:.2e}, perspective={best['perspective']:.5f}, "
            f"scale={best['scale']:.3f}, translate={best['translate']:.3f}, "
            f"degrees={best['degrees']:.2f} | score={study.best_value:.4f}"
        )

        return BayesianResult(
            epochs=best["epochs"],
            freeze=best["freeze"],
            lr0=best["lr0"],
            perspective=best["perspective"],
            scale=best["scale"],
            translate=best["translate"],
            degrees=best["degrees"],
            best_score=study.best_value,
        )
