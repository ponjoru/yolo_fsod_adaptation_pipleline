#!/usr/bin/env python3
"""
Auto-Adaptation Pipeline — entry point.

Usage:
    python run_pipeline.py [--ml-config config_ml.yaml] [--user-config config_user.yaml]

Runs:
  1. Dataset construction + temporal CV fold generation
  2. Phase 1: model selection  (model × 3 folds, fixed baseline hyperparameters)
  3. Phase 2: joint Bayesian search (epochs, freeze, lr0, geometry augmentation)
  4. Final retrain on full dataset
  5. ONNX export

Output layout under runs/run_<id>/:
  grid_search/          — train logs for Phase 1 runs
  bayesian_search/      — train logs for Phase 2 runs
  final/                — train log for the final full-dataset run
  robustness_check_images/
    augmented_images/   — letterboxed perturbed val images (Phase 1, no predictions)
    final_predictions/  — same images with model predictions overlaid (final model)
  weights/              — top-k best runs (full Ultralytics run folders)
  results.csv           — per-run metrics for all phases
  main.log              — pipeline orchestration log
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from loguru import logger

from pipeline.config import load_config, generate_run_id
from pipeline.dataset import DatasetBuilder
from pipeline.export import export_onnx
from pipeline.grid_search import GridSearcher, ModelSelectionResult
from pipeline.bayesian_search import BayesianSearcher
from pipeline.head_init import HeadInitializer
from pipeline.demo import run_demo_inference
from pipeline.robustness import RobustnessEvaluator
from pipeline.trainer import run_training, suppress_ultralytics_console
from pipeline.utils import TopKWeightsTracker, append_csv_row, cleanup_run_artifacts


class _InterceptHandler(logging.Handler):
    """Forward stdlib logging records to loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _setup_logging(cfg: dict, run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    log_file = run_root / "main.log"

    level = "DEBUG" if cfg["logging"]["verbose"] else "INFO"

    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} | {message}",
    )

    # Route all stdlib logging (pipeline modules) through loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Must come after basicConfig so propagate=False takes effect on the new root handler
    suppress_ultralytics_console()


def _build_head_initializer(cfg: dict, class_names: list[str]) -> HeadInitializer | None:
    mapping = cfg["classes"].get("mapping", {})
    if not mapping:
        logger.info("No class mapping defined — all heads will use random initialization.")
        return None

    model_name = cfg["grid_search"]["models"][0]
    weights_file = f"{model_name}.pt"

    try:
        return HeadInitializer(
            coco_weights_path=weights_file,
            target_classes=class_names,
            class_mapping=mapping,
        )
    except Exception as e:
        logger.warning(f"HeadInitializer setup failed: {e}. Proceeding with random head init.")
        return None


def _init_csv(csv_path: str, metric_suffix: str) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_name", "phase", "score",
            f"cv_mean_{metric_suffix}",
            f"cv_std_{metric_suffix}",
            f"robustness_{metric_suffix}",
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot YOLO adaptation pipeline")
    parser.add_argument("--ml-config", default="config_ml.yaml")
    parser.add_argument("--user-config", default="config_user.yaml")
    args = parser.parse_args()

    cfg = load_config(args.ml_config, args.user_config)

    run_id = generate_run_id(cfg)
    run_root = Path(cfg["logging"]["save_dir"]) / run_id
    _setup_logging(cfg, run_root)

    logger.info("=" * 60)
    logger.info("Auto-Adaptation Pipeline")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Dataset: {cfg['data']['dataset_dir']}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    grid_search_dir  = str(run_root / "grid_search")
    bayesian_dir     = str(run_root / "bayesian_search")
    final_dir        = str(run_root / "final")
    weights_dir      = str(run_root / "weights")
    augmented_dir    = str(run_root / "robustness_check_images" / "augmented_images")
    final_pred_dir   = str(run_root / "robustness_check_images" / "final_predictions")
    csv_path         = str(run_root / "results.csv")

    metric = cfg["scoring"]["metric"]
    metric_suffix = "map50" if metric == "map50" else "map"
    _init_csv(csv_path, metric_suffix)

    top_k = cfg["logging"].get("top_k_weights", 3)
    weights_tracker = TopKWeightsTracker(weights_dir=weights_dir, k=top_k)

    # ------------------------------------------------------------------
    # 1. Dataset construction
    # ------------------------------------------------------------------
    logger.info("[1/5] Building temporal CV folds...")
    builder = DatasetBuilder(cfg)
    folds = builder.build_folds()

    class_names = builder.class_names
    nc = builder.nc
    logger.info(f"  Classes ({nc}): {class_names}")
    logger.info(f"  Folds: {len(folds)}, train sizes: {[len(f.train_images) for f in folds]}")
    logger.info(f"  Val sizes: {[len(f.val_images) for f in folds]}")

    head_initializer = _build_head_initializer(cfg, class_names)
    if head_initializer:
        logger.info(f"  Class mapping: {cfg['classes']['mapping']}")

    # ------------------------------------------------------------------
    # 2. Phase 1: Model selection
    # ------------------------------------------------------------------
    logger.info("[2/5] Phase 1: Model selection...")
    append_csv_row(csv_path, ["--- Phase 1: Grid Search ---", "", "", "", "", ""])

    grid_searcher = GridSearcher(cfg, run_dir=grid_search_dir)
    model_result: ModelSelectionResult = grid_searcher.run(
        folds=folds,
        head_initializer=head_initializer,
        class_names=class_names,
        nc=nc,
        probe_mosaic_dir=augmented_dir,
        csv_path=csv_path,
        weights_tracker=weights_tracker,
    )
    logger.info(
        f"  Best model → {model_result.model_name} | "
        f"freeze_bn={model_result.freeze_bn}, "
        f"composite_score={model_result.composite_score:.4f}"
    )

    # ------------------------------------------------------------------
    # 3. Phase 2: Joint Bayesian search
    # ------------------------------------------------------------------
    logger.info("[3/5] Phase 2: Joint Bayesian search...")
    append_csv_row(csv_path, ["--- Phase 2: Bayesian Search ---", "", "", "", "", ""])

    bayesian_searcher = BayesianSearcher(cfg, run_dir=bayesian_dir)
    best_recipe = bayesian_searcher.run(
        model_result=model_result,
        folds=folds,
        head_initializer=head_initializer,
        class_names=class_names,
        nc=nc,
        csv_path=csv_path,
        weights_tracker=weights_tracker,
    )
    logger.info(
        f"  Best recipe → epochs={best_recipe.epochs}, freeze={best_recipe.freeze}, "
        f"lr0={best_recipe.lr0:.2e} | "
        f"perspective={best_recipe.perspective:.5f}, scale={best_recipe.scale:.3f}, "
        f"translate={best_recipe.translate:.3f}, degrees={best_recipe.degrees:.2f} | "
        f"score={best_recipe.best_score:.4f}"
    )

    # ------------------------------------------------------------------
    # 4. Final training on full dataset
    # ------------------------------------------------------------------
    logger.info("[4/5] Final training on full dataset...")
    append_csv_row(csv_path, ["--- Final Training ---", "", "", "", "", ""])

    full_fold = builder.build_full_train()
    head_cb = head_initializer.make_callback() if head_initializer else None

    final_result = run_training(
        model_name=model_result.model_name,
        data_yaml=full_fold.data_yaml,
        fold_idx=-1,
        epochs=best_recipe.epochs,
        freeze=best_recipe.freeze,
        freeze_bn=model_result.freeze_bn,
        lr0=best_recipe.lr0,
        augment_params={
            "perspective": best_recipe.perspective,
            "scale": best_recipe.scale,
            "translate": best_recipe.translate,
            "degrees": best_recipe.degrees,
        },
        run_dir=final_dir,
        cfg=cfg,
        head_init_callback=head_cb,
    )
    logger.info(
        f"  Final training done: mAP50={final_result.map50:.4f}, "
        f"mAP={final_result.map:.4f}"
    )

    # Final robustness evaluation — must happen before cleanup so weights are available
    final_rob = 0.0
    if final_result.weights_path and Path(final_result.weights_path).exists():
        logger.info("  Running final robustness evaluation...")
        robustness_evaluator = RobustnessEvaluator(cfg)
        final_rob = robustness_evaluator.evaluate(
            weights_path=final_result.weights_path,
            val_images=folds[0].val_images,
            val_labels=folds[0].val_labels,
            nc=nc,
            class_names=class_names,
            mosaic_dir=final_pred_dir,
            overlay_predictions=True,
        )
        logger.info(f"  Final robustness score: {final_rob:.4f}")

    final_score = final_result.map50 if metric == "map50" else final_result.map
    append_csv_row(csv_path, [
        "final", "final",
        f"{final_score:.6f}", f"{final_score:.6f}", "0.000000", f"{final_rob:.6f}",
    ])

    # ------------------------------------------------------------------
    # 5. ONNX export — before cleanup so weights_path is still valid
    # ------------------------------------------------------------------
    logger.info("[5/5] Exporting to ONNX...")
    if not final_result.weights_path or not Path(final_result.weights_path).exists():
        logger.error("Final weights not found — export skipped.")
        sys.exit(1)

    onnx_path = export_onnx(
        weights_path=final_result.weights_path,
        output_dir=cfg["export"]["output_dir"],
        imgsz=cfg["compute"]["imgsz"],
        run_id=run_id,
    )
    logger.info(f"  ONNX model: {onnx_path}")

    # Demo inference — before cleanup so weights_path is still valid
    demo_dir = str(Path(cfg["data"]["dataset_dir"]) / "demo")
    if Path(demo_dir).exists():
        logger.info("Running demo inference...")
        demo_paths = run_demo_inference(
            weights_path=final_result.weights_path,
            demo_dir=demo_dir,
            output_dir=str(run_root / "demo_predictions"),
            conf_threshold=cfg.get("demo", {}).get("conf_threshold", 0.25),
            imgsz=cfg["compute"]["imgsz"],
            device=cfg["compute"]["device"],
        )
        if demo_paths:
            logger.info(f"  Demo predictions: {len(demo_paths)} video(s) saved.")

    # Top-k tracking + cleanup — after export so weights_path is still valid above
    if final_result.save_dir:
        weights_tracker.consider(final_score, Path(final_result.save_dir).name, final_result.save_dir)
        cleanup_run_artifacts(final_result.save_dir)

    DatasetBuilder.cleanup_fold(full_fold)
    for fold in folds:
        DatasetBuilder.cleanup_fold(fold)

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info(f"  ONNX output : {onnx_path}")
    logger.info(f"  Run dir     : {run_root}")
    logger.info(f"  Main log    : {run_root / 'main.log'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
