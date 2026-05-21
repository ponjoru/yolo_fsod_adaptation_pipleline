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
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import load_config, generate_run_id
from pipeline.dataset import DatasetBuilder
from pipeline.export import export_onnx
from pipeline.grid_search import GridSearcher, ModelSelectionResult
from pipeline.bayesian_search import BayesianSearcher
from pipeline.head_init import HeadInitializer
from pipeline.robustness import RobustnessEvaluator
from pipeline.trainer import run_training


def _setup_logging(cfg: dict, run_id: str) -> None:
    save_dir = Path(cfg["logging"]["save_dir"]) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "debug.log"

    level = logging.DEBUG if cfg["logging"]["verbose"] else logging.INFO
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("ultralytics").setLevel(
        logging.DEBUG if cfg["logging"]["verbose"] else logging.WARNING
    )


def _build_head_initializer(cfg: dict, class_names: list[str]) -> HeadInitializer | None:
    mapping = cfg["classes"].get("mapping", {})
    if not mapping:
        logging.getLogger(__name__).info(
            "No class mapping defined — all heads will use random initialization."
        )
        return None

    # Use the smallest model variant for COCO weight extraction (weights are architecture-specific)
    model_name = cfg["grid_search"]["models"][0]
    weights_file = f"{model_name}.pt"

    try:
        return HeadInitializer(
            coco_weights_path=weights_file,
            target_classes=class_names,
            class_mapping=mapping,
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"HeadInitializer setup failed: {e}. Proceeding with random head init."
        )
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot YOLO adaptation pipeline")
    parser.add_argument("--ml-config", default="config_ml.yaml")
    parser.add_argument("--user-config", default="config_user.yaml")
    args = parser.parse_args()

    cfg = load_config(args.ml_config, args.user_config)

    run_id = generate_run_id(cfg)
    _setup_logging(cfg, run_id)
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("Auto-Adaptation Pipeline")
    log.info(f"Run ID: {run_id}")
    log.info(f"Dataset: {cfg['data']['dataset_dir']}")
    log.info("=" * 60)

    # -----------------------------------------------------------------------
    # Dataset construction
    # -----------------------------------------------------------------------
    log.info("[1/5] Building temporal CV folds...")
    builder = DatasetBuilder(cfg)
    folds = builder.build_folds()

    class_names = builder.class_names
    nc = builder.nc
    log.info(f"  Classes ({nc}): {class_names}")
    log.info(f"  Folds: {len(folds)}, train sizes: {[len(f.train_images) for f in folds]}")
    log.info(f"  Val sizes: {[len(f.val_images) for f in folds]}")

    # -----------------------------------------------------------------------
    # Head initialization setup
    # -----------------------------------------------------------------------
    head_initializer = _build_head_initializer(cfg, class_names)
    if head_initializer:
        log.info(f"  Class mapping: {cfg['classes']['mapping']}")

    # -----------------------------------------------------------------------
    # Debug paths
    # -----------------------------------------------------------------------
    debug_root = Path(cfg["logging"]["save_dir"]) / run_id / "debug" / "probes"

    # -----------------------------------------------------------------------
    # Phase 1: Model selection
    # -----------------------------------------------------------------------
    log.info("[2/5] Phase 1: Model selection...")
    grid_searcher = GridSearcher(cfg)
    model_result: ModelSelectionResult = grid_searcher.run(
        folds=folds,
        head_initializer=head_initializer,
        class_names=class_names,
        nc=nc,
        probe_mosaic_dir=str(debug_root / "phase1"),
    )
    log.info(
        f"  Best model → {model_result.model_name} | "
        f"freeze_bn={model_result.freeze_bn}, "
        f"composite_score={model_result.composite_score:.4f}"
    )

    # -----------------------------------------------------------------------
    # Phase 2: Joint Bayesian search (epochs, freeze, lr0, geometry)
    # -----------------------------------------------------------------------
    log.info("[3/5] Phase 2: Joint Bayesian search...")
    bayesian_searcher = BayesianSearcher(cfg)
    best_recipe = bayesian_searcher.run(
        model_result=model_result,
        folds=folds,
        head_initializer=head_initializer,
        class_names=class_names,
        nc=nc,
    )
    log.info(
        f"  Best recipe → epochs={best_recipe.epochs}, freeze={best_recipe.freeze}, "
        f"lr0={best_recipe.lr0:.2e} | "
        f"perspective={best_recipe.perspective:.5f}, scale={best_recipe.scale:.3f}, "
        f"translate={best_recipe.translate:.3f}, degrees={best_recipe.degrees:.2f} | "
        f"score={best_recipe.best_score:.4f}"
    )

    # -----------------------------------------------------------------------
    # Final training on full dataset
    # -----------------------------------------------------------------------
    log.info("[4/5] Final training on full dataset...")
    full_fold = builder.build_full_train()
    final_run_dir = str(Path(cfg["logging"]["save_dir"]) / "final" / run_id)
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
        run_dir=final_run_dir,
        cfg=cfg,
        head_init_callback=head_cb,
    )
    log.info(
        f"  Final training done: mAP50={final_result.map50:.4f}, "
        f"mAP={final_result.map:.4f}"
    )

    # Final robustness evaluation with prediction overlays on mosaics
    if final_result.weights_path and Path(final_result.weights_path).exists():
        log.info("  Running final robustness evaluation...")
        robustness_evaluator = RobustnessEvaluator(cfg)
        final_rob = robustness_evaluator.evaluate(
            weights_path=final_result.weights_path,
            val_images=folds[0].val_images,
            val_labels=folds[0].val_labels,
            nc=nc,
            class_names=class_names,
            mosaic_dir=str(debug_root / "final"),
            overlay_predictions=True,
        )
        log.info(f"  Final robustness score: {final_rob:.4f}")

    DatasetBuilder.cleanup_fold(full_fold)

    # -----------------------------------------------------------------------
    # ONNX export
    # -----------------------------------------------------------------------
    log.info("[5/5] Exporting to ONNX...")
    if not final_result.weights_path or not Path(final_result.weights_path).exists():
        log.error("Final weights not found — export skipped.")
        sys.exit(1)

    onnx_path = export_onnx(
        weights_path=final_result.weights_path,
        output_dir=cfg["export"]["output_dir"],
        imgsz=cfg["compute"]["imgsz"],
        run_id=run_id,
    )
    log.info(f"  ONNX model: {onnx_path}")

    # Clean up fold temp dirs
    for fold in folds:
        DatasetBuilder.cleanup_fold(fold)

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info(f"  ONNX output : {onnx_path}")
    log.info(f"  Debug log   : {Path(cfg['logging']['save_dir']) / run_id / 'debug.log'}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
