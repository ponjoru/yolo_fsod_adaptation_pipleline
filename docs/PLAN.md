```markdown
# Auto-Adaptation Pipeline for Presale Video Analytics Detector Training

## Goal

Automatically adapt an object detector to a customer-specific video scene using:
- 5–10 minute video (from a single scene mainly)
- minimal/no ML engineer involvement
- limited data (≥30 labelled frames per class)
- possible lack of viewpoint diversity

Primary objective:
- maximize robustness for the target scene
- preserve limited generalization to small camera/view changes
- avoid catastrophic scene memorization

Deployment context:
- overnight automated run (not live demo)
- operated by ML engineers, not customers
- output: ONNX weights + debug logs

---

# High-Level Pipeline

```text
sampled frames from a single video + negatives (optionally frames from a video with another view)
→ temporal 3-fold CV recipe search (phase 1: grid, phase 2: Bayesian)
→ robustness evaluation
→ best recipe selection
→ retrain on full dataset
→ export adapted detector (ONNX)
```

---

# 1. Input Data

## Dataset Directory Structure

The pipeline expects a single YOLO-format dataset directory:

```text
dataset/
├── images/
│   ├── train/            # labelled frames — required
│   ├── negatives/        # background-only frames — optional
│   └── new_view_train/   # side-angle frames — optional
├── labels/
│   └── train/            # YOLO-format label files matching images/train/
├── subsets/
│   ├── train.txt         # fixed train split (for debug/reproducibility)
│   └── val.txt           # fixed val split (for debug/reproducibility)
└── data.yaml             # YOLO dataset config (class names, nc, paths)
```

## Notes on Structure

- `images/train/` — source for all temporal CV splits, computed dynamically at runtime
- `images/negatives/` — paired with empty label files; ratio controlled by config
- `images/new_view_train/` — presence is auto-detected; activates optional multi-view path
- `labels/train/` — one `.txt` file per image in `images/train/`, YOLO format
- `subsets/train.txt` and `subsets/val.txt` — base fixed splits for deterministic debugging only; not used in the main pipeline search
- `data.yaml` — standard YOLO dataset config; class names are read from here and cross-referenced with the class mapping in config

## Frame Naming Convention
- Format: `{video_id}_{frame_id}.png` (e.g. `002_01067.png`)
- Frame index is parsed from the filename suffix for temporal ordering within temporal CV

## Data Requirements
- Minimum: 30 labelled frames per class in `images/train/`
- Labels in YOLO format (class_id cx cy w h, normalised)

## Frame Sampling (out of scope for now)
Frames are sampled outside the pipeline using activityScore evaluation and k-means filtering.
This step may be integrated into the pipeline in the future.

---

# 2. Class Mapping

## Strategy

Start from COCO-pretrained weights.

Config allows specifying a mapping from target class name to one or more COCO class names:

```yaml
classes:
  mapping:
    "person": "person"           # 1-to-1: reuse COCO head weights directly
    "vehicle": ["car", "truck", "bus"]  # many-to-1: merge COCO head weights
```

- If a target class is in the mapping: reuse pretrained detection head weights
- If a target class has no mapping (novel class): randomly initialize the head
- No previous COCO classes need to be preserved — this is a class-replacement fine-tune

---

# 3. Dataset Construction

## Negative Frames

- Loaded from `images/negatives/` with empty label files
- A `negative_ratio` config field controls the target ratio of negatives to positives
- If the available negatives are insufficient to meet the ratio:
  - Print a warning with a tip (e.g. "Add more images to images/negatives/; current ratio is X, target is Y")
  - Proceed with all available negatives — never subsample positives

## Temporal Split Strategy

DO NOT use random split.

Use temporal blocks derived from frame index parsed from filenames in `images/train/`.

Enforce a `min_gap_frames` parameter at block boundaries to prevent near-identical frames leaking between train and val.

---

# 4. Temporal Cross Validation

## Purpose

Temporal CV is used for:
- recipe selection (hyperparameter search)
- stability estimation
- overfit detection

NOT for:
- unbiased benchmark reporting

## Setup

3-fold temporal CV. Frames sorted by frame index, split into 3 consecutive temporal blocks.

All 3 rotations are used (each block serves as validation exactly once):

```text
Rotation 1: train = [block_1, block_2], val = [block_3]
Rotation 2: train = [block_1, block_3], val = [block_2]
Rotation 3: train = [block_2, block_3], val = [block_1]
```

Bidirectional splits (val on earlier frames) are acceptable because:
- this is not a forecasting task
- we want distributional coverage across the full temporal range (lighting changes, etc.)
- the goal is robustness, not temporal prediction

The `min_gap_frames` parameter enforces a minimum frame index gap at every train/val boundary, regardless of direction, to prevent consecutive-frame leakage.

---

# 5. Hyperparameter Search

## Two-Phase Strategy

### Phase 1 — Grid Search

Exhaustive grid over discrete structural choices:

```yaml
models: [yolo11s, yolo11m, yolo11l]
epochs: [5, 10, 15, 20, 25, 30, 35]
freeze: [full_backbone, partial_backbone, early_only, full_finetune]
freeze_bn: [true, false]
```

Total combinations: 3 × 7 × 4 × 2 = 168 per fold × 3 folds = 504 training runs.

Each run is fast given small dataset size (~30 frames). No early pruning for now — simple exhaustive search. Optimisations (e.g. ASHA, Median Pruner) can be added later if needed.

### Phase 2 — Bayesian Search (Optuna)

After selecting the best model/freeze/epochs/freeze_bn from phase 1, run Bayesian optimisation over geometry augmentation hyperparameters only:

```yaml
bayesian_search:
  n_trials: 50  # configurable
  geometry:
    perspective: [min, max]
    scale: [min, max]
    translate: [min, max]
    rotation: [min, max]
```

Phase 2 fixes all phase 1 decisions and only searches geometry.

## Time Budget

Up to 12 hours is acceptable for the full search.

---

# 6. Recommended Augmentation Philosophy

## IMPORTANT

The main challenge is viewpoint robustness. Geometry augmentations are critical.

## Recommended augmentations

```yaml
scale: 0.3–0.5
translate: 0.1
perspective: 0.001–0.003
rotation: small
blur: enabled
compression: enabled
brightness/contrast: enabled
```

## Reduce or disable

```yaml
mosaic: 0.0   # often hurts few-shot scene adaptation
mixup: 0.0
copy_paste: false
```

These can optionally be re-enabled via config for experimentation.

---

# 7. Synthetic Robustness Probes

## Goal

Estimate invariance to small scene changes.

NOT intended as realistic evaluation.

## Perturbations

Apply to validation frames:
- perspective warp
- crop
- blur
- compression
- brightness shift
- scale perturbation

Each perturbation is individually toggle-able in the ML config.

## Measure

```text
robustness_score = mAP on synthetically perturbed validation frames
```

Uses the same metric as CV scoring (map50 or map, configurable).

---

# 8. Composite Recipe Score

## Formula

```text
score = cv_mean_weight * cv_mean_map
      - cv_std_weight  * cv_std
      + robustness_weight * robustness_map
```

Default weights:
- `cv_mean_weight`: 0.5
- `cv_std_weight`: 0.2  (penalty — higher variance is worse)
- `robustness_weight`: 0.2

All weights are configurable in the ML config.

The score is used for ranking only, not as an absolute metric — normalization is not required.

## Metric

Configurable via `scoring.metric`:
- `map50` — mAP at IoU 0.50 (more forgiving on localization, more interpretable for CCTV)
- `map` — mAP at IoU 0.50:0.95 (stricter, better for precise localization tasks)

## Philosophy

Prioritize:
- stable adaptation (low cv_std)
- robustness (robustness_map)
- consistent CV performance (cv_mean_map)

NOT:
- highest single-fold mAP

---

# 9. Final Training

After selecting best recipe from phase 1 + phase 2:

```text
retrain on full dataset (all folds combined) using selected hyperparameters
```

Output:
- final adapted ONNX weights
- debug logs (for ML engineer inspection only)

---

# 10. Configuration Design

## Two-tier config system

### config_ml.yaml — ML Engineer Config
Full scope. All knobs exposed. Used internally.

Sections:
- `data`: dataset_dir, negative_ratio, min_gap_frames
- `classes`: mapping (target → COCO class or list)
- `cv`: n_folds, min_gap_frames
- `grid_search`: models, epochs, freeze strategies, freeze_bn
- `bayesian_search`: n_trials, geometry aug bounds
- `scoring`: weights, metric (map50 or map)
- `augmentations`: mosaic, mixup, copy_paste, blur, compression, brightness_contrast
- `robustness_probes`: per-perturbation toggles
- `export`: format (onnx), output_dir
- `compute`: device, workers
- `logging`: verbose, save_dir

### config_user.yaml — Project Manager Config
Strict subset of ML config. Overrides ML defaults. PM-facing (basic terminal/Python knowledge assumed).

Exposed fields only:
- `data.dataset_dir`: path to the YOLO dataset directory (see dataset structure in Section 1)
- `classes`: mapping
- `bayesian_search`: n_trials
- `export`: output_dir
- `compute`: device

The pipeline derives all internal paths (`images/train/`, `images/negatives/`, `labels/train/`, `data.yaml`, `subsets/`) from `dataset_dir` automatically.

## Merge strategy
Pipeline loads config_ml.yaml as base, deep-merges config_user.yaml on top.

---

# 11. Logging

- Logs are saved alongside run output: `{output_dir}/{run_id}/debug.log`
- Controlled by `logging.verbose` flag in ML config (default: false)
- Not shown to end user — intended for ML engineer debugging only
- Per-trial metrics are logged for post-hoc analysis

---

# 12. Export

- Format: ONNX
- Saved to `export.output_dir`

---

# 13. Optional Multi-View Support

If additional side-angle video/images exist:

## Recommended usage

```text
new_view_train   (small portion — calibration/training)
new_view_holdout (larger portion — OOD evaluation)
```

## IMPORTANT

Do not train on all new-view frames — the holdout must remain a real robustness signal.

---

# 14. Phone Camera Data

Phone images/videos are useful for viewpoint diversity and pose expansion, but should not dominate training distribution.

## Recommended approach

Convert phone frames toward CCTV style:
- blur, compression, downscale, noise, contrast reduction

Use mostly as:
- geometry source
- synthetic augmentation source

---

# 15. Optional Synthetic Object-Centric Augmentation

If masks are available:

```text
mask → object crop → perspective warp → paste on CCTV backgrounds → recompute bbox
```

Useful for:
- small viewpoint changes
- position invariance
- reducing background memorization

---

# 16. Expected Limitations

Single-view training cannot reliably produce:
- large viewpoint invariance
- strong 3D understanding

If object appearance changes significantly under camera rotation, additional views are strongly recommended.

---

# 17. Production Philosophy

This is NOT intended to:
- produce SOTA detectors
- maximize benchmark mAP

This IS intended to:
- provide robust automatic adaptation
- minimize manual ML involvement
- support scalable presale workflows
- produce predictable/stable behaviour

---

# Future Work (Not In Scope Now)

## Frame Sampling Integration
Auto-sample frames from raw video using activityScore + k-means inside the pipeline.

## Temporal Stability Metrics
Estimate detector stability without dense annotations:

- **Detection flicker**: detected ↔ missing transitions
- **Confidence jitter**: frame-to-frame confidence variation
- **Box jitter**: IoU(box_t, box_t-1)
- **Track fragmentation**: via ByteTrack / BoT-SORT — fragmented tracks, longest continuous track
- **Gap statistics**: max/mean gap, number of gaps

## Search Optimisation
ASHA scheduler, Median Pruner, or other early-stopping strategies to reduce phase 1 runtime if 12 hours proves insufficient.
```
