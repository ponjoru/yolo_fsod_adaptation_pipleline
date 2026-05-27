# Auto-Adaptation Pipeline

Automatically adapts a COCO-pretrained YOLO11 detector to a customer-specific video scene (few-shot pre-sale scenario) with minimal ML engineer involvement. Takes 30+ labelled frames per class and a short scene video as input; outputs a deployment-ready ONNX model.


---

## How it works

```
labelled frames
  → temporal 3-fold CV
  → Phase 1: model selection   (yolo11s / m / l, fixed baseline hyperparameters, 9 runs)
  → Phase 2: Bayesian search   (epochs × freeze × lr0 × geometry augmentation, N trials)
  → Final retrain on full dataset
  → ONNX export
  → Demo video inference (optional)
```

Composite scoring across CV folds penalises variance and rewards robustness under synthetic perturbations (blur, crop, perspective warp, etc.), so the selected recipe favours stable adaptation over peak single-fold mAP.

---

## Project structure

```
adaptation_pipeline/
├── run_pipeline.py          # main entry point — runs the full pipeline
├── run_demo.py              # standalone demo inference with custom conf threshold and tracker
├── config_ml.yaml           # full ML engineer config (all knobs)
├── config_user.yaml         # user-facing config (dataset path, classes, device)
├── requirements.txt
└── pipeline/
    ├── config.py            # config loading and run ID generation
    ├── dataset.py           # temporal CV fold construction
    ├── trainer.py           # single-run YOLO training wrapper
    ├── grid_search.py       # Phase 1: model selection
    ├── bayesian_search.py   # Phase 2: Bayesian hyperparameter search (Optuna)
    ├── robustness.py        # synthetic perturbation evaluation
    ├── scoring.py           # composite score formula
    ├── head_init.py         # COCO → target class head weight transfer
    ├── export.py            # ONNX export
    ├── demo.py              # video inference and annotation
    └── utils.py             # shared utilities (CSV writing)
```

### Run output layout

Each run produces a self-contained directory under `runs/`:

```
runs/
  run_<id>/
    grid_search/             # Phase 1 training runs (one subfolder per fold)
      <run_name>/
        args.json            # exact training arguments passed to YOLO
        fold<n>.log          # Ultralytics training output
    bayesian_search/         # Phase 2 trial runs
      <run_name>/
        args.json
        fold<n>.log
    final/                   # full-dataset retraining run (complete Ultralytics folder)
      weights/
        best.pt
        last.pt
    robustness_check_images/
      augmented_images/      # perturbed val images from Phase 1 (no predictions)
      final_predictions/     # same images with final model predictions overlaid
    demo_predictions/        # annotated output videos (if demo/ folder exists in dataset)
    results.csv              # per-run metrics across all phases
    main.log                 # pipeline orchestration log
```

`run_<id>` is a deterministic 10-character hex hash of the stable config fields (class mapping, search params, seed, etc.), so the same experiment always maps to the same directory.

---

## Dataset structure

```
dataset/
├── images/
│   ├── train/            # labelled frames — required (≥30 per class)
│   ├── negatives/        # background-only frames — optional
│   └── new_view_train/   # side-angle frames — optional (Note: not tested for now)
├── labels/
│   └── train/            # YOLO-format .txt files matching images/train/
├── demo/                 # videos for inference visualization — optional
│   └── <video_name>.mp4  # .mp4 / .avi / .mov / .mkv / .webm / .m4v
└── data.yaml             # YOLO dataset config (nc, names, paths)
```

Frame filenames must follow the `{video_id}_{frame_id}.png` convention (e.g. `002_01067.png`) so the pipeline can infer temporal order for CV splitting.

---

## Configuration

The pipeline uses a two-tier config system. `config_ml.yaml` is the authoritative base; `config_user.yaml` is deep-merged on top of it at runtime.

### config_user.yaml — user-facing

Edit this file for each project. Minimal required change is `data.dataset_dir`.

```yaml
data:
  dataset_dir: "/path/to/your/dataset"   # required

classes:
  mapping:
    "person": "person"                   # target_class: coco_class(es)
    # "vehicle": ["car", "truck", "bus"] # many-to-one merge

bayesian_search:
  n_trials: 50          # total Optuna trials (default 150 in config_ml.yaml)

export:
  output_dir: ./output  # where to write the ONNX file

compute:
  device: "0"           # GPU index or "cpu"
```

### config_ml.yaml — ML engineer config

All pipeline knobs. Key sections:

| Section | Purpose |
|---|---|
| `data` | `negative_ratio`, `min_gap_frames` |
| `cv` | `n_folds` (default 3) |
| `grid_search` | candidate `models`, baseline `epochs`/`freeze`/`lr0`, `freeze_bn` mode |
| `bayesian_search` | `n_trials`, search ranges for `epochs`, `freeze`, `lr0`, geometry augs |
| `scoring` | `metric` (`map50`/`map`), `cv_mean_weight`, `cv_std_weight`, `robustness_weight` |
| `augmentations` | `mosaic`, `mixup`, `copy_paste`, `fliplr`, `flipud` |
| `robustness_probes` | toggle per perturbation type, `n_mosaic_samples` |
| `compute` | `device`, `workers`, `batch`, `imgsz`, `seed` |
| `export` | `format`, `output_dir` |
| `demo` | `conf_threshold` |
| `logging` | `verbose`, `save_dir` |

---

## Installation

```bash
pip install -r requirements.txt
```

PyTorch is included in `requirements.txt` with a CPU/CUDA 12.1 wheel. For a different CUDA version install it manually first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline

```bash
python run_pipeline.py \
  --ml-config config_ml.yaml \
  --user-config config_user.yaml
```

Both flags are optional and default to the filenames above.

The pipeline prints a compact progress summary to stdout. Full detail is in `runs/run_<id>/main.log`. Per-run Ultralytics output goes to `runs/run_<id>/grid_search/<run>/fold<n>.log` (and similarly for Bayesian).

### Re-run demo inference with a different confidence threshold

```bash
python run_demo.py \
  --weights runs/run_<id>/final/weights/best.pt \
  --demo-dir /path/to/dataset/demo \
  --conf 0.35 \
  --output-dir runs/run_<id>/demo_predictions_conf0.35 \
  --tracker tracker
```

Optionally pull `imgsz` and `device` from a config file:

```bash
python run_demo.py \
  --weights runs/run_<id>/final/weights/best.pt \
  --demo-dir /path/to/dataset/demo \
  --ml-config config_ml.yaml \
  --conf 0.40
```

### Resume an interrupted run

Re-run with the same config files. Phase 1 resumes from its checkpoint file (`grid_search/results.jsonl`), Phase 2 resumes from its Optuna SQLite database (`bayesian_search/optuna_study.db`).

---

## Output files

| File | Description |
|---|---|
| `output/best_<run_id>.onnx` | Deployed ONNX model |
| `runs/run_<id>/final/weights/best.pt` | PyTorch weights for demo re-runs or fine-tuning |
| `runs/run_<id>/results.csv` | Per-run scores for all phases |
| `runs/run_<id>/main.log` | Full orchestration log with timestamps |
| `runs/run_<id>/robustness_check_images/` | Visual robustness probe output |
| `runs/run_<id>/demo_predictions/` | Annotated demo videos |
