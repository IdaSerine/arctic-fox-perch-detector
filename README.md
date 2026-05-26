# Arctic Fox Perch Detector (standalone)

Reusable command-line package for the **Perch v2 sliding-window detector** used in the thesis year pipeline (`runs/year_pipeline_1200x600/perch_standalone`).

The pipeline:

1. Load each WAV as **mono** (any sample rate; resampled to **32 kHz** for Perch, same as thesis inference)
2. Slide a **5 s** window with **2.5 s** hop (50% overlap)
3. Extract **Perch v2** embeddings (TensorFlow Hub)
4. Classify with the trained **logistic regression** checkpoint (`lr_nestedcv_final_C0.01.joblib`)
5. Merge overlapping windows and write **CSV label files**

---

## 1. Python version

**Python 3.11** (tested with **3.11.8**).

```bash
python3.11 --version
```

---

## 2. Required packages and versions

See [`requirements.txt`](requirements.txt). Main dependencies:

| Package | Version |
|---------|---------|
| numpy | 2.4.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| joblib | 1.5.3 |
| tensorflow | 2.20.0 |
| tensorflow-hub | 0.16.1 |
| librosa | 0.11.0 |
| soundfile | 0.13.1 |
| tqdm | 4.67.3 |

**GPU:** CUDA-capable GPU required (same as thesis inference). Install includes `jax[cuda12]` for CUDA 12 libraries — matches `models/perch2/requirements.txt`. On first run, Perch v2 is downloaded from TensorFlow Hub (~hundreds of MB, cached under `~/.cache/tfhub_modules`).

### Install

```bash
cd arctic-fox-perch-detector
bash scripts/setup_venv.sh
source .venv/bin/activate   # required — system python has no tensorflow
```

Then run with `python run_detector.py ...` (uses the activated venv).

Or without activating, call the venv python directly:

```bash
.venv/bin/python run_detector.py --audio-dir samples/test_30_files --output-dir output/test_30_files
```

---

## 3. Command-line syntax

```bash
python run_detector.py \
  --audio-dir  /path/to/wav_folder \
  --output-dir /path/to/output_folder
```

### Full example (defaults match year pipeline)

```bash
python run_detector.py \
  --audio-dir  samples/input \
  --output-dir output \
  --checkpoint models/lr_nestedcv_final_C0.01.joblib \
  --threshold 0.01 \
  --hop-ratio 0.5 \
  --merge-iou 0.3 \
  --confidence-mode fox_vs_noise_winner \
  --device gpu \
  --gpu-index 0
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--audio-dir` | (required) | Folder with `.wav` / `.WAV` files (recursive search) |
| `--output-dir` | (required) | Where CSVs are written |
| `--checkpoint` | `models/lr_nestedcv_final_C0.01.joblib` | Trained classifier |
| `--threshold` | `0.01` | Primary threshold (`all_detections.csv`) |
| `--thresholds` | `0.01,0.6` | Comma-separated thresholds (default also writes conf 0.6) |
| `--hop-ratio` | `0.5` | Hop as fraction of 5 s window → 2.5 s hop |
| `--merge-iou` | `0.3` | Merge overlapping same-class detections (IoU) |
| `--confidence-mode` | `fox_vs_noise_winner` | Fox-vs-noise winner-take-all scoring |
| `--no-filter-noise` | off | If set, keep noise-labeled rows |
| `--device` | `gpu` | `gpu` or `cpu` |
| `--gpu-index` | `0` | GPU index when `--device gpu` |
| `--no-recursive` | off | Only scan top level of `--audio-dir` |
| `--no-per-file-csv` | off | Only write `all_detections.csv` |

### Outputs

By default each run writes detections at **two confidence thresholds**: **0.01** (year-pipeline default) and **0.6**.

| File | Description |
|------|-------------|
| `output_dir/all_detections.csv` | All detections at threshold **0.01** |
| `output_dir/all_detections_conf0.60.csv` | All detections at threshold **0.6** |
| `output_dir/labels/<stem>_labels.csv` | Per-file labels at threshold 0.01 |
| `output_dir/labels_conf0.60/<stem>_labels.csv` | Per-file labels at threshold 0.6 |
| `output_dir/run_config.json` | Run parameters and paths |

**CSV columns:** `start_time`, `start_time_formatted`, `end_time`, `end_time_formatted`, `class`, `confidence`, `num_windows`, `file`, `threshold`

- Times are in **seconds** from the start of each WAV file.
- `class` is a fox call type code (`fab`, `fac`, `fgp`, …) or `noise` before filtering.
- `file` is the input WAV **basename**.

---

## 4. Input audio expectations

Same behaviour as `pipelines/detect.py` in the thesis repo (`run_perch_standalone` / year pipeline).

| Property | Requirement |
|----------|-------------|
| **Format** | WAV (`.wav` or `.WAV`) |
| **Channels** | Any; **converted to mono** (channels averaged) |
| **Sample rate** | **Any** (e.g. 16 kHz, 48 kHz — no need to resample beforehand) |
| **Duration** | Any; files shorter than 5 s are **zero-padded** to one window |

You do **not** need to resample files before running the detector. The code resamples every file to **32 kHz** internally before Perch embedding (required by the Perch v2 model). That matches the thesis inference runs on `sigma2_year` and other folders where native rates vary.

**Internal processing (fixed):**

- Resample to **32 kHz** mono for Perch v2
- Window length: **5.0 s**
- Hop: **2.5 s** (`hop_ratio=0.5`)
- Embedding model: [Google Perch v2](https://www.kaggle.com/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2) via TensorFlow Hub

---

## 5. Sample input and expected output

### Quick smoke test (2 short clips)

- [`samples/input/`](samples/input/) — a few seconds each

```bash
python run_detector.py --audio-dir samples/input --output-dir samples/expected_output
```

### 1200 stratified eval subset (30 × 10 min files)

- [`samples/test_30_files/`](samples/test_30_files/) — first 30 files from `nird_eval_4days_stratified_balanced_v4_1200` (~550 MB)
- [`samples/test_30_files/manifest.csv`](samples/test_30_files/manifest.csv) — season / time-of-day / station per file

```bash
python run_detector.py --audio-dir samples/test_30_files --output-dir output/test_30_files
```

**Note:** The `test_30_files` WAVs are large. For Git hosting, consider [Git LFS](https://git-lfs.github.com/) or distribute them separately; a normal git push may be slow or hit size limits.

### Reference outputs (2-file smoke test)

Regenerate with:

```bash
python run_detector.py --audio-dir samples/input --output-dir samples/expected_output
```

- [`samples/expected_output/all_detections.csv`](samples/expected_output/all_detections.csv) (threshold 0.01)
- [`samples/expected_output/all_detections_conf0.60.csv`](samples/expected_output/all_detections_conf0.60.csv) (threshold 0.6)
- [`samples/expected_output/labels/`](samples/expected_output/labels/)
- [`samples/expected_output/labels_conf0.60/`](samples/expected_output/labels_conf0.60/)

---

## Model checkpoint

The included checkpoint is the same as in the thesis run:

- **File:** `models/lr_nestedcv_final_C0.01.joblib`
- **Source:** nested CV on arctic fox data with Perch embeddings (`C=0.01`)
- **Contents:** `classifier`, `label_encoder`, `scaler`, `metadata`

This matches `checkpoint_used` in `runs/year_pipeline_1200x600/perch_standalone/shard_*/combined_results.json`.

---

## Fox class codes

Target classes (default):  
`fab`, `fac`, `facp`, `fag`, `fc`, `fg`, `fgp`, `fsb`, `ftb`, `ftbp`, `fw`, `fwb`, `fwp`

Noise is modeled internally and removed when `filter_noise` is enabled (default).

---

## Repository layout

```
arctic-fox-perch-detector/
  run_detector.py          # CLI entry point
  perch_detector/          # Python package
  models/                  # Trained LR checkpoint
  samples/input/           # 2 short smoke-test WAVs
  samples/test_30_files/       # 30 files from 1200 stratified eval (~550 MB)
  samples/expected_output/ # Example CSV outputs (smoke test)
  requirements.txt
  scripts/setup_venv.sh
```

---

## Troubleshooting

- **No GPU visible:** Install CUDA drivers and a TensorFlow build with GPU support; verify with `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
- **TF Hub download fails:** Ensure network access; corporate proxies may need `HTTP_PROXY` set.
- **Unreadable WAV:** Try re-exporting as PCM WAV; librosa is used as fallback loader.
