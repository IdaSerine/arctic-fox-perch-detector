# test_30_files — stratified eval subset (30 files)

Thirty **10-minute** WAV files from `nird_eval_4days_stratified_balanced_v4_1200` (indices 00001–00030 in `selected_manifest.csv`). They cover mixed seasons, times of day, and stations — same format as the thesis 1200-file eval set.

| Property | Value |
|----------|-------|
| Files | 30 |
| Duration each | 600 s (10 min) |
| Total size | ~550 MB |
| Source | `Master-Thesis/data/arctic_fox/nird_eval_4days_stratified_balanced_v4_1200/audio/` |

### Test run

```bash
python run_detector.py \
  --audio-dir  samples/test_30_files \
  --output-dir output/test_30_files \
  --device gpu --gpu-index 0
```

Expect several minutes on GPU (Perch loads once; ~30 × 10 min files with sliding windows).

See [`manifest.csv`](manifest.csv) for season / time-of-day / station metadata per file.
