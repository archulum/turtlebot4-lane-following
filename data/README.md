# Dataset

The full dataset is **not stored in git** (≈ 540 MB with full frames). Download it from the project's GitHub Release (or the link in the main README) and unpack it here so the layout is:

```
data/lane_dataset/
├── fwd_1_<timestamp>/
│   ├── labels.csv
│   └── scenes/
│       ├── cropped/   frame_000000.jpg ...   mirror ROI as the network sees it (flipped, 98×215)
│       └── full/      frame_000000.jpg ...   full 250×250 OAK-D preview frame (unflipped)
├── fwd_2_<timestamp>/ ...
└── rev_1_<timestamp>/ ...
```

`training/train.py` only uses `scenes/cropped/`; the full frames are kept for experiments with look-ahead (e.g. combining the mirror's near view with the lower part of the frame).

## Runs

| Run | Direction | Frames | Used for |
|---|---|---:|---|
| fwd_1 | forward | 2719 | train |
| fwd_2 | forward | 1683 | **validation** |
| fwd_3 | forward | 2547 | train |
| rev_1 | reverse | 2618 | train |
| rev_2 | reverse | 1846 | **validation** |
| rev_3 | reverse | 2707 | train |
| **Total** | | **14 120** | 10 591 train / 3 529 val |

Whole runs are held out for validation (no random frame split), so neighbouring, near-identical frames can't leak between train and validation. Frames were recorded at roughly 25 Hz.

## `labels.csv`

| Column | Meaning |
|---|---|
| `frame` | filename in `scenes/cropped/` and `scenes/full/` |
| `stamp_sec` | ROS clock time of the sample |
| `linear_x` | commanded forward speed [m/s] (0.15 while tracking) |
| `angular_z` | commanded turn rate [rad/s] — **the regression target** |
| `line_detected` | 1 if the red line was being tracked |
| `lateral_error` | smoothed, normalised line offset the controller used |

Only frames where the line was being tracked were saved (`save_only_when_tracking=True`).

## Sample

`data/sample/` holds 24 evenly spaced cropped frames and their labels from `fwd_2`, so you can look at the data without downloading anything.
