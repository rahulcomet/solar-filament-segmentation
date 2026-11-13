# Solar filament instance segmentation

This solution uses YOLO11s-seg and a mask-guided U-Net to segment individual filaments in GONG H-alpha images. The selected recipe uses the original 2048 view, a horizontally flipped 2048 view, and the same unflipped image upscaled to 2560 for the detector. A two-of-three pixel vote adjusts matching instance boundaries while preserving disjoint masks.

| Recipe | Local PQ | TP | FP | FN | Public Kaggle PQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original view | 0.458323 | 976 | 413 | 546 | 0.37 |
| Mirror admissions | 0.459652 | 990 | 430 | 532 | 0.37 |
| **Three-view boundary vote** | **0.463374** | **994** | **426** | **528** | **0.37** |
| Vote with additional isolated admissions | 0.465434 | 1013 | 446 | 509 | 0.37 |

Local PQ uses 146 physical images and 234 separate annotation records held out by observation day. Kaggle listed the three-view boundary vote first in the submissions view on November 14. The displayed scores do not reveal unrounded differences.

## Reproduce the selected CSV

Use Python 3.11 and an NVIDIA GPU. The pinned environment was verified with PyTorch 2.6.0 and CUDA 12.6 on an RTX 3060 Ti. Put the provided competition test images in `data/test/images`. Both trained checkpoints are included in `weights/`. The runner checks their SHA-256 hashes before inference.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe reproduce.py --images data/test/images --output submission.csv --variant boundary-vote
```

On Linux, use `.venv/bin/python`. The [notebook](pipeline.ipynb) runs the same path in the project environment. [Recipe](recipe.json) records checkpoint hashes and settings. [Export verification](submission-verification.json) records the original CSV hash and its 1,187 disjoint masks across the 180-image test inventory. The CSV uses one row per nonempty instance.

The technical report is a separate upload through the competition's organizer form. Voting misses some faint or fragmented filaments and may change the extent preferred by a given annotator. The project uses the [Ultralytics AGPL-3.0 distribution](https://www.ultralytics.com/license). The supplied competition images and labels are available through [Kaggle](https://www.kaggle.com/competitions/filament-segmentation-2026/data).
