# Solar filament instance segmentation

This is the three-view boundary-voting entry for the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026). It scored **0.37 public Kaggle PQ** and **0.463374 local PQ** on 146 held-out images with 234 annotation records. Local validation was split by observation day.

YOLO11s-seg proposes masks from the native 2048 × 2048 image, a horizontally flipped copy, and the unflipped image upscaled to 2560 × 2560. A mask-guided U-Net refines each proposal on the original image. Matched masks vote on boundary pixels, and the exporter checks that final instances do not overlap.

## Reproduce the submitted CSV

Use Python 3.11 and an NVIDIA GPU. The pinned environment was verified on an RTX 3060 Ti with PyTorch 2.6.0 and CUDA 12.6. Both checkpoints are included in `weights/`. The runner checks their SHA-256 hashes against [`recipe.json`](recipe.json) before inference.

Clone the repository and run the commands from its root. Set `--images` to the competition's `test_images` folder, with the 180 H-alpha images directly inside it. The example uses `data/test/images`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce.py --images data/test/images --output submission.csv --variant boundary-vote
```

On Linux, use `.venv/bin/python` in place of the Windows environment executable. The CLI requires a new output filename if `submission.csv` already exists.

To run the same pipeline interactively, open [`pipeline.ipynb`](pipeline.ipynb) from the repository root, select the `.venv` Python kernel, and set `IMAGES` to the same folder. Run the cells in order. The notebook uses the same inference and CSV export code as the CLI.

## Verify the output

The submitted CSV contains **1,187 instance rows** with the columns `filament_id` and `segmentation_rle`. All 180 images were processed, and three had no detections. The masks use compressed COCO RLE at 2048 × 2048. Compare the SHA-256 hash of your CSV with [`submission-verification.json`](submission-verification.json).

```powershell
Get-FileHash submission.csv -Algorithm SHA256
```

On Linux, run `sha256sum submission.csv`.

The technical report is a separate upload through the organizer's form. Get the test images from the [Kaggle Data tab](https://www.kaggle.com/competitions/filament-segmentation-2026/data). See [LICENSE](LICENSE) for the code license.
