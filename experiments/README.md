# experiments/ — SeaDronesSee Ablation

Training and evaluation pipeline for four YOLOv8m variants on SeaDronesSee
Object Detection v2 (ODV2). Built on top of the official `ultralytics`
Trainer.

```
experiments/
├── cfg/
│   ├── yolov8m.yaml              # baseline
│   ├── yolov8m-p2.yaml           # + P2 head (Detect P2/P3/P4/P5)
│   ├── yolov8m-sppf-k3.yaml      # SPPF kernel 5 -> 3
│   └── yolov8m-p2-sppf-k3.yaml   # P2 + SPPF k=3
├── data/sds.yaml                  # dataset descriptor (5 classes)
├── scripts/
│   ├── convert_sds.py             # COCO JSON -> YOLO txt
│   ├── check_pretrained_transfer.py
│   └── summarize_runs.py          # aggregate per-variant val metrics
├── train.py                       # entry point: --model {baseline|sppf-k3|p2|p2-sppf-k3}
├── weights/                       # drop yolov8m.pt here
└── runs/                          # ultralytics outputs (gitignored)
```

---

## Dataset

**SeaDronesSee Object Detection v2 (ODV2)** — aerial maritime imagery for
search-and-rescue.

| split | images | used here |
|---|---|---|
| train | 8,930  | training  |
| val   | 1,547  | evaluation |
| test  | —      | not used (labels not publicly released) |

The test split's annotations are not publicly released — they are kept
private for the official benchmark leaderboard, which scores predictions
server-side. Since we want reproducible local evaluation across variants,
all numbers in this project are reported on the **val** split.

Classes (`ignored` category is dropped during training):

| id | name                    | size profile |
|----|-------------------------|--------------|
| 0  | swimmer                 | small |
| 1  | boat                    | large |
| 2  | jetski                  | medium |
| 3  | life_saving_appliances  | small, scarce |
| 4  | buoy                    | small |

---

## Training Configuration

All variants train under **identical hyperparameters** so that any metric
difference is attributable to the architecture change alone.

| Item | Value |
|---|---|
| Base model     | YOLOv8m (25.9M params) |
| Pretrained     | COCO `yolov8m.pt`, transfer-learned (shape-mismatched layers re-init) |
| Image size     | 640 |
| Batch          | 16 |
| Epochs         | 100 with `patience=30` early stopping |
| Optimizer      | SGD (lr0=0.01, momentum=0.937, weight_decay=5e-4) |
| LR schedule    | 3-epoch warmup → cosine decay |
| Loss gains     | box=7.5, cls=0.5, dfl=1.5 |
| Augmentation   | Mosaic + HSV + scale/translate + fliplr (flipud disabled) |
| Mosaic close   | last 10 epochs |
| Precision      | AMP (mixed) |
| Hardware       | NVIDIA A100 80GB PCIe MIG 3g.40gb |

See `TRAIN_KW` at the top of `train.py` for the full list.

---

## Usage

### 1. Environment

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch torchvision pyyaml numpy ultralytics opencv-python-headless
```

### 2. COCO-pretrained weights

```bash
mkdir -p experiments/weights
curl -L -o experiments/weights/yolov8m.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt
```

### 3. Prepare data

Download SDS ODV2, then convert per split:

```bash
python experiments/scripts/convert_sds.py \
    --coco /path/to/instances_train.json \
    --images /path/to/images/train \
    --out /path/to/sds --split train

python experiments/scripts/convert_sds.py \
    --coco /path/to/instances_val.json \
    --images /path/to/images/val \
    --out /path/to/sds --split val
```

Then edit the `path:` line in `data/sds.yaml` to point at `/path/to/sds`.

### 4. Sanity-check weight transfer

```bash
python experiments/scripts/check_pretrained_transfer.py
```

Expected: baseline / sppf-k3 transfer ≈ 99.8% (475/475 with one BN counter
excluded). p2 / p2-sppf-k3 ≈ 55% (the P2 branch and the head's P2 input
convs are freshly initialized; everything else transfers).

### 5. Train

```bash
python experiments/train.py --model baseline
python experiments/train.py --model sppf-k3
python experiments/train.py --model p2
python experiments/train.py --model p2-sppf-k3
```

Each run writes to `experiments/runs/<model>/` containing `weights/best.pt`,
`results.csv`, `results.png`, `confusion_matrix.png`, etc.

CLI overrides: `--epochs N`, `--batch B`, `--imgsz S`, `--name NAME`.

### 6. Evaluate

Re-run `model.val()` for every finished variant and print a comparison
table (Markdown + CSV):

```bash
python experiments/scripts/summarize_runs.py \
    --models baseline sppf-k3 p2 p2-sppf-k3
```

Output: three Markdown tables (overall / per-class AP@50 / per-class
AP@50-95) and `experiments/summary.csv`.

See [`RESULTS.md`](RESULTS.md) for the current numbers.
