# Small Object Detection on SeaDronesSee

Course project for Korea University COSE474 (Deep Learning, Spring 2026).

We use YOLOv8m as the baseline and run an architecture-level ablation
targeting **small object detection** on **SeaDronesSee Object Detection v2
(ODV2)** — a maritime aerial dataset for search-and-rescue.

## Motivation

In a maritime SAR setting the drone must accurately detect small targets:
people (`swimmer`), rescue gear (`life_saving_appliances`), markers
(`buoy`). A vanilla YOLOv8m baseline handles large objects (`boat`) well
but shows two characteristic small-object weaknesses:

- a large gap between AP@50 and AP@50-95 — boxes are found but not
  precisely localized;
- low Recall on the rarer small classes — many ground-truth instances are
  missed.

We probe two architectural axes — **receptive field** and **feature-map
resolution** — and measure their per-class effect.

## Variants

| ID | Variant     | Change                                | Hypothesis |
|----|-------------|---------------------------------------|------------|
| M0 | baseline    | unmodified YOLOv8m                    | reference point |
| M1 | sppf-k3     | SPPF kernel 5 → 3                     | smaller receptive field helps small objects |
| M2 | p2          | + P2 detection head (stride 4)        | higher-resolution feature map is the principal remedy |
| M3 | p2-sppf-k3  | P2 + SPPF k=3                         | combined: synergy or conflict |

## Repository Layout

```
yolov8/              # detection-only YOLOv8 reimplementation
                     # (forward + loss verified bit-exact vs official yolov8m.pt)
  ├── model.py            # parse_model + YOLOv8 class
  ├── loss.py             # v8DetectionLoss, BboxLoss, DFLoss
  ├── tal.py              # TaskAlignedAssigner, anchors, dist <-> bbox
  ├── modules/            # Conv, C2f, Bottleneck, SPPF, DFL, Detect
  ├── cfg/yolov8.yaml
  ├── tests/              # forward / loss equivalence vs official weights
  └── verify.py           # COCO weight transfer + forward allclose

experiments/         # SeaDronesSee ablation pipeline (ultralytics Trainer)
  ├── train.py            # entry point: --model {baseline|sppf-k3|p2|p2-sppf-k3}
  ├── cfg/                # 4 architecture yamls
  ├── data/sds.yaml       # dataset descriptor (5 classes)
  ├── scripts/
  │   ├── convert_sds.py             # COCO JSON -> YOLO txt
  │   ├── check_pretrained_transfer.py
  │   └── summarize_runs.py          # aggregate per-variant val metrics
  ├── README.md           # setup + training guide
  └── RESULTS.md          # current numbers + analysis
```

## Method

- **Base model**: YOLOv8m (25.9M params, transfer-learned from COCO
  `yolov8m.pt`).
- **Dataset**: SeaDronesSee ODV2, 5 classes (swimmer, boat, jetski,
  life_saving_appliances, buoy). Train 8,930 images / val 1,547 images.
  The test split is not used in this project — its labels are not
  publicly released (kept private for the official leaderboard), so all
  numbers here are reported on val.
- **Training recipe** (identical across all variants): SGD lr0=0.01 with
  3-epoch warmup → cosine decay, 100 epochs with patience=30, batch=16,
  imgsz=640, AMP, Mosaic augmentation.
- **Evaluation**: best-epoch val mAP@0.5 / mAP@0.5:0.95 plus per-class
  AP; recomputed by re-running `model.val()` on the saved `best.pt`.
- **Hardware**: NVIDIA A100 80GB PCIe MIG 3g.40gb.

## Status

See [`experiments/RESULTS.md`](experiments/RESULTS.md) for tables and
analysis.

- [x] M0 baseline — trained + evaluated
- [x] M1 sppf-k3 — trained + evaluated
- [x] M2 p2 — trained + evaluated (principal contribution validated)
- [ ] M3 p2-sppf-k3 — pending

## Reproduction

```bash
# 1. environment
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch torchvision pyyaml numpy ultralytics opencv-python-headless

# 2. COCO-pretrained weights
mkdir -p experiments/weights
curl -L -o experiments/weights/yolov8m.pt \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt

# 3. dataset (convert SDS ODV2 COCO JSON to YOLO txt; do this for train and val)
python experiments/scripts/convert_sds.py \
    --coco /path/to/instances_train.json \
    --images /path/to/images/train \
    --out /path/to/sds --split train

# 4. train
python experiments/train.py --model baseline
python experiments/train.py --model sppf-k3
python experiments/train.py --model p2
python experiments/train.py --model p2-sppf-k3

# 5. evaluate
python experiments/scripts/summarize_runs.py \
    --models baseline sppf-k3 p2 p2-sppf-k3
```

## License

Built on top of `ultralytics` (AGPL-3.0); this project follows the same
license.
