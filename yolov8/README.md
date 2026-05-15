# yolov8 — slim detection-only baseline

A modular reimplementation of YOLOv8 detection, distilled from the
official `ultralytics` repository. Forward pass and loss are
**bit-identical** to upstream when loading the published
`yolov8{n,s,m,l,x}.pt` weights.

This package is used in the project as a **verification / analysis tool**
for the model and loss math. Actual training in `experiments/` runs on
ultralytics' Trainer using the same YAML configs.

## Contents

| File | Contents |
|---|---|
| `cfg/yolov8.yaml`         | Backbone + head topology + n/s/m/l/x scales |
| `modules/conv.py`         | `Conv` (Conv+BN+SiLU), `Concat`, `autopad` |
| `modules/block.py`        | `Bottleneck`, `C2f`, `SPPF`, `DFL` |
| `modules/head.py`         | `Detect` head (DFL + cls branches, anchor-free) |
| `model.py`                | `parse_model` (yaml → `nn.Sequential`) and `YOLOv8` class |
| `tal.py`                  | `TaskAlignedAssigner`, `make_anchors`, `dist2bbox`, `bbox2dist` |
| `loss.py`                 | `v8DetectionLoss`, `BboxLoss` (CIoU), `DFLoss` |
| `ops.py`                  | `bbox_iou`, `xywh2xyxy`, `xyxy2xywh`, `make_divisible` |
| `verify.py`               | Forward-equivalence verifier vs official `yolov8*.pt` |
| `tests/test_loss_smoke.py`       | End-to-end forward → loss → backward sanity test |
| `tests/test_loss_equivalence.py` | Confirms our loss equals ultralytics' v8DetectionLoss |

## Not included (by design)

- Data pipeline (`YOLODataset`, augmentation, collate)
- Trainer / EMA / AMP scheduling
- Validation metrics (`ap_per_class`, `ConfusionMatrix`, NMS-based eval)
- Non-detection heads (Segment, Pose, OBB, Classify, World, YOLOE, RT-DETR)

For training we use ultralytics directly from `experiments/`.

## Quickstart

```bash
# environment
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install torch torchvision pyyaml numpy ultralytics

# build a model
python -c "
from yolov8 import YOLOv8
import torch
m = YOLOv8(cfg='cfg/yolov8.yaml', scale='m', verbose=False).eval()
print(sum(p.numel() for p in m.parameters()), 'params')
print(m(torch.zeros(1, 3, 640, 640))[0].shape)  # (1, 84, 8400)
"

# verify forward equivalence vs official yolov8m.pt
python -m yolov8.verify --scale m --atol 1e-4

# verify all five scales
python -m yolov8.verify --all --atol 1e-4

# loss equivalence vs ultralytics' v8DetectionLoss
python -m yolov8.tests.test_loss_equivalence
```

## Verification (torch 2.11.0, CPU)

| scale | params (ours) | params (expected) | state_dict keys mapped | max\|Δ output\| |
|---|---|---|---|---|
| n | 3,157,200  | 3,157,200  | 355 / 355 | **0.000e+00** |
| s | 11,166,560 | 11,166,560 | 355 / 355 | **0.000e+00** |
| m | 25,902,640 | 25,902,640 | 475 / 475 | **0.000e+00** |
| l | 43,691,520 | 43,691,520 | 595 / 595 | **0.000e+00** |
| x | 68,229,648 | 68,229,648 | 595 / 595 | **0.000e+00** |

Loss equivalence on yolov8m (fixed random input + 5 random GT boxes):

```
ours   : loss=30.958422  box=2.996883  cls=9.399422  dfl=3.082906
theirs : loss=30.958422  box=2.996883  cls=9.399422  dfl=3.082906
|Δ loss| = 0.000e+00   max|Δ component| = 0.000e+00
```

## Source mapping

Distilled from the following upstream files:

```
ultralytics/cfg/models/v8/yolov8.yaml
ultralytics/nn/modules/{conv,block,head}.py
ultralytics/nn/tasks.py             (parse_model, DetectionModel)
ultralytics/utils/tal.py
ultralytics/utils/loss.py
ultralytics/utils/ops.py            (xywh <-> xyxy, make_divisible)
ultralytics/utils/metrics.py        (bbox_iou)
ultralytics/utils/torch_utils.py    (fuse_conv_and_bn, initialize_weights)
```

Upstream license: AGPL-3.0. This re-package preserves the same license.
