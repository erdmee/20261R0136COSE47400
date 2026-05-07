"""Sanity check: forward pass + parameter count vs the reference YOLOv8m.

Reference (from ultralytics/cfg/models/v8/yolov8.yaml comments):
    YOLOv8m summary: 169 layers, 25,902,640 parameters, 79.3 GFLOPS
"""

import torch

from model import YOLOv8m


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    model = YOLOv8m(nc=80)
    model.eval()

    n_params = count_params(model)
    print(f"Total parameters: {n_params:,}")
    print(f"Reference        : 25,902,640")
    print(f"Match            : {n_params == 25_902_640}")

    # Training-mode forward: returns list of 3 raw multi-scale tensors
    model.train()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        train_out = model(x)
    print("\nTraining-mode output (list of per-FPN-level tensors):")
    for i, t in enumerate(train_out):
        print(f"  level {i}: {tuple(t.shape)}")

    # Inference-mode forward: returns decoded (B, 4 + nc, total_anchors)
    model.eval()
    with torch.no_grad():
        infer_out = model(x)
    print(f"\nInference-mode output: {tuple(infer_out.shape)}")
    print(f"  expected total anchors: {80*80 + 40*40 + 20*20} = {80*80 + 40*40 + 20*20}")
    print(f"  expected channels     : 4 + nc = {4 + 80}")


if __name__ == "__main__":
    main()
