"""YOLOv8m assembled in pure PyTorch from explicitly enumerated layers.

This is the equivalent of taking ultralytics/cfg/models/v8/yolov8.yaml,
applying the yolov8m scaling rules (depth=0.67, width=0.75, max_channels=768),
and writing out the resulting nn.Module by hand. No YAML parsing, no registry.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

try:
    from .modules import C2f, Conv, Detect, SPPF
except ImportError:  # allow `python model.py` from this directory
    from modules import C2f, Conv, Detect, SPPF


def _make_divisible(x: float, divisor: int = 8) -> int:
    """Round up to the nearest multiple of `divisor`."""
    return math.ceil(x / divisor) * divisor


def _scale_channels(c: int, width: float = 0.75, max_channels: int = 768) -> int:
    return _make_divisible(min(c, max_channels) * width, 8)


def _scale_depth(n: int, depth: float = 0.67) -> int:
    return max(round(n * depth), 1) if n > 1 else n


class YOLOv8m(nn.Module):
    """YOLOv8 medium-scale object detection model.

    Topology (channel sizes after width=0.75 scaling, max_channels=768):

      backbone:
        L0  Conv 3   -> 48   k=3 s=2          (P1/2,  320x320)
        L1  Conv 48  -> 96   k=3 s=2          (P2/4,  160x160)
        L2  C2f  96  -> 96,  n=2,  shortcut=T
        L3  Conv 96  -> 192  k=3 s=2          (P3/8,   80x80)
        L4  C2f  192 -> 192, n=4,  shortcut=T  --> skip to head
        L5  Conv 192 -> 384  k=3 s=2          (P4/16,  40x40)
        L6  C2f  384 -> 384, n=4,  shortcut=T  --> skip to head
        L7  Conv 384 -> 576  k=3 s=2          (P5/32,  20x20)
        L8  C2f  576 -> 576, n=2,  shortcut=T
        L9  SPPF 576 -> 576, k=5

      head (PAN-FPN, top-down then bottom-up):
        L10 Upsample(x2)                                (576, 40x40)
        L11 Concat(L10, L6)                             (960, 40x40)
        L12 C2f  960 -> 384, n=2,  shortcut=F           --> skip to L17
        L13 Upsample(x2)                                (384, 80x80)
        L14 Concat(L13, L4)                             (576, 80x80)
        L15 C2f  576 -> 192, n=2,  shortcut=F   ===> P3 detect input
        L16 Conv 192 -> 192, k=3, s=2                   (192, 40x40)
        L17 Concat(L16, L12)                            (576, 40x40)
        L18 C2f  576 -> 384, n=2,  shortcut=F   ===> P4 detect input
        L19 Conv 384 -> 384, k=3, s=2                   (384, 20x20)
        L20 Concat(L19, L9)                             (960, 20x20)
        L21 C2f  960 -> 576, n=2,  shortcut=F   ===> P5 detect input
        L22 Detect(nc, ch=[192, 384, 576])
    """

    DEPTH = 0.67
    WIDTH = 0.75
    MAX_CHANNELS = 768
    STRIDES = (8, 16, 32)

    def __init__(self, nc: int = 80, ch_in: int = 3):
        super().__init__()
        self.nc = nc

        sc = _scale_channels                                       # short alias
        sd = lambda n: _scale_depth(n, self.DEPTH)                 # noqa: E731

        # ---- Backbone ----
        self.b0 = Conv(ch_in,    sc(64),   k=3, s=2)               # P1
        self.b1 = Conv(sc(64),   sc(128),  k=3, s=2)               # P2
        self.b2 = C2f (sc(128),  sc(128),  n=sd(3), shortcut=True)
        self.b3 = Conv(sc(128),  sc(256),  k=3, s=2)               # P3
        self.b4 = C2f (sc(256),  sc(256),  n=sd(6), shortcut=True) # -> skip P3
        self.b5 = Conv(sc(256),  sc(512),  k=3, s=2)               # P4
        self.b6 = C2f (sc(512),  sc(512),  n=sd(6), shortcut=True) # -> skip P4
        self.b7 = Conv(sc(512),  sc(1024), k=3, s=2)               # P5
        self.b8 = C2f (sc(1024), sc(1024), n=sd(3), shortcut=True)
        self.b9 = SPPF(sc(1024), sc(1024), k=5)                    # -> skip P5

        # ---- Head ----
        self.up   = nn.Upsample(scale_factor=2, mode="nearest")    # stateless, reused

        # top-down P5 -> P4
        self.h12 = C2f(sc(1024) + sc(512), sc(512), n=sd(3), shortcut=False)

        # top-down P4 -> P3 (output for small objects)
        self.h15 = C2f(sc(512) + sc(256), sc(256), n=sd(3), shortcut=False)

        # bottom-up P3 -> P4 (output for medium objects)
        self.h16 = Conv(sc(256), sc(256), k=3, s=2)
        self.h18 = C2f(sc(256) + sc(512), sc(512), n=sd(3), shortcut=False)

        # bottom-up P4 -> P5 (output for large objects)
        self.h19 = Conv(sc(512), sc(512), k=3, s=2)
        self.h21 = C2f(sc(512) + sc(1024), sc(1024), n=sd(3), shortcut=False)

        # ---- Detect head ----
        self.detect = Detect(nc=nc, ch=(sc(256), sc(512), sc(1024)))
        self.detect.stride = torch.tensor(self.STRIDES, dtype=torch.float)
        self.detect.bias_init(image_size=640)

    def forward(self, x: torch.Tensor):
        # backbone
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        p3 = self.b4(x)   # stride 8 features
        x = self.b5(p3)
        p4 = self.b6(x)   # stride 16 features
        x = self.b7(p4)
        x = self.b8(x)
        p5 = self.b9(x)   # stride 32 features

        # top-down path
        u1 = self.up(p5)
        n4 = self.h12(torch.cat([u1, p4], dim=1))    # mid P4 feature
        u2 = self.up(n4)
        out_p3 = self.h15(torch.cat([u2, p3], dim=1))

        # bottom-up path
        d1 = self.h16(out_p3)
        out_p4 = self.h18(torch.cat([d1, n4], dim=1))
        d2 = self.h19(out_p4)
        out_p5 = self.h21(torch.cat([d2, p5], dim=1))

        return self.detect([out_p3, out_p4, out_p5])
