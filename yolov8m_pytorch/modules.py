"""YOLOv8 building blocks reimplemented in pure PyTorch.

Mirrors the structure of ultralytics/nn/modules/{conv,block,head}.py
with no external dependencies beyond torch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def autopad(k, p=None, d: int = 1):
    """'same' padding for stride-1 convs (and conventional k//2 for stride-2).

    Accepts `k` as either an int or a tuple/list per spatial dim.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv2d → BatchNorm2d → SiLU. The atomic unit of YOLOv8."""

    default_act = nn.SiLU()

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None,
                 g: int = 1, d: int = 1, act: bool | nn.Module = True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Two 3x3 Convs with optional residual (only added when c1 == c2)."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1,
                 k: tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP block, faster variant. Splits → n bottlenecks → concat → project.

    Output channel layout fed into cv2:
      [first_half, second_half, bn_out_1, bn_out_2, ..., bn_out_n]
    Total = (2 + n) * c, projected back to c2.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False,
                 g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast.

    Equivalent to SPP(k=(5, 9, 13)) but achieved by chaining 3 identical 5x5
    maxpools so that downstream features have receptive fields of 5/9/13.
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class DFL(nn.Module):
    """Distribution Focal Loss integral: convert reg_max-bin distribution to a scalar."""

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape  # (batch, 4*reg_max, anchors)
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


def make_anchors(feats: list[torch.Tensor], strides: list[int],
                 grid_cell_offset: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-anchor center points and stride values for all FPN levels."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor,
              xywh: bool = True, dim: int = 1) -> torch.Tensor:
    """Convert (left, top, right, bottom) distances into bbox coordinates."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        cxy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([cxy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)


class Detect(nn.Module):
    """YOLOv8 anchor-free decoupled detection head with DFL.

    For each FPN level:
        cv2 branch -> 4 * reg_max channels (per-side distance distribution)
        cv3 branch -> nc channels (class logits, sigmoid at inference)
    """

    def __init__(self, nc: int = 80, ch: tuple[int, ...] = (), reg_max: int = 16):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.zeros(self.nl)  # filled in by caller after a dry forward

        c2 = max(16, ch[0] // 4, reg_max * 4)        # box-branch hidden width
        c3 = max(ch[0], min(nc, 100))                # cls-branch hidden width

        # box regression branch (legacy v8 layout)
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1))
            for x in ch
        )
        # classification branch (legacy v8 layout)
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1))
            for x in ch
        )
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

        self._anchors: torch.Tensor | None = None
        self._strides: torch.Tensor | None = None
        self._cached_shape: tuple | None = None

    def bias_init(self, image_size: int = 640) -> None:
        """Set sensible initial biases. Call after self.stride is populated."""
        for box, cls, s in zip(self.cv2, self.cv3, self.stride):
            box[-1].bias.data[:] = 2.0
            cls[-1].bias.data[: self.nc] = math.log(5 / self.nc / (image_size / float(s)) ** 2)

    def forward(self, x: list[torch.Tensor]):
        # Per-level raw predictions
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x  # raw multi-scale predictions for loss computation

        # Inference: decode to (B, 4 + nc, total_anchors)
        shape = x[0].shape
        if self._anchors is None or self._cached_shape != shape:
            self._anchors, self._strides = make_anchors(x, self.stride.tolist(), 0.5)
            self._anchors = self._anchors.transpose(0, 1)
            self._strides = self._strides.transpose(0, 1)
            self._cached_shape = shape

        bs = x[0].shape[0]
        x_cat = torch.cat([xi.view(bs, self.no, -1) for xi in x], dim=2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), dim=1)
        dbox = dist2bbox(self.dfl(box), self._anchors.unsqueeze(0), xywh=True, dim=1) * self._strides
        return torch.cat((dbox, cls.sigmoid()), 1)
