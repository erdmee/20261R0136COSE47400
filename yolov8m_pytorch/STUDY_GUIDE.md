# YOLOv8m From-Scratch — Study Guide

A walkthrough of this codebase aimed at readers who already understand
ResNet-level PyTorch code (basic blocks, residual connections,
`Conv2d → BatchNorm → ReLU` stacks, the difference between training
and inference modes). The goal is to take you from "I know ResNet" to
"I can read every line of YOLOv8m and explain what it does."

---

## 0. What you should already be comfortable with

Before reading further, make sure you can mentally answer these:

- What does `nn.Conv2d(in, out, k, stride, padding)` do to an input
  `(B, C, H, W)` tensor?
- Why do most modern CNNs use `Conv2d → BatchNorm2d → activation`
  back-to-back?
- In ResNet's `BasicBlock`, what is the residual connection
  (`out = F(x) + x`) and why does it help training?
- How does an FPN-style network use feature maps from multiple
  resolutions of a backbone?

If those feel familiar, you are ready.

---

## 1. The 30-second mental model

YOLOv8m takes one image and outputs the locations and class labels of
every object it can see. That is the whole job.

```
input image  ─►  Backbone   ─►   PAN-FPN Head   ─►   Detect Head   ─►   output
(B, 3, 640, 640)  (extract        (mix features      (predict box       (B, 84, 8400)
                   features)       across scales)     + class)
```

In numbers:

- Input: a batch of `640 × 640` RGB images, shape `(B, 3, 640, 640)`.
- Output (inference): `(B, 84, 8400)` where
    - `84 = 4 (box) + 80 (classes)` — for the COCO dataset's 80 classes
    - `8400 = 80² + 40² + 20²` — predictions are made at three
      resolutions (one prediction per pixel of each feature map).
- The high-resolution map (`80×80`) handles small objects, the
  low-resolution map (`20×20`) handles large objects.

---

## 2. Codebase layout

There are only four files. Read them in this order.

| File | Role | Read when |
|------|------|-----------|
| `modules.py` | Defines the building blocks (`Conv`, `Bottleneck`, `C2f`, `SPPF`, `DFL`, `Detect`). | First — these are the "Lego bricks." |
| `model.py` | Assembles the bricks into the full `YOLOv8m` class with 23 layers wired together. | Second — see how the bricks connect. |
| `verify.py` | Builds the model, runs a forward pass, prints the parameter count and output shapes. | Third — run it to confirm everything works. |
| `__init__.py` | Re-exports the public symbols so `from yolov8m_pytorch import YOLOv8m` works. | Reference only. |

**No external dependencies beyond `torch`.** No YAML parsing, no config
registry, no Ultralytics package. Just pure PyTorch.

---

## 3. The three architectural parts

Object detection networks like YOLOv8 are built as three stages.
Knowing what each stage *exists for* is more important than memorizing
the layer counts.

### 3.1 Backbone — feature extraction

This is the part that looks most like ResNet. It takes the raw image
and progressively reduces resolution while increasing the number of
channels, building up richer and more abstract features at each stage.

Key difference from ResNet:

- ResNet uses `BasicBlock` or `Bottleneck` stacked in stages.
- YOLOv8 uses a custom block called **C2f**, which is similar in
  spirit but has more skip connections inside (more on this in §4.3).

The output of the backbone is **three feature maps at three
resolutions** (called `P3`, `P4`, `P5`), not just one. Detection needs
multiple scales because objects in an image come in many sizes.

### 3.2 Head (PAN-FPN) — multi-scale feature fusion

ResNet only flows information one direction (input → output). But for
detection, a small object detector benefits from "global context"
(what does the rest of the image look like?), and a large object
detector benefits from "fine detail" (where exactly is the edge?). So
the head mixes information *both* directions.

This is called **PAN-FPN** (Path Aggregation Network on top of a
Feature Pyramid Network) and it has two passes:

- **Top-down**: take the deepest, lowest-resolution feature, upsample
  it, and merge into shallower features. This injects high-level
  semantic info into fine-resolution maps.
- **Bottom-up**: take the now-enriched shallow feature, downsample,
  and merge back into deeper features. This injects fine spatial info
  into coarse maps.

### 3.3 Detect head — predicting boxes and classes

The final stage. For every pixel in each of the three output feature
maps, predict:

- 4 numbers describing where a bounding box is (more precisely: the
  distance from this pixel to each of the 4 box edges)
- 80 numbers, one per class, indicating "is this object a *cat*? a
  *car*? ..."

This is the most YOLO-specific part and has two ideas worth learning
in detail:

1. **Anchor-free detection** — unlike YOLOv5, no pre-defined box
   templates. Each pixel directly predicts distances.
2. **Distribution Focal Loss (DFL)** — instead of regressing a single
   number per box edge, predict a *probability distribution* over 16
   possible distances and take the expectation.

---

## 4. Walking through `modules.py`

Now let's read every block in `modules.py` carefully. For each one I
will tell you (a) what it computes, (b) why it exists, and (c) how it
relates to something you already know.

### 4.1 `Conv` — the atomic unit

```python
class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, ...):
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), bias=False)
        self.bn   = nn.BatchNorm2d(c2)
        self.act  = self.default_act
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

This is exactly the pattern you saw in ResNet (`Conv2d → BatchNorm →
ReLU`), with two minor changes:

- The activation is **SiLU** (also called Swish: `x * sigmoid(x)`)
  instead of ReLU. SiLU is smooth and tends to work slightly better
  for object detection.
- `bias=False` on the `Conv2d` because BatchNorm subtracts the mean
  and adds its own learned offset, making a Conv bias redundant.

Helper function `autopad`:

```python
def autopad(k, p=None, d=1):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
```

For stride-1 convs this gives "same" padding (output H/W equals input
H/W). For stride-2 convs it halves the spatial dimensions cleanly,
which is what you want for downsampling.

### 4.2 `Bottleneck` — residual block

```python
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, ...):
        c_ = int(c2 * 0.5)
        self.cv1 = Conv(c1, c_, 3, 1)
        self.cv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y
```

This **is** ResNet's BasicBlock with an extra channel-compression
step:

```
x ──► Conv(c1 → c1/2, 3x3) ──► Conv(c1/2 → c2, 3x3) ──► (+x if shapes match)
```

The "bottleneck" name comes from the channel compression in the
middle — the same naming convention as ResNet's deeper-than-50-layer
variant.

The `self.add and c1 == c2` guard is important: if the input/output
channel counts differ, you cannot do a residual sum without an extra
projection. In YOLOv8, the input/output channel counts match for
backbone Bottlenecks, so the residual is always active there.

### 4.3 `C2f` — the YOLOv8 signature block

This is the block that distinguishes YOLOv8 from YOLOv5. If you only
remember one block from this codebase, remember this one.

```python
class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, ...):
        self.c   = int(c2 * 0.5)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m   = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut) for _ in range(n)
        )
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))         # split into [a, b]
        y.extend(m(y[-1]) for m in self.m)        # b -> Bot1 -> Bot2 -> ...
        return self.cv2(torch.cat(y, 1))          # concatenate ALL of them
```

The forward pass diagram:

```
x
│
▼  cv1 (1x1 conv, doubles channels)
│
├──► chunk(2, dim=1) ────► a (kept aside, plain identity branch)
│                          │
│                    ┌─────┘
│                    │
└──► b ──► Bot1 ──► Bot2 ──► ... ──► BotN
                │      │      │       │
                ▼      ▼      ▼       ▼
   concat all of these in channel dim:
        [a, b, Bot1(b), Bot2(Bot1(b)), ..., BotN(...)]
                │
                ▼  cv2 (1x1 conv, projects back to c2)
                │
              output
```

**Why is this better than just stacking N Bottlenecks?**

In a plain ResNet stage, gradients only flow through the residual
path: a fixed depth from input to output. In `C2f`, the final output
sees *all intermediate Bottleneck outputs* via the concatenation.
Gradient can flow back through paths of varying length (1 conv, 2
convs, 3 convs, ...), which empirically helps optimization, especially
for medium-sized models.

This idea — split, process some branches more than others, concat — is
called **CSP** (Cross Stage Partial Networks). The "f" in C2f stands
for "faster": this is a streamlined form of an earlier `C3` block.

The `shortcut` argument:

- `shortcut=True` in the **backbone** — Bottlenecks have residuals,
  giving stable gradient flow in deep networks.
- `shortcut=False` in the **head** — input/output channels differ
  because of concatenations, so residuals would not match dimensions.

### 4.4 `SPPF` — Spatial Pyramid Pooling Fast

```python
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m   = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
```

This block exists in YOLO but not in ResNet. Its purpose: capture
**multi-scale context** at the very end of the backbone, so the
deepest features know about small *and* large surroundings.

The trick: applying the same 5x5 max-pool three times in a row gives
features with effective receptive fields of 5, 9, and 13 — equivalent
to using three different pooling sizes (which is the original "SPP"),
but cheaper to compute.

```
input ─► cv1 (1x1) ─► y0 ─► pool ─► y1 ─► pool ─► y2 ─► pool ─► y3
                       │            │            │            │
                       └────────────┴────────────┴────────────┘
                              concat (4× channels)
                                       │
                                       ▼
                                 cv2 (1x1) ─► output
```

Always placed at the end of the backbone, just before the head.

### 4.5 `DFL` — Distribution Focal Loss decoder

This is the most YOLO-specific module and the hardest to understand
without context. Read this section twice if needed.

**The problem.** Standard bounding-box regression predicts 4 numbers
per box: `(x, y, w, h)` or `(left, top, right, bottom)` distances.
This single number per side struggles with ambiguous cases (where is
the boundary of a piece of clothing? where exactly does a fluffy dog
end?).

**The DFL idea.** Instead of one number per box edge, predict a
*probability distribution* over 16 discrete distances `(0, 1, 2, ...,
15)`, then take the expected value.

The Detect head's box branch outputs `4 * reg_max = 4 * 16 = 64`
channels per pixel. DFL converts these 64 channels into the 4 distance
values:

```python
class DFL(nn.Module):
    def __init__(self, c1=16):
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)        # [0, 1, 2, ..., 15]
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
    def forward(self, x):
        b, _, a = x.shape
        return self.conv(
            x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
        ).view(b, 4, a)
```

What this does:

1. Reshape `(B, 64, A)` into `(B, 4, 16, A)` — group the 64 channels
   into 4 groups of 16.
2. Apply softmax across the 16 bins → a probability distribution per
   edge per anchor.
3. Apply a fixed `Conv2d(16, 1)` whose weights are `[0, 1, 2, ..., 15]`.
   This is mathematically the dot product of probabilities with bin
   indices = the expected value.

The conv's weights are *frozen*: `requires_grad_(False)`. There is
nothing to learn here — DFL is a deterministic decoder. The actual
learning happens upstream, in the Detect head's box branch, which
learns to output sensible 16-bin distributions.

**Why this works.** A learned distribution can express uncertainty.
For a sharp box edge, the model concentrates probability on one bin.
For a fuzzy edge, it spreads probability over neighboring bins, and
the expected-value decoder gives a smooth interpolation. This produces
both better accuracy and smoother training.

### 4.6 `Detect` — the prediction head

```python
class Detect(nn.Module):
    def __init__(self, nc=80, ch=(192, 384, 576), reg_max=16):
        self.no = nc + reg_max * 4   # = 80 + 64 = 144 outputs per pixel
        c2 = max(16, ch[0] // 4, reg_max * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(   # box branch
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                          nn.Conv2d(c2, 4 * reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(   # class branch
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3),
                          nn.Conv2d(c3, nc, 1))
            for x in ch
        )
        self.dfl = DFL(reg_max)
```

Two ideas worth highlighting:

**Decoupled head.** Notice there are two parallel branches per FPN
level: `cv2` for boxes and `cv3` for classes. Earlier YOLO versions
used a single shared conv that produced both at once. Splitting them
allows each branch to learn representations specialized for its
output, which improves accuracy.

**Anchor-free.** Unlike YOLOv3/v4/v5, there are no predefined "anchor
boxes" of fixed sizes. Each pixel of each feature map predicts
distances to the four edges of the box that contains the object at
that location. The "where" is implicit in the pixel position.

**Two forward modes.** During training, the head returns the raw
multi-scale predictions and the loss function handles all the
decoding:

```python
if self.training:
    return x   # list of 3 tensors with shape (B, 144, H, W)
```

During inference, the head decodes the predictions into actual box
coordinates:

```python
# build a flat list of anchor centers and per-anchor strides
anchors, strides = make_anchors(x, self.stride.tolist(), 0.5)

# split into box-distances and class scores
box, cls = x_cat.split((reg_max * 4, self.nc), dim=1)

# DFL converts 64-channel distributions into 4 distances
# dist2bbox converts (l, t, r, b) distances into (x, y, w, h)
dbox = dist2bbox(self.dfl(box), anchors.unsqueeze(0)) * strides

return torch.cat((dbox, cls.sigmoid()), 1)   # (B, 84, 8400)
```

**Helper functions** (also in `modules.py`):

- `make_anchors(feats, strides)` — for each pixel of each FPN level,
  computes the input-image coordinates of that pixel's center. Returns
  shape `(8400, 2)`.
- `dist2bbox(distance, anchor_points)` — given the 4 distances and the
  anchor center, computes `(x, y, w, h)` of the actual box.

---

## 5. Walking through `model.py`

`modules.py` gave you the bricks. `model.py` assembles them into
`YOLOv8m`.

### 5.1 The scaling rules

YOLOv8 ships in five sizes (n, s, m, l, x). They all share the **same
architecture topology** — same number of layers, same connections —
but differ in width (channels) and depth (number of Bottlenecks per
C2f). The reference yaml file lists numbers for the largest variant
and uses two scaling factors:

```python
def _scale_channels(c, width=0.75, max_channels=768):
    return _make_divisible(min(c, max_channels) * width, 8)

def _scale_depth(n, depth=0.67):
    return max(round(n * depth), 1) if n > 1 else n
```

For the medium variant (yolov8m):

| factor | value | meaning |
|--------|-------|---------|
| `WIDTH = 0.75`  | scale all channel counts to 75% | "thinner network" |
| `DEPTH = 0.67`  | scale all bottleneck repetitions to 67% | "shallower" |
| `MAX_CHANNELS = 768` | hard cap before width scaling | prevents the deepest layers from blowing up |

`_make_divisible(x, 8)` rounds up to the nearest multiple of 8 because
GPU tensor cores are most efficient on channel counts that are
multiples of 8.

Worked example: the yaml says the deepest stage has 1024 channels.

```
min(1024, 768) * 0.75 = 576   →   make_divisible(576, 8) = 576
```

### 5.2 The backbone

```python
self.b0 = Conv(3,    48,   k=3, s=2)   # P1: 640 → 320
self.b1 = Conv(48,   96,   k=3, s=2)   # P2: 320 → 160
self.b2 = C2f (96,   96,   n=2, shortcut=True)
self.b3 = Conv(96,   192,  k=3, s=2)   # P3:  160 → 80
self.b4 = C2f (192,  192,  n=4, shortcut=True)   # ★ P3 skip
self.b5 = Conv(192,  384,  k=3, s=2)   # P4:  80 → 40
self.b6 = C2f (384,  384,  n=4, shortcut=True)   # ★ P4 skip
self.b7 = Conv(384,  576,  k=3, s=2)   # P5:  40 → 20
self.b8 = C2f (576,  576,  n=2, shortcut=True)
self.b9 = SPPF(576,  576,  k=5)                   # ★ P5 skip
```

A few things to notice:

- The pattern is `Conv(stride=2) → C2f → Conv(stride=2) → C2f → ...`
  — alternating "downsample" and "process". This is identical in
  spirit to ResNet's `conv → stage1 → conv → stage2 → ...`.
- Three feature maps marked with ★ are **passed forward to the
  head**: `P3` (80×80, 192 channels), `P4` (40×40, 384 channels),
  `P5` (20×20, 576 channels).
- "P*n*" naming convention: `Pn` means "the feature map whose
  resolution is the input divided by 2ⁿ." So `P3` is at 1/8
  resolution, `P5` at 1/32.

### 5.3 The head — PAN-FPN

```python
self.up = nn.Upsample(scale_factor=2, mode="nearest")

# Top-down pass: deep semantic info flows to high-resolution maps.
self.h12 = C2f(1024+512, 512, n=2, shortcut=False)   # mid-P4
self.h15 = C2f(512+256,  256, n=2, shortcut=False)   # OUT-P3

# Bottom-up pass: high-resolution detail flows to deep maps.
self.h16 = Conv(256, 256, k=3, s=2)
self.h18 = C2f(256+512, 512, n=2, shortcut=False)    # OUT-P4
self.h19 = Conv(512, 512, k=3, s=2)
self.h21 = C2f(512+1024, 1024, n=2, shortcut=False)  # OUT-P5
```

Numbers like `1024+512` indicate channel counts after concatenation,
e.g. P5 (576, but written here as 1024 pre-scaling) concatenated with
P4 (384, written as 512 pre-scaling). The `_scale_channels` function
handles the actual conversion when the layers are constructed.

The information flow:

```
backbone:   P3 ────── P4 ────── P5
                                 │   upsample + concat (h12)
                                 ▼
                              mid-P4
                                 │   upsample + concat (h15)
                                 ▼
P3 ─────────────────────►   OUT-P3   (small object detector)
                                 │   stride-2 conv (h16) + concat
                                 ▼
                              OUT-P4  (medium object detector)
                                 │   stride-2 conv (h19) + concat
                                 ▼
                              OUT-P5  (large object detector)
```

### 5.4 The full forward pass

```python
def forward(self, x):
    # backbone
    x = self.b0(x); x = self.b1(x); x = self.b2(x); x = self.b3(x)
    p3 = self.b4(x)
    x  = self.b5(p3); p4 = self.b6(x)
    x  = self.b7(p4); x  = self.b8(x); p5 = self.b9(x)

    # head: top-down
    n4     = self.h12(torch.cat([self.up(p5), p4], dim=1))
    out_p3 = self.h15(torch.cat([self.up(n4), p3], dim=1))

    # head: bottom-up
    out_p4 = self.h18(torch.cat([self.h16(out_p3), n4], dim=1))
    out_p5 = self.h21(torch.cat([self.h19(out_p4), p5], dim=1))

    return self.detect([out_p3, out_p4, out_p5])
```

This is the entire model. Twelve lines of forward code. Compare this
to the original Ultralytics implementation, which builds the same
graph by parsing a yaml file at runtime — equivalent behavior, but
much harder to *read*.

---

## 6. Train mode vs eval mode

PyTorch's `model.train()` and `model.eval()` toggles behavior in two
places relevant to YOLOv8:

1. **BatchNorm and Dropout** behavior (the standard PyTorch behavior
   you already know — BN uses running statistics in eval).
2. **The Detect head's output format**, which we explicitly handle.

In **train mode**, `Detect.forward` returns three raw tensors:

```python
[(B, 144, 80, 80), (B, 144, 40, 40), (B, 144, 20, 20)]
```

The training loop's loss function unpacks the 144 channels into
`(64 box-distance bins) + (80 class logits)` and applies CIoU loss to
boxes, BCE loss to classes, and DFL loss to the box distributions.

In **eval mode**, `Detect.forward` runs the inference path:

```python
(B, 84, 8400)
```

This means you can drop this exact model into an inference pipeline
without changing any code — `eval()` does the right thing.

After this comes the post-processing pipeline (not included in this
codebase): score thresholding to drop low-confidence predictions, then
non-max suppression (NMS) to remove duplicate boxes for the same
object.

---

## 7. Suggested study order

A concrete plan if you want to walk through this codebase from
scratch:

1. **Read `modules.py` top to bottom**, in the order it is written:
   `autopad → Conv → Bottleneck → C2f → SPPF → DFL → Detect`. Each
   block builds on the previous ones.

2. **Run `verify.py`** to confirm everything imports and produces the
   expected shapes:

   ```bash
   cd yolov8m_pytorch
   python verify.py
   ```

   Expected output: `Total parameters: 25,902,640` (matches the
   reference YOLOv8m exactly).

3. **Read `model.py` carefully**, paying attention to the docstring
   topology diagram and the `forward` method. Trace the shapes
   manually for each line: start with `(1, 3, 640, 640)` and write
   down the shape after every layer.

4. **Pick one block and break it.** For example, replace `C2f` with a
   plain `Bottleneck` stack and re-run `verify.py`. The parameter
   count should drop. This kind of ablation builds real intuition.

5. **Read the original Ultralytics source** at
   `../ultralytics/ultralytics/nn/modules/{conv,block,head}.py` and
   compare to this implementation. The original supports many model
   variants and inference modes (export to ONNX, RT-DETR, world
   models, etc.), which is why it looks more complex. But the core
   `Conv`, `Bottleneck`, `C2f`, `SPPF`, `DFL`, `Detect` classes are
   the same.

---

## 8. Mental cheat sheet

| Concept | YOLOv8 implementation | ResNet equivalent |
|---------|----------------------|--------------------|
| Atomic conv unit | `Conv` (Conv2d + BN + SiLU) | Conv2d + BN + ReLU |
| Residual block | `Bottleneck` | `BasicBlock` / `Bottleneck` |
| Stage with several blocks | `C2f` (CSP-style with extra skips) | `_make_layer` |
| Multi-resolution features | three outputs (`P3`, `P4`, `P5`) | only one final output |
| Multi-scale fusion | PAN-FPN head | (none — ResNet is a classifier) |
| Multi-scale receptive field at end of backbone | `SPPF` | (none) |
| Per-pixel prediction | `Detect` (anchor-free) | `avgpool + Linear` |
| Box regression | DFL on 4 distances | (none — different task) |

---

## 9. What's next

Once this codebase makes sense, here are natural directions to study:

- **The training loss.** The original code's `loss.py`
  (`ultralytics/utils/loss.py`) implements task-aligned label
  assignment and the CIoU/BCE/DFL combination. This is the next layer
  of YOLO machinery.

- **NMS implementation.** Look up `non_max_suppression` in
  `ultralytics/utils/ops.py`. It runs after eval-mode forward to drop
  duplicate boxes.

- **The data pipeline.** `ultralytics/data/` handles image
  augmentation (mosaic, mixup, hsv jitter). This is what actually
  makes detection work in practice — the augmentation is more
  important to final mAP than most architecture choices.

- **Try training.** You can copy the weights from the official
  `yolov8m.pt` checkpoint into this from-scratch model layer-by-layer
  (the layer names map cleanly because the topology is identical),
  and run inference on real images.
