# SeaDronesSee ODV2 — Experiment Results

Comparison of four YOLOv8m architecture variants on SeaDronesSee Object
Detection v2. All numbers are **val-set metrics at the best epoch**,
obtained by re-running `model.val()` on the saved `best.pt`
(`scripts/summarize_runs.py`).

---

## Setup

| Item | Value |
|---|---|
| Base model            | YOLOv8m (25.9M params) |
| Pretrained            | COCO `yolov8m.pt` (transfer learning; shape-incompatible layers re-init) |
| Dataset               | SeaDronesSee Object Detection v2 |
| Train / Val images    | 8,930 / 1,547 |
| Classes (5)           | swimmer, boat, jetski, life_saving_appliances, buoy |
| Evaluation split      | val (test labels are not publicly released — kept for the official leaderboard — so the test set is not used here) |
| Image size            | 640 |
| Batch                 | 16 |
| Optimizer             | SGD (lr0=0.01, momentum=0.937, weight_decay=5e-4) |
| Schedule              | 3-epoch warmup → cosine; 100 epochs, `patience=30` |
| Augmentation          | Mosaic + HSV + scale/translate + fliplr (no flipud) |
| Precision             | AMP (mixed precision) |
| Hardware              | NVIDIA A100 80GB PCIe MIG 3g.40gb (Elice) |

Every variant trains under **identical** hyperparameters; only the
architecture cfg differs.

---

## Variants

| ID | Variant | Cfg | Change vs baseline |
|---|---|---|---|
| **M0** | baseline    | `cfg/yolov8m.yaml`             | — (unmodified YOLOv8m) |
| **M1** | sppf-k3     | `cfg/yolov8m-sppf-k3.yaml`     | SPPF kernel 5 → 3 (smaller receptive field) |
| **M2** | p2          | `cfg/yolov8m-p2.yaml`          | + P2 detection head (stride 4) |
| M3     | p2-sppf-k3  | `cfg/yolov8m-p2-sppf-k3.yaml`  | P2 + SPPF k=3 (pending) |

---

## Overall Results (val)

| Model     | mAP@50    | mAP@50-95 | Precision | Recall    |
|-----------|-----------|-----------|-----------|-----------|
| baseline  | 0.732     | 0.439     | 0.889     | 0.708     |
| sppf-k3   | 0.731     | 0.439     | 0.885     | 0.719     |
| **p2**    | **0.785** | **0.468** | **0.897** | **0.757** |

### Pairwise deltas (overall)

| Pair                    | Δ mAP@50  | Δ mAP@50-95 | Δ Precision | Δ Recall |
|-------------------------|-----------|-------------|-------------|----------|
| sppf-k3 − baseline      | −0.001    | ±0.000      | −0.004      | +0.011   |
| p2 − baseline           | **+0.053** | **+0.029**  | +0.008      | **+0.049** |
| p2 − sppf-k3            | **+0.054** | **+0.029**  | +0.012      | **+0.038** |

SPPF k=3 alone barely moves the aggregate numbers (every delta ≤ 0.011).
Adding the P2 head delivers the only large effect: +0.053 mAP@50,
+0.049 Recall — a notable safety gain in a SAR setting where each unit
of Recall corresponds to fewer missed targets.

---

## Per-Class AP@50

| Model     | boat      | buoy      | jetski    | life_saving_appliances | swimmer   |
|-----------|-----------|-----------|-----------|------------------------|-----------|
| baseline  | **0.965** | 0.639     | **0.911** | 0.378                  | 0.766     |
| sppf-k3   | 0.963     | 0.658     | 0.924     | 0.356                  | 0.753     |
| **p2**    | 0.960     | **0.803** | 0.869     | **0.490**              | **0.802** |

### Pairwise deltas (AP@50)

| Pair                    | boat   | buoy       | jetski | life_saving_appliances | swimmer |
|-------------------------|--------|------------|--------|------------------------|---------|
| sppf-k3 − baseline      | −0.002 | +0.019     | +0.013 | −0.022                 | −0.013  |
| p2 − baseline           | −0.005 | **+0.164** | −0.042 | **+0.112**             | **+0.036** |
| p2 − sppf-k3            | −0.003 | **+0.145** | −0.055 | **+0.134**             | **+0.049** |

Per-pair reading:

- **sppf-k3 vs baseline**: small, mixed-sign movements (≤ 0.022 absolute).
  Marginal gains on buoy/jetski; regressions on life_saving_appliances
  and swimmer. No clear pattern.
- **p2 vs baseline**: large, consistent gains on every small-object class
  (swimmer, buoy, life_saving_appliances), with the trade-off being a
  4.2-point drop on jetski. boat unchanged.
- **p2 vs sppf-k3**: same shape as p2 vs baseline, slightly larger gaps
  on the small classes — p2 is doing the work, not sppf-k3.

---

## Per-Class AP@50-95 (stricter, COCO-style)

| Model     | boat      | buoy      | jetski    | life_saving_appliances | swimmer   |
|-----------|-----------|-----------|-----------|------------------------|-----------|
| baseline  | **0.724** | 0.390     | **0.596** | 0.181                  | 0.307     |
| sppf-k3   | 0.721     | 0.388     | 0.632     | 0.156                  | 0.300     |
| **p2**    | 0.722     | **0.474** | 0.580     | **0.231**              | **0.332** |

### Pairwise deltas (AP@50-95)

| Pair                    | boat   | buoy       | jetski | life_saving_appliances | swimmer    |
|-------------------------|--------|------------|--------|------------------------|------------|
| sppf-k3 − baseline      | −0.003 | −0.002     | +0.036 | −0.025                 | −0.007     |
| p2 − baseline           | −0.002 | **+0.084** | −0.016 | **+0.050**             | **+0.025** |
| p2 − sppf-k3            | +0.001 | **+0.086** | −0.052 | **+0.075**             | **+0.032** |

Per-pair reading:

- **sppf-k3 vs baseline**: mostly noise (≤ 0.025 absolute). Only jetski
  moves meaningfully (+0.036), but at the cost of the smallest classes.
- **p2 vs baseline**: localization itself improves on every small class
  — buoy +21.5% relative, life_saving_appliances +27.6% relative,
  swimmer +8.1% relative — confirming that the new P2/4 feature map
  helps the network not only *find* small objects but also *place the
  boxes* on them. jetski regresses slightly.
- **p2 vs sppf-k3**: nearly identical sign pattern, larger magnitudes —
  again, p2 is the source of the SOD improvement.

---

## Analysis

The three variants form three pairwise comparisons. Each pair isolates a
single architectural intervention:

| Pair                | Architectural change isolated                              |
|---------------------|------------------------------------------------------------|
| sppf-k3 vs baseline | receptive field of SPPF (k=5 → k=3) only                   |
| p2 vs baseline      | extra detection head at stride 4 (P2/4 feature map) only   |
| p2 vs sppf-k3       | gives an "is the gain from P2 or from RF?" cross-check     |

### Pair 1 — sppf-k3 vs baseline (RF intervention)

- **Overall**: Δ mAP@50 −0.001, Δ mAP@50-95 ±0.000, Δ Recall +0.011.
  The deep-backbone receptive field is not, by itself, a strong lever
  for this dataset.
- **Per-class**: small movements (≤ 0.022 absolute) with **mixed signs**.
  buoy +0.019, jetski +0.013, but life_saving_appliances −0.022 and
  swimmer −0.013 at AP@50. No coherent story emerges.
- **Interpretation**: shrinking the SPPF kernel narrows the
  global-context window. Objects whose recognition depends on the
  surrounding scene (people, rescue gear) lose information; objects
  that look like themselves (buoy) tolerate it. Net effect on
  aggregate metrics: noise.

### Pair 2 — p2 vs baseline (resolution intervention)

- **Overall**: Δ mAP@50 **+0.053**, Δ mAP@50-95 **+0.029**,
  Δ Recall **+0.049**. Precision is also slightly up (+0.008), so the
  Recall gain does not come from extra false positives.
- **Per-class AP@50**: **all three small-object classes improve
  substantially** (buoy +0.164, life_saving_appliances +0.112,
  swimmer +0.036). Large-object class (boat) is unchanged
  (−0.005, already at 0.965 ceiling). One trade-off: jetski −0.042.
- **Per-class AP@50-95**: same direction with double-digit relative
  gains on small classes (buoy +21.5%, life_saving_appliances +27.6%,
  swimmer +8.1%). Localization quality itself improves, not just the
  detected count.
- **Interpretation**: adding a stride-4 detection branch gives small
  objects a feature map at the resolution they need to be represented
  in. The improvement is exactly where SOD theory predicts it.

### Pair 3 — p2 vs sppf-k3 (cross-check)

- **Overall**: Δ mAP@50 **+0.054**, Δ mAP@50-95 **+0.029**. Almost
  identical to p2 vs baseline.
- **Per-class**: same sign pattern as p2 vs baseline, slightly larger
  magnitudes on the small classes (e.g. buoy +0.145 vs +0.164 at AP@50).
- **Interpretation**: the improvement from p2 is **not** explained by
  the RF change — sppf-k3 itself does not improve over baseline, and
  p2 improves over sppf-k3 by the same amount as it improves over
  baseline. The P2 head is doing the work.

### The jetski regression

jetski is the one class where p2 *underperforms* baseline (and sppf-k3).

| Model     | jetski AP@50 | jetski AP@50-95 |
|-----------|--------------|-----------------|
| baseline  | 0.911        | 0.596           |
| sppf-k3   | **0.924**    | **0.632**       |
| p2        | 0.869        | 0.580           |

Jetski is medium-sized (~320 val instances). Plausible causes:

- The new P2 branch adds ~4× anchors (8,400 → 34,000) and absorbs
  gradient signal previously concentrated on P3/P4 — the levels that
  were responsible for medium objects.
- jetski is the variant most sensitive to this redistribution, since
  it sits at the boundary between "small enough to benefit from P2"
  and "medium enough to be served by P3/P4."

Class- or size-aware branch routing is a candidate fix.

### Recall in the SAR context

Recall is the most operationally meaningful metric for maritime
search-and-rescue: each missed detection is a target the system fails
to surface.

| Model     | Overall Recall |
|-----------|----------------|
| baseline  | 0.708          |
| sppf-k3   | 0.719          |
| **p2**    | **0.757**      |

The +0.049 Recall gain of p2 over baseline translates to roughly five
additional true detections per hundred ground-truth instances — a
direct safety relevance, not merely a benchmark number.

### Caveats

- `life_saving_appliances` has only ~330 val instances, so per-class
  AP variance is higher. The sppf-k3 ↔ baseline fluctuations are
  ≤ 0.025; the p2 improvements (+0.112 AP@50, +0.050 AP@50-95) are
  several times larger and therefore unlikely to be noise.
- Numbers are single-seed runs. Multi-seed averaging is a follow-up
  item.

---

## Takeaways

1. **P2 head addresses the SOD bottleneck**: feature-map resolution
   matters more than receptive-field size for small-object detection on
   this dataset.
2. **SPPF k=3 is not a sufficient intervention** on its own.
3. **P2 does not hurt the large-object class** — a clean Pareto
   improvement on small objects.
4. **Open directions**: P2 + SPPF k=3 (M3, pending), size-aware branch
   design to recover the jetski regression, multi-seed evaluation.

---

## Status

- [x] M0 baseline — trained + evaluated
- [x] M1 sppf-k3 — trained + evaluated
- [x] **M2 p2 — trained + evaluated** (core contribution validated)
- [ ] M3 p2-sppf-k3 — training pending

---

## Reproduction

```bash
# train (all variants use identical hyperparameters)
python experiments/train.py --model baseline
python experiments/train.py --model sppf-k3
python experiments/train.py --model p2
python experiments/train.py --model p2-sppf-k3

# evaluate
python experiments/scripts/summarize_runs.py \
    --models baseline sppf-k3 p2 p2-sppf-k3
```

Per-variant outputs under `experiments/runs/<model>/`:

- `weights/best.pt` — checkpoint at best val mAP@0.5:0.95
- `results.csv` — per-epoch metrics
- `results.png` — training curves
- `confusion_matrix.png` — class confusion on val
- `experiments/summary.csv` — aggregated comparison
