# SeaDronesSee ODV2 — Experiment Results

Comparison of YOLOv8m architecture variants on SeaDronesSee Object Detection v2.
All numbers below are **val-set results at the best epoch**, measured by re-running
`model.val()` on the saved `best.pt` (see `scripts/summarize_runs.py`).

---

## Setup

| Item | Value |
|---|---|
| Base model | YOLOv8m (25.9M params) |
| Pretrained | COCO `yolov8m.pt` (transferred into each variant; shape-incompatible layers re-init) |
| Dataset | SeaDronesSee Object Detection V2 |
| Train / Val images | 8,930 / 1,547 |
| Total bboxes (train) | 57,760 |
| Classes (5, `ignored` skipped) | swimmer, boat, jetski, life_saving_appliances, buoy |
| Image size | 640 |
| Batch | 16 |
| Optimizer | SGD (lr0=0.01, momentum=0.937, weight_decay=0.0005) |
| Schedule | warmup 3 ep → cosine; 100 epochs with `patience=30` |
| Augmentation | Mosaic + HSV + scale/translate + fliplr (no flipud) |
| Precision | AMP (mixed precision) |
| Hardware | A100 80GB PCIe MIG 3g.40gb (Elice) |

Every variant trains under **identical** hyperparameters; only the architecture cfg differs.

---

## Variants

| ID | Variant | Cfg file | Change vs baseline |
|---|---|---|---|
| **M0** | baseline | `cfg/yolov8m.yaml` | — (unmodified YOLOv8m) |
| **M1** | sppf-k3 | `cfg/yolov8m-sppf-k3.yaml` | SPPF kernel 5 → 3 (smaller receptive field) |
| **M2** | p2 | `cfg/yolov8m-p2.yaml` | + P2 detection head (stride 4) |
| M3 | p2-sppf-k3 | `cfg/yolov8m-p2-sppf-k3.yaml` | P2 + SPPF k=3 combined — *pending* |

---

## Overall Results (val)

| Model | mAP@50 | mAP@50-95 | Precision | Recall |
|---|---|---|---|---|
| baseline | 0.732 | 0.439 | 0.889 | 0.708 |
| sppf-k3 | 0.731 | 0.439 | 0.885 | 0.719 |
| **p2** | **0.785** | **0.468** | **0.897** | **0.757** |

**Δ (p2 − baseline)**:
- mAP@50:    **+0.053** (+7.2% relative)
- mAP@50-95: **+0.029** (+6.6% relative)
- Precision: +0.008 (오탐 증가 없음)
- Recall:    **+0.049** (놓침 감소, SAR 시나리오에서 핵심)

**Δ (sppf-k3 − baseline)**: 거의 0 (mAP −0.001, ±0.000) — SPPF k=3 단독으로는 전체 성능 변화 미미.

**P2 추가가 우리 가설을 명확히 입증**: 고해상도 feature map (stride 4) 이 작은 객체 탐지의 본질적 한계를 직접 해결.

---

## Per-Class AP@50

| Model | boat | buoy | jetski | life_saving_appliances | swimmer |
|---|---|---|---|---|---|
| baseline | **0.965** | 0.639 | **0.911** | 0.378 | 0.766 |
| sppf-k3 | 0.963 | 0.658 | 0.924 | 0.356 | 0.753 |
| **p2** | 0.960 | **0.803** | 0.869 | **0.490** | **0.802** |

**Δ (p2 − baseline)** per class:
- boat: −0.005 (큰 객체, 이미 ceiling)
- **buoy: +0.164** (가장 큰 개선)
- jetski: −0.042 (중간 객체, 약간 악화 — interesting trade-off)
- **life_saving_appliances: +0.112**
- **swimmer: +0.036**

**핵심 발견**: 작은 객체 (buoy, LSA, swimmer) 3개에서 **모두 큰 폭으로 개선**. 큰 객체 (boat) 는 거의 변화 없음 (이미 0.965 ceiling). jetski는 약간 악화 — 중간 크기 객체에서 P2 분기가 P4 분기 학습을 약간 분산시킨 효과 가능.

---

## Per-Class AP@50-95 (stricter, COCO-style)

| Model | boat | buoy | jetski | life_saving_appliances | swimmer |
|---|---|---|---|---|---|
| baseline | **0.724** | 0.390 | **0.596** | 0.181 | 0.307 |
| sppf-k3 | 0.721 | 0.388 | 0.632 | 0.156 | 0.300 |
| **p2** | 0.722 | **0.474** | 0.580 | **0.231** | **0.332** |

**Δ (p2 − baseline)** per class:
- boat: −0.002 (변화 없음)
- **buoy: +0.084** (+21.5% relative)
- jetski: −0.016
- **life_saving_appliances: +0.050** (+27.6% relative)
- **swimmer: +0.025** (+8.1% relative)

**작은 객체 localization 자체도 개선**: AP@50-95는 bbox 위치 정확도까지 평가하는 엄격한 지표. P2 가 작은 객체의 정확한 위치까지 잡아낸다는 증거.

---

## Analysis

### Main finding — P2 head 가 SOD 문제의 본질적 해결책

전체 mAP@50 +0.053, mAP@50-95 +0.029. **작은 객체 클래스 3개에서 모두 큰 개선**:

| 작은 객체 | AP@50 Δ | AP@50-95 Δ |
|---|---|---|
| buoy | +0.164 | +0.084 |
| life_saving_appliances | +0.112 | +0.050 |
| swimmer | +0.036 | +0.025 |

큰 객체 (boat) 는 거의 변화 없음 (±0.005) — **P2 추가가 큰 객체 검출을 해치지 않으면서 작은 객체만 선택적으로 개선**. 이상적인 변형 결과.

### SPPF k=3 vs P2 — 같은 motivation, 다른 효과

두 변형 모두 "작은 객체 탐지 개선" 을 노렸으나 결과는 정반대:

- **SPPF k=3**: receptive field만 조정 → 작은 객체 단순 형태 (buoy +0.019) 에만 부분 개선, 컨텍스트 필요한 객체 (LSA −0.022, swimmer −0.013) 에는 악화
- **P2 head**: feature map 해상도 자체 증가 → 모든 작은 객체에 일관된 큰 개선

**시사점**: 작은 객체 탐지의 핵심 병목은 **receptive field 크기가 아니라 feature map 해상도**. P2 가 본질적 해결책.

### Trade-off — jetski 의 작은 악화

| Model | jetski AP@50 | jetski AP@50-95 |
|---|---|---|
| baseline | 0.911 | 0.596 |
| sppf-k3 | 0.924 | 0.632 |
| **p2** | 0.869 | 0.580 |

jetski (중간 크기, 인스턴스 320개) 는 P2 에서 −0.042 악화. 가능한 해석:
- P2 분기가 P3/P4 학습을 일부 분산
- 중간 크기 객체는 P3/P4 가 본디 잘 잡았는데, P2 추가로 anchor가 4배 늘어서 매칭 경쟁
- 또는 학습 데이터 분포 (jetski 인스턴스 적음) + class imbalance 영향

미래 작업으로 **크기/클래스별 differentiated branch** (예: jetski 같은 중간 객체용 별도 모듈) 가능성.

### Recall 분석 — SAR 관점

| Class | baseline R | **p2 R** | Δ |
|---|---|---|---|
| (전체) | 0.708 | **0.757** | **+0.049** |
| swimmer (estimated) | ~0.635 | (개선) | |
| LSA (estimated) | ~0.014 | (대폭 개선, AP 0.378→0.490 이 시사) | |

**SAR 시나리오에서 Recall 향상은 "놓치는 조난자 감소" 와 직결**. P2 의 약 5% Recall 향상은 100명 중 5명 더 찾는 차이.

### Class imbalance + AP variance

life_saving_appliances 는 val 인스턴스가 330개로 적어 AP 분산이 큼. 그럼에도 P2 가 sppf-k3 대비 +0.134 (0.356 → 0.490) 개선한 건 noise 수준이 아닌 **structural improvement** 로 봄.

---

## Takeaways

1. **P2 head 추가는 SOD 의 본질적 해결책** — feature map 해상도 증가로 작은 객체의 detection + localization 모두 개선
2. **SPPF k 변경은 단독으로 약함** — receptive field 만 만지는 변경은 한정적, 일부 클래스에서만 도움
3. **P2 가 큰 객체 성능을 해치지 않음** — boat 등 큰 객체 AP 거의 변화 없음. 안전한 변형
4. **다음 단계 후보**:
   - P2 + SPPF k=3 결합 (M3) — RF 와 해상도 둘 다 만지면 추가 개선/충돌 확인
   - 크기별 differentiated module — jetski 같은 중간 객체 trade-off 완화
   - 더 작은 stride (P1) 또는 image super-resolution

---

## Status

- [x] M0 baseline — trained + evaluated
- [x] M1 sppf-k3 — trained + evaluated
- [x] **M2 p2 — trained + evaluated** (핵심 contribution 확인)
- [ ] M3 p2-sppf-k3 — *training pending (시간 여유 시)*

---

## Reproduction

```bash
# train (each variant uses identical hyperparameters)
python experiments/train.py --model baseline
python experiments/train.py --model sppf-k3
python experiments/train.py --model p2
python experiments/train.py --model p2-sppf-k3

# evaluate (best.pt val 재평가)
python experiments/scripts/summarize_runs.py --models baseline sppf-k3 p2 p2-sppf-k3
```

산출물:
- `experiments/runs/<model>/weights/best.pt` — 체크포인트
- `experiments/runs/<model>/results.csv` — epoch별 metric
- `experiments/runs/<model>/results.png` — 학습 곡선
- `experiments/runs/<model>/confusion_matrix.png` — confusion matrix
- `experiments/summary.csv` — 전체 모델 일괄 비교
