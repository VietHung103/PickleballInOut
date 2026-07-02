# Pickleball In/Out Auto-Referee — Computer Vision Pipeline

Automated system that watches a pickleball match video and calls whether the ball lands **IN** or **OUT** of the court, using three computer vision models working in sequence.

<video src="output.mp4" controls width="100%"></video>

---

## Project Goal

The goal of this project is to develop a deep learning-based system that can analyze a video
containing a controversial pickleball rally and provide support for in/out decisions. Given a rally
video as input, the system automatically detects the ball, tracks its trajectory, identifies the bounce
location, and determines whether the ball lands inside or outside the court boundary:

1. Detects the ball in every frame (YOLOv11s)
2. Tracks the ball trajectory across frames (TrackNetV5 5-frame)
3. Detects bounce events from the trajectory (3-pass algorithm)
4. Detects 12 court keypoints per frame (PickleballCourtNet)
5. Maps the bounce pixel coordinate → real-world court cm via homography
6. Compares against court boundary → **IN / OUT decision**

---

## Full Pipeline

```
Raw video (.mp4)
    │
    ▼
[Model 1] Ball Detection — YOLOv11s                          Done
    │   Latest_Yolo.ipynb — dataset prep + training
    │   Output: (x, y, confidence) per frame
    │
    ▼
[Model 2] Ball Tracking — TrackNetV5 5-frame                 Done
    │   Tracknetv5_last.ipynb — training
    │   TracknetV5_Reconstruct.ipynb — training result plots
    │   Output: (cx, cy) heatmap per frame + smoothed trajectory
    │
    ▼
[Model 3] Court Keypoint Detection — PickleballCourtNet      Done
    │   Pickleball_Court_Net.ipynb — training
    │   Output: 12 stabilized court keypoints per frame
    │
    ▼
[System] Bounce Detection + IN/OUT Decision                  Done
    │   System_Combination.ipynb — end-to-end pipeline
    │   TrackNetV5 only (no YOLO) + 3-pass bounce detector
    │   Output: bounce frame + IN/OUT call + annotated video
```

---

## Notebook 1 — Ball Detection Training (`Latest_Yolo.ipynb`)

Builds the YOLO dataset from two annotation formats and trains YOLOv11s on Google Colab.

### Dataset

Two sources of annotated data are merged:

| Source | Games | Role |
|--------|-------|------|
| Old data (CVAT XML inside `clip/frames/`) | games 1, 2, 3, 5 | train |
| Old data (CVAT XML inside `clip/frames/`) | game 4 | val |
| New data (CVAT XML alongside `frames/` folder) | games 3, 4, 6, 7, 8 | train |

Both formats use CVAT XML with point annotations (single-pixel ball center). Two separate parsers handle the different XML layouts. The converter expands each point → fixed-radius bounding box in YOLO `.txt` format.

### Training config (Cell 11)

```python
MODEL    = 'yolo11s.pt'
EPOCHS   = 150
IMGSZ    = 1280
BATCH    = 16
PATIENCE = 30   # early stopping on mAP@50
```

Augmentation: mosaic=0.5, scale=0.4, translate=0.1, hsv enabled. Vertical flip and rotation disabled to preserve court orientation.

Custom training loop (Cell 13) shows a live 6-panel dashboard per epoch and implements its own early-stopping counter on top of YOLO's built-in patience.

### Inference (Test section)

Loads `best.pt`, runs frame-by-frame with size/aspect-ratio/travel filters, renders a motion-trail overlay, and optionally tunes the confidence threshold on a short clip.

### Key cells

| Cell | Purpose |
|------|---------|
| 0 | Mount Drive |
| 1 | Config — data paths, augmentation |
| 2 | CVAT XML parsers (old + new format) |
| 3–4 | Inspect + verify clip folders |
| 5 | Build YOLO dataset (merge old + new) |
| 6 | Write `data.yaml` |
| 7–8 | Verify dataset + visualize sample annotations |
| 9–10 | GPU check + install Ultralytics |
| 11–12 | Training config + verify `data.yaml` |
| 13 | Custom training loop + live dashboard |
| 14 | Evaluate best model on val |
| 15 | Plot training curves |
| Test section | Inference on video — detection, stats, threshold tuning |

---

## Notebook 2 — TrackNetV5 Training (`Tracknetv5_last.ipynb`)

Trains TrackNetV5 5-frame from scratch on pickleball footage, using the YOLO ball detection dataset as input.

### Architecture

```
Input: B × 15 × 288 × 512   (5 frames × 3 RGB channels)

MDDLayer5
  Learnable sigmoid attention gates (alpha, beta parameters)
  Computes 4 inter-frame motion-difference maps (+ and − polarity)
  Output: B × 23 × H × W  (5×3 frames interleaved with 8 attention maps)

V2Backbone5  (UNet encoder-decoder)
  enc1: 23 → 64    ← only change vs. original 3-frame TrackNet
  enc2: 64 → 128
  enc3: 128 → 256
  enc4: 256 → 512
  dec:  skip-connected mirror of encoder
  head: 1×1 conv → sigmoid → B × 1 × 288 × 512 heatmap
```

### Dataset preparation

- Source: existing YOLO ball detection dataset (images + `.txt` labels)
- Cell 3/4: converts YOLO normalized centers → TrackNet CSV (`file_name, x_px, y_px, visibility`)
- Cell 3b: copies images from Google Drive → local Colab NVMe SSD (25–50× faster I/O)
- Cell 4: full dataset loaded into RAM (~50 GB on A100) — eliminates disk I/O per batch

### Training

| Param | Value |
|-------|-------|
| Loss | WBCELoss (weighted BCE with focal-style false-negative penalty) |
| Optimizer | AdamW |
| Scheduler | MultiStepLR |
| Early stopping | val F1 |
| Output | 288 × 512 sigmoid heatmap |

Live 6-panel training dashboard: train/val loss, F1, precision, recall, TP/FP/FN, detected trajectory preview.

### Inference (Test Video section)

- Buffer all frames into RAM once
- Sliding 5-frame window; boundary frames use clamped context `preprocessed[clamp(i+d, 0, N-1)]` so ALL N frames produce a heatmap
- Ball center extracted via contour centroid at `TN_THRESH = 0.5`
- Output video: trajectory trail + HUD showing detection rate

### Key cells

| Cell | Purpose |
|------|---------|
| 1 | Config — paths, output directory |
| 2 | GPU check |
| 3/4 | YOLO labels → TrackNet CSV |
| 3b | Copy images Drive → SSD |
| 4 | RAM-cached Dataset + DataLoader |
| 5 | TrackNetV5 5-frame model definition |
| 6 | WBCELoss + evaluation metrics |
| 7 | Training loop + live dashboard |
| 8 | Final val evaluation |
| Test section | Inference + output video + threshold tuner |

---

## Notebook 3 — TrackNetV5 Result Visualization (`TracknetV5_Reconstruct.ipynb`)

Loads the saved training checkpoint and reconstructs all training plots without re-running training. Useful for reporting and slide preparation.

### What it produces

- **Cell 3** — Full 6-panel dashboard (same layout as the live training view)
- **Cell 4** — Clean loss curve: train loss, val loss, LR decay markers
- **Cell 5** — F1 + Precision + Recall curves with best-epoch annotation
- **Cell 6** — Validation accuracy curve
- **Cell 7** — 2-panel summary (loss + F1) — optimized for slides
- **Cell 8** — Full epoch-by-epoch table (train loss, val loss, acc, prec, recall, F1) for reports

### Key cells

| Cell | Purpose |
|------|---------|
| 1 | Config — checkpoint path, output directory, LR/decay params for axis labels |
| 2 | Load history dict from `last_tracknetv5_5frame.pth` |
| 3–7 | Training plots (dashboard, loss, F1/prec/rec, accuracy, summary) |
| 8 | Epoch-by-epoch metric table |

---

## Notebook 4 — Court Keypoint Training (`Pickleball_Court_Net.ipynb`)

Trains a custom heatmap-based model (`PickleballCourtNet`) to detect 12 pickleball court keypoints simultaneously.

### Court keypoint schema — 12 keypoints

```
KP0 ─────── KP1 ─────── KP2      ← Far  baseline   (L_BL_BG, M_BL_BG, R_BL_BG)
 │           │           │
KP5 ─────── KP4 ─────── KP3      ← Far  kitchen     (L_KL_BG, M_KL_BG, R_KL_BG)
 │                       │
 │           NET          │
 │                       │
KP6 ─────── KP7 ─────── KP8      ← Near kitchen     (L_KL_FG, M_KL_FG, R_KL_FG)
 │           │           │
KP11─────── KP10─────── KP9      ← Near baseline    (L_BL_FG, M_BL_FG, R_BL_FG)
```

**Real-world coordinates (cm) — origin at KP11 (near-left corner):**

| KP | Name | X_cm | Y_cm |
|----|------|------|------|
| 0 | L_BL_BG | 0 | 1372 |
| 1 | M_BL_BG | 305 | 1372 |
| 2 | R_BL_BG | 610 | 1372 |
| 3 | R_KL_BG | 610 | 915 |
| 4 | M_KL_BG | 305 | 915 |
| 5 | L_KL_BG | 0 | 915 |
| 6 | L_KL_FG | 0 | 457 |
| 7 | M_KL_FG | 305 | 457 |
| 8 | R_KL_FG | 610 | 457 |
| 9 | R_BL_FG | 610 | 0 |
| 10 | M_BL_FG | 305 | 0 |
| 11 | L_BL_FG | 0 | 0 |

Court: 610 cm wide × 1372 cm long. Kitchen line at 457 cm from near baseline. Net at 686 cm.

### Model architecture (`PickleballCourtNet`)

- Input: `1 × 3 × 360 × 640` (RGB frame resized to 640×360)
- UNet-style ConvBlocks: `Conv2d → ReLU → BN`, encoder-decoder with skip connections
- Output: `1 × 13 × 360 × 640` — 12 keypoint heatmaps + 1 net/background channel
- Keypoint location extracted per channel via argmax on Gaussian heatmap

### Dataset preparation

- **Dataset A** — your own CVAT/Roboflow labels → **train split**
- **Dataset B** — original TennisCourtDetector dataset → **val + test split**
- Labels converted from YOLO format → per-channel Gaussian heatmaps at 640×360
- Gaussian radius per keypoint is wider for far-court KPs (higher prediction error)

### Training

| Param | Value |
|-------|-------|
| Epochs | 100 |
| Batch | 8 |
| LR | 1e-5 |
| Loss | MSELoss on heatmaps |
| Early stopping | 5 epochs on val loss |

### Post-processing pipeline

**Stage 1 — Hough line refinement** (far-court KPs):
1. Crop 120 px window around predicted KP
2. HSV masking: isolate white court lines (S=0–50, V=170–255), remove court surface
3. Morphological cleanup + optional Zhang-Suen thinning (`opencv-contrib-python`)
4. HoughLines → separate horizontal/vertical groups → solve 2-line intersection
5. Reject if shift > 80% of crop window or white-pixel check fails

**Stage 2 — KPStabilizer** (all keypoints):
```
8-frame sliding median buffer
EMA alpha = 0.05  (very smooth)
Lock:  if |new − smoothed| < 6 px → keep previous smoothed position
```

**Stage 3 — Homography correction**: RANSAC homography from 12 KPs → real-world grid snaps remaining outlier KPs to geometric court structure.

### Key cells

| Cell | Purpose |
|------|---------|
| 1 | Install, mount, clone TennisCourtDetector |
| 2 | Download Dataset A + B from Roboflow |
| 3 | Merge: Train=A, Val/Test=B |
| 4 | YOLO → JSON + resize to 640×360 |
| 5 | PickleballCourtNet model definition |
| 6 | Dataset + Gaussian heatmap generation |
| 7 | Visual dataset verification |
| 8 | Training loop |
| 9 | Loss + accuracy curves |
| 10 | Hough + homography post-processing functions |
| 11 | Single-frame inference test |
| 12/13 | Full video inference |
| End-to-end section | Stabilized video with KPStabilizer |

---

## Notebook 5 — Full System (`System_Combination.ipynb`)

Combines TrackNetV5 + PickleballCourtNet into one end-to-end pipeline with bounce detection and IN/OUT decision. **Uses TrackNetV5 as the sole ball detector — no YOLOv11s involved.**

### Pipeline phases

| Phase | Cell | Description |
|-------|------|-------------|
| 1 | Cell 6 | Buffer all video frames into RAM |
| 2 | Cell 7 | TrackNetV5 inference → raw ball (cx, cy) per frame |
| 3 | Cell 7 | Trajectory smoothing (median filter + gap fill) |
| 4 | Cell 8 | PickleballCourtNet → 12 KPs + KPStabilizer + homography per frame |
| 5 | Cell 9 | **3-pass bounce detection + IN/OUT decision** |
| 6 | Cell 11 | Render annotated output video |
| — | Cell 12 | Save bounce log as JSON |

### Configuration (Cell 2)

```python
TRACKNET_PATH          = '...best_tracknetv5_5frame.pth'
COURTNET_PATH          = '...TennisCourtNet_v2/best.pt'
SMOOTH_WINDOW          = 3       # median filter window for trajectory
SMOOTH_MAX_GAP         = 3       # max missing frames to interpolate
BOUNCE_COOLDOWN_FRAMES = 8       # min frames between two bounce events
BOUNCE_VEL_MIN_PX      = 0.8    # min vy (px/frame) to count as motion
BOUNCE_APPROACH_MIN_PX = 3.0    # min approach vy — guards against player-hit FPs
POLYGON_MARGIN_PX      = 5      # pixel tolerance on court polygon
COURT_CM_MARGIN        = 5.0    # cm tolerance for homography fallback
PARAB_RESIDUE_THRESH   = 100.0  # MSE spike threshold for parabolic pass
GAP_BOUNCE_MAX_FRAMES  = 8      # max gap length to trigger GAP pass
COURT_X_MIN = 0;  COURT_X_MAX = 610   # court bounds in cm
COURT_Y_MIN = 0;  COURT_Y_MAX = 1372
```

### Bounce detection — 3-pass algorithm

Camera: elevated side-view. Ball pixel Y increases as ball falls toward the near court (camera side). A near-end bounce produces a **Y local maximum**: vy goes positive → negative.

**Pass 1 — VEL (velocity sign reversal)**
```
avg_before = mean(vy[i-2 : i])     # approach speed
avg_after  = mean(vy[i+1 : i+3])   # departure speed

primary_near:  avg_before ≥ 3.0  AND  avg_after ≤ −0.8
fallback_near: avg_before > 0    AND  avg_after < 0    AND  delta_vy ≥ 6.0
decel_near:    avg_before ≥ 6.4  AND  0 ≤ avg_after < 0.8  AND  delta_vy ≥ 6.4
```

`BOUNCE_APPROACH_MIN_PX = 3.0` prevents **player-hit false positives**: real bounces
arrive under gravity (avg_before ≥ 4.5 px/frame); player hits show the ball floating
at arc peak (avg_before ≈ 2.0 px/frame). `decel_near` catches fast bounces that don't
fully reverse direction.

**Pass 2 — GAP (ball disappears at ground contact)**
Detects gaps ≤ 8 missing frames where pre-gap vy is positive and post-gap vy is
negative — consistent with ball briefly hidden behind the court surface at bounce.

**Pass 3 — PARAB (parabolic residue spike)**
Fits a parabola to the last 5 detected positions before each candidate. At a real bounce
the curvature abruptly changes → MSE spike > 100. Avoids firing on smooth arc segments.

**Merge priority:** `VEL > GAP > PARAB` — same-frame detections from multiple passes
keep the highest-priority result only.

### IN/OUT decision — two-tier approach

**Tier 1 — pixel polygon test (primary):**
```python
polygon = build_court_polygon(court_kps)   # convex hull of 12 KPs
decision, dist_px = pixel_polygon_inout(x_bounce, y_bounce, polygon)
# dist_px > +5  → IN   (clearly inside)
# dist_px < −5  → OUT  (clearly outside)
# |dist_px| ≤ 5 → UNCERTAIN
```

**Tier 2 — homography fallback (UNCERTAIN only):**
```python
x_cm, y_cm = pixel_to_court_cm(x_bounce, y_bounce, H_px2court)
if COURT_X_MIN − 5 ≤ x_cm ≤ COURT_X_MAX + 5 and
   COURT_Y_MIN − 5 ≤ y_cm ≤ COURT_Y_MAX + 5:
    decision = 'IN'
else:
    decision = 'OUT'
```

### Example results (7-bounce test video)

| Frame | Decision | dist_px | Method |
|-------|----------|---------|--------|
| 32 | OUT | −20.3 | VEL |
| 67 | OUT | −23.4 | VEL |
| 98 | OUT | −20.6 | VEL |
| 128 | OUT | −22.0 | VEL |
| 275 | IN | +38.5 | VEL |
| 332 | IN | +43.7 | VEL |
| 400 | OUT | −8.8 | VEL + cm fallback |

### Key cells

| Cell | Purpose |
|------|---------|
| 0 | Architecture overview |
| 2 | CONFIG — all tunable parameters |
| 3 | Model definitions (TrackNetV5 + PickleballCourtNet) |
| 4 | Helper functions: bounce detector, polygon IN/OUT, homography, KPStabilizer |
| 5 | Load model weights |
| 6 | Buffer video frames |
| 7 | TrackNetV5 inference + trajectory smoothing |
| 8 | Court KP detection + KPStabilizer + homography per frame |
| 9 | Bounce detection + IN/OUT decision |
| 10 | Diagnostic plots (trajectory, velocity, bounce markers) |
| 11 | Render annotated output video |
| 12 | Save bounce JSON log |

---

## Data Layout

```
PickleballInOut/
├── Latest_Yolo.ipynb                    YOLOv11s dataset prep + training
├── Tracknetv5_last.ipynb                TrackNetV5 5-frame training
├── TracknetV5_Reconstruct.ipynb         TrackNetV5 training result plots
├── Pickleball_Court_Net.ipynb           PickleballCourtNet training
├── System_Combination.ipynb             End-to-end system (TrackNet + CourtNet)
└── extract_frame.py                     Frame extraction utility

pickleball/
├── vid/        Raw game footage (game1–game8)        [gitignored]
├── data/       Extracted frames + CVAT annotations   [gitignored]
├── model/      Model weights                         [gitignored]
│   ├── best.pt                     YOLOv11s ball detector (19 MB)
│   ├── best_2.pt                   Alternative ball model (76 MB)
│   └── TennisCourtNet_v2/best.pt   PickleballCourtNet (12 KPs)
└── output/     Inference output videos               [gitignored]
```

---

## Setup

```bash
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib pillow pyyaml scipy einops

# Enables Zhang-Suen thinning in court KP Hough refinement
pip install opencv-contrib-python
```

All training notebooks run on **Google Colab with A100 GPU** and read data from Google Drive.

---

## Roadmap

| Component | Status |
|-----------|--------|
| Ball detection dataset (old + new data merged) | Done |
| YOLOv11s ball detector training | Done |
| TrackNetV5 5-frame training | Done |
| TrackNetV5 training result visualization | Done |
| PickleballCourtNet (12 KPs) training | Done |
| Court KP post-processing (Hough + KPStabilizer) | Done |
| Homography pixel → real-world cm | Done |
| 3-pass bounce detector (VEL + GAP + PARAB) | Done |
| IN/OUT decision (polygon + cm fallback) | Done |

