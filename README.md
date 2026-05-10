# Pickleball In/Out Auto-Referee — Computer Vision Pipeline

Automated system that watches a pickleball match video and calls whether the ball lands **in** or **out** of the court, using three computer vision models working in sequence.

---

## Project Goal

Replace manual line calls with an AI referee. Given a match video, the system:
1. Detects the ball in every frame
2. Tracks the ball trajectory and finds the **bounce point**
3. Detects the **court boundaries** using keypoint detection
4. Maps the bounce point from pixel coordinates to real-world court coordinates via **homography**
5. Compares against court boundary → **IN / OUT decision**

---

## Full Pipeline

```
Raw video (.mp4)
    │
    ▼
[Model 1] Ball Detection — YOLOv11s                           ✅ Done
    │   Detects ball bounding box per frame
    │   Output: (x, y, confidence) per frame
    │
    ▼
[Model 2] Ball Tracking — TrackNetV5                          🚧 In progress
    │   Tracks ball trajectory across frames
    │   Finds bounce point (where vertical velocity reverses)
    │   Output: trajectory curve + bounce pixel coordinate
    │
    ▼
[Model 3] Court Keypoint Detection — YOLOv8-Pose              🚧 In progress
    │   Detects 12 court keypoints per frame
    │   Post-processing: Hough refinement + temporal smoothing
    │   Output: 12 pixel-precise court keypoints
    │
    ▼
Homography H = cv2.findHomography(pixel_pts, real_world_pts)  🔜 Planned
    │
    ▼
Bounce point → H → real-world (x_cm, y_cm)                   🔜 Planned
    │
    ▼
IN / OUT decision (compare against court boundary ± tolerance) 🔜 Planned
```

---

## Model 1 — Ball Detection (YOLOv11s)

**Status: Done**

### What was done
- Extracted frames from raw match footage using `pickleball/data/extract_frame.py`
- Annotated ball positions in **CVAT** as point labels (single pixel at ball center), covering game_1 through game_5
- Converted CVAT XML → YOLO `.txt` format using `pickleball/code/cvat_to_yolov11_colab.ipynb`:
  - Point annotations expanded to fixed-radius bounding boxes
  - 80/20 train/val split
  - Generated `data.yaml` for training
- Trained **YOLOv11s** on Google Colab (A100 GPU):
  - 150 epochs, batch=16, imgsz=1280
  - Optimizer: AdamW, lr=0.001
  - Early stopping patience=30
  - Augmentation: mosaic enabled, vertical flip and rotation disabled (preserve court orientation)
- Saved best weights to `pickleball/model/best.pt` (19 MB)

### Inference
```bash
python pickleball/code/detect_ball.py --video path/to/video.mp4 --conf 0.25
```
Output: annotated video with bounding boxes, confidence scores, and motion trail overlay.

### Key files
| File | Purpose |
|------|---------|
| `pickleball/code/detect_ball.py` | Inference script — frame-by-frame detection with trail overlay |
| `pickleball/code/yolo11s_ball_training.ipynb` | Training notebook (Colab A100) |
| `pickleball/code/cvat_to_yolov11_colab.ipynb` | CVAT XML → YOLO format converter |
| `pickleball/data/extract_frame.py` | Frame extraction from raw video clips |
| `pickleball/model/best.pt` | Primary model weights (YOLOv11s, 19 MB) |
| `pickleball/model/best_2.pt` | Alternative model weights (76 MB) |

---

## Model 2 — Ball Tracking (TrackNetV5)

**Status: In progress**

TrackNet is a deep learning model originally developed for tennis ball tracking, adapted here for pickleball. It inputs a sequence of frames and outputs the ball trajectory — including sub-frame-accurate positions even when the ball is motion-blurred or partially occluded.

### What was done
- Studied TrackNetV5 architecture and dataset format
- Began adapting dataset pipeline for pickleball footage
- Training notebooks: `pickleball/code/tracknet_v5.ipynb`, `pickleball/code/TracknetV5_ver2.ipynb`
- Inference test script: `pickleball/code/track_ball.py`

### What is left
- Complete dataset preparation in TrackNet format
- Train on pickleball footage
- Implement bounce point detection from trajectory (direction reversal)

---

## Model 3 — Court Keypoint Detection (YOLOv8-Pose)

**Status: In progress — model trained, post-processing built, retrain needed**

### Court keypoint schema — 12 keypoints

```
KP0 ─────── KP1 ─────── KP2      ← Far  baseline   (L_BL_BG, M_BL_BG, R_BL_BG)
 │           │           │
KP5 ─────── KP4 ─────── KP3      ← Far  kitchen    (L_KL_BG, M_KL_BG, R_KL_BG)
 │                       │
 │           NET          │
 │                       │
KP6 ─────── KP7 ─────── KP8      ← Near kitchen    (L_KL_FG, M_KL_FG, R_KL_FG)
 │           │           │
KP11─────── KP10─────── KP9      ← Near baseline   (L_BL_FG, M_BL_FG, R_BL_FG)
```

**Real-world coordinates (cm):**

| KP | Name | X_cm | Y_cm | Location |
|----|------|------|------|----------|
| 0 | L_BL_BG | 0 | 1372 | Far left baseline corner |
| 1 | M_BL_BG | 305 | 1372 | Far center baseline |
| 2 | R_BL_BG | 610 | 1372 | Far right baseline corner |
| 3 | R_KL_BG | 610 | 915 | Far right kitchen corner |
| 4 | M_KL_BG | 305 | 915 | Far center kitchen (near net) |
| 5 | L_KL_BG | 0 | 915 | Far left kitchen corner |
| 6 | L_KL_FG | 0 | 457 | Near left kitchen corner |
| 7 | M_KL_FG | 305 | 457 | Near center kitchen |
| 8 | R_KL_FG | 610 | 457 | Near right kitchen corner |
| 9 | R_BL_FG | 610 | 0 | Near right baseline corner |
| 10 | M_BL_FG | 305 | 0 | Near center baseline |
| 11 | L_BL_FG | 0 | 0 | Near left baseline corner (origin) |

Court dimensions: 610 cm wide × 1372 cm long. Kitchen line 457 cm from baseline. Net at 686 cm.

### What was done

**Training:**
- Annotated court keypoints using **Roboflow** (project `pb-rrerm/pb-9bsin`, version 5)
- 2914 training images, 407 validation images
- Trained **YOLOv8s-Pose** on Google Colab:
  - imgsz=640, epochs=100, batch=16, optimizer=AdamW, lr=0.001
  - `flip_idx=[2,1,0,5,4,3,8,7,6,11,10,9]` for correct left↔right symmetry
  - `kpt_shape=[12, 3]`

**Problems discovered and fixed:**
| Problem | Root cause | Fix applied |
|---------|-----------|-------------|
| ~0% detection on real video | Inference ran at `imgsz=1280` but model trained at `imgsz=640` | Changed `IMGSZ=640` in `court_detect.py` |
| Fake mAP=0.995 | Roboflow random per-frame split — adjacent frames leaked between train and val | Need clip-wise re-split (whole clips assigned to train or val) |
| Skeleton/colors coded for 14 KPs | Script written before training; model outputs 12 KPs | Updated to 12-KP schema |
| Domain gap on game_6 test video | game_6 had no annotated frames in training set | Need to annotate game_6 and retrain |

**Post-processing pipeline built** (see section below):

**Inference:**
```bash
python pickleball/code/court_detect.py
python pickleball/code/court_detect.py --video path/to/video.mp4 --conf 0.3
```

### What is left — retrain checklist
- [ ] Re-split Roboflow dataset `pb-rrerm/pb-9bsin` by clip (not random frames)
- [ ] Extract frames from game_6 (`pickleball/data/extract_frame.py`) and annotate 12 KPs
- [ ] Retrain with: `imgsz=1280, epochs=150, patience=30, scale=0.5, perspective=0.0005, shear=2, mixup=0.1`
- [ ] Honest val mAP should be 0.6–0.85 (NOT 0.995)
- [ ] Sanity check: run on held-out clip before exporting weights

### Key files
| File | Purpose |
|------|---------|
| `pickleball/code/court_detect.py` | Inference script with post-processing integrated |
| `pickleball/code/yolo_pose.ipynb` | YOLOv8-Pose training notebook (Colab) |
| `pickleball/code/POST_PROCESS.md` | Full post-processing specification (algorithm + interface) |
| `pickleball/code/post_process/refiner.py` | `CourtKeypointRefiner` — Hough refinement |
| `pickleball/code/post_process/temporal.py` | `TemporalSmoother` — EMA smoothing across frames |
| `pickleball/model/yolov8s_court_kp_best.pt` | Current model weights (12 KPs, imgsz=640) |

---

## Post-Processing — Court Keypoint Refinement

YOLO Pose predicts far-court keypoints (KP 0–5) with 15–30 px error due to perspective foreshortening — far court lines appear very thin (1–2 px wide). A Hough-based refinement step snaps each keypoint to the exact line intersection pixel.

### Algorithm (per keypoint)

```
1. Crop 120 px window around YOLO prediction (80 px for near-court KPs)
2. HSV masking — remove court surface (green H=35-85 or blue H=90-130),
   isolate white lines (S=0-50, V=170-255)
3. Morphological cleanup: dilate(3×3) then erode(3×3)
4. Thickening for far-court lines: dilate(5×5)×2 for baseline, (4×4)×2 for kitchen
5. Zhang-Suen thinning (requires opencv-contrib-python; skipped if unavailable)
6. HoughLines detection (threshold=10 for far, 15 for near)
7. Separate detected lines into horizontal and vertical groups
8. Solve 2-line intersection via numpy.linalg.solve
9. Distance gate: reject if moved > 80% of crop size
10. White-pixel verification: check 4 px radius neighborhood in original frame
11. Accept refined position or keep original YOLO prediction unchanged
```

### Refinement policy

| Group | KPs | Policy |
|-------|-----|--------|
| ALWAYS_REFINE | 0, 1, 2 | Far baseline — highest error, always run Hough |
| MAYBE_REFINE | 3, 5 | Far kitchen corners — skip if YOLO conf ≥ 0.5 |
| MAYBE_REFINE | 6, 7, 8 | Near kitchen — skip if YOLO conf ≥ 0.7 |
| NEVER_REFINE | 4 | Far center kitchen — net post area, unreliable |
| NEVER_REFINE | 9, 10, 11 | Near baseline — already very accurate from YOLO |

### Temporal smoothing

After Hough refinement, Exponential Moving Average (EMA) smoothing is applied across frames to reduce jitter in the homography matrix.

```
smoothed[t] = α × refined[t] + (1 − α) × smoothed[t−1]

α = 0.3 for far-court KPs (0–5)    — more smoothing
α = 0.5 for near-court KPs (6–11)  — less smoothing

Jump detection: if |refined[t] − smoothed[t−1]| > 25 px (far) or 15 px (near)
               → bad frame, keep smoothed[t−1]
```

### Expected improvement after retrain

| Metric | Before post-processing | After post-processing |
|--------|----------------------|-----------------------|
| Far court MPJPE (KP 0–2) | ~20–30 px | ~5–10 px |
| Near court MPJPE (KP 6–11) | ~3–5 px | ~3–5 px (unchanged) |
| Real-world error at far court | ~10–15 cm | ~2–5 cm |

Pickleball line width = 5 cm, so sub-5 cm precision is needed for accurate calls near the line.

---

## Data

```
pickleball/
├── vid/            Raw game footage (game1–game8, ~4 GB)         [gitignored]
├── data/           Extracted frames + annotations (~4 GB)        [gitignored]
│   ├── game_1/clip_1..5/frames/   ~610–713 frames per clip
│   ├── game_2..5/  annotated for ball detection (CVAT XML)
│   ├── game_6/     raw .mp4 only — no frames extracted yet
│   └── game_7/     extracted, not yet annotated for court KPs
├── model/          Model weights                                  [gitignored]
│   ├── best.pt                    YOLOv11s ball detector (19 MB)
│   ├── best_2.pt                  Alternative ball model (76 MB)
│   └── yolov8s_court_kp_best.pt  Court KP model (12 KPs, imgsz=640)
└── output/         Inference output videos                        [gitignored]
```

---

## Setup

```bash
# Core dependencies
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib pillow pyyaml

# Optional — enables Zhang-Suen thinning in post-processing (better Hough accuracy)
pip install opencv-contrib-python
```

GPU acceleration requires CUDA-compatible PyTorch. Training notebooks are designed for **Google Colab with A100 GPU**.

---

## Common Commands

```bash
# Extract frames from a raw video clip
python pickleball/data/extract_frame.py

# Run ball detection on a video
python pickleball/code/detect_ball.py --video path/to/video.mp4 --conf 0.25

# Run court keypoint detection on a video (with post-processing)
python pickleball/code/court_detect.py
python pickleball/code/court_detect.py --video path/to/video.mp4 --conf 0.3 --court-color green

# Convert CVAT XML annotations to CSV
python transform_xml_csv.py
```

---

## Roadmap

| Step | Status | Description |
|------|--------|-------------|
| Ball detection model | ✅ Done | YOLOv11s trained and running |
| Ball tracking model | 🚧 In progress | TrackNetV5 adaptation for pickleball |
| Court keypoint model v1 | ✅ Done | YOLOv8s-Pose, 12 KPs, needs retrain |
| Court KP post-processing | ✅ Done | Hough refinement + EMA smoother |
| Court KP retrain | 🔜 Next | Clip-wise split, game_6 data, stronger aug |
| Homography computation | 🔜 Planned | `cv2.findHomography` from 12 KPs → real-world cm |
| Bounce point detection | 🔜 Planned | From TrackNet trajectory |
| In/Out decision logic | 🔜 Planned | Transform bounce → court coords → compare boundary |
| End-to-end pipeline | 🔜 Planned | `model_combination.py` wires all three models |
