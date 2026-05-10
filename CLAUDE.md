# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Goal:** Automated in/out call system for pickleball, using computer vision to determine whether the ball lands inside or outside the court boundaries.

**Full Pipeline (3 models):**

1. **Ball Detection** — YOLOv11s ✅ Done
   - Detects ball position per frame; trained on CVAT-annotated frames

2. **Ball Tracking** — TrackNetV5 🚧 In progress
   - Tracks ball trajectory across frames; dataset format follows the TrackNet standard (originally for tennis, adapted for pickleball)
   - Labels created with CVAT

3. **Court Keypoint Detection** — YOLOv8-Pose 🚧 In progress
   - Detects 14 keypoints on the pickleball court (labeled in CVAT)
   - Will be used to compute the homography transform mapping court image coordinates → real-world court coordinates

**Decision Logic (planned):**
- Use homography (from 14 detected keypoints) to map pixel coordinates to real court coordinates
- Use TrackNet output to find the ball's bounce point (where trajectory direction changes)
- Transform bounce point to real-world coordinates → compare against court boundary → **in / out decision**

## Dependencies

No package manager config exists. Install manually:

```bash
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib pillow pyyaml
```

GPU acceleration requires CUDA-compatible PyTorch.

## Common Commands

```bash
# Extract frames from a video
python pickleball/data/extract_frame.py

# Convert CVAT XML annotations to CSV
python transform_xml_csv.py

# Run ball detection on a video
python pickleball/code/detect_ball.py --video <path_to_video.mp4> --conf 0.25

# Launch training/conversion notebooks locally
jupyter notebook pickleball/code/yolo11s_ball_training.ipynb
jupyter notebook pickleball/code/cvat_to_yolov11_colab.ipynb
```

Training is designed to run on **Google Colab with an A100 GPU**. The notebooks connect to Google Drive for data access.

## Architecture & Data Pipeline

```
Raw video (.mp4)
  → extract_frame.py         — extracts JPG frames at target FPS
  → CVAT annotation tool     — annotates ball positions (outputs annotations.xml)
  → cvat_to_yolov11_colab.ipynb  — converts CVAT XML point annotations to YOLO .txt labels,
                                    creates train/val split (80/20), generates data.yaml
  → yolo11s_ball_training.ipynb  — trains YOLOv11s on Colab A100
                                    (150 epochs, batch=16, imgsz=1280, early stopping patience=30)
  → pickleball/model/best.pt     — saved model weights
  → detect_ball.py               — inference: draws bounding boxes, confidence, motion trail overlay
```

### Key Files

| File | Purpose |
|------|---------|
| `pickleball/code/detect_ball.py` | Main inference script (251 lines) — loads `.pt` model, processes video frame-by-frame, writes annotated output |
| `pickleball/code/yolo11s_ball_training.ipynb` | YOLOv11s training notebook for Colab |
| `pickleball/code/cvat_to_yolov11_colab.ipynb` | Converts CVAT XML → YOLO format, builds dataset splits |
| `pickleball/data/extract_frame.py` | Frame extraction from raw video clips |
| `transform_xml_csv.py` | Standalone CVAT XML → CSV converter (alternative format) |

### Data Layout

- `pickleball/vid/` — raw game footage (game1–game5, ~4 GB)
- `pickleball/data/game_1..5/clip_1..5/frames/` — extracted JPG frames (~4 GB)
- `pickleball/model/best.pt` — primary model (YOLOv11s, 19 MB); `best_2.pt` — alternative (76 MB)
- `pickleball/output/` — inference output videos (~106 MB)
- `pickleball/vid_test_model/` — test videos for model evaluation

### Annotation Format

- Source: CVAT XML with **point** annotations (single pixel ball center, not bounding boxes)
- `cvat_to_yolov11_colab.ipynb` converts points → normalized YOLO bounding boxes using a fixed radius
- Labels have a single class: `ball` (class index 0)

### Training Configuration Notes

- Image size 1280×1280 (high res for small ball detection)
- Augmentation disabled for vertical flips and rotation (preserve court orientation)
- Mosaic augmentation enabled
- Optimizer: AdamW, lr=0.001
