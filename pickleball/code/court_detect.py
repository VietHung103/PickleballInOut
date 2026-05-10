"""
Pickleball Court Keypoint Detection — Local Inference
======================================================
Runs yolov8s_court_kp_best.pt (YOLOv8-Pose) on a video and saves
output with 12 court keypoints, Hough refinement, and temporal smoothing.

Requirements:
    pip install ultralytics opencv-python numpy
    pip install opencv-contrib-python   # optional — enables Zhang-Suen thinning

Usage:
    python court_detect.py
    python court_detect.py --video "C:/path/to/video.mp4"
    python court_detect.py --video "C:/path/to/video.mp4" --conf 0.3
"""

import cv2
import sys
import torch
import argparse
import time
from pathlib import Path

# Allow importing post_process package from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from post_process import CourtKeypointRefiner, TemporalSmoother

from ultralytics import YOLO


# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────

MODEL_PATH = r"C:\AI\pickleball\model\yolov8s_court_kp_best.pt"
VIDEO_PATH = r"C:\AI\pickleball\data\game_6\clip_1\0505.mp4"
OUTPUT_DIR = r"C:\AI\pickleball\output"

# Detection settings
CONF  = 0.3
IOU   = 0.45
IMGSZ = 640    # must match training resolution (model trained at imgsz=640)

# Only draw keypoints whose confidence is above this
KP_CONF_THRESH = 0.3

# Court colour for HSV preprocessing: 'green' or 'blue'
COURT_COLOR = 'green'

# ── 12-keypoint schema (Roboflow project pb-rrerm/pb-9bsin v5) ─────────────
# Camera is on the near/foreground side looking toward the far baseline.
# kpt_shape=[12,3], flip_idx=[2,1,0,5,4,3,8,7,6,11,10,9]
#
#  KP0 ─────── KP1 ─────── KP2      ← Far  baseline   (L_BL_BG, M_BL_BG, R_BL_BG)
#   │           │           │
#  KP5 ─────── KP4 ─────── KP3      ← Far  kitchen    (L_KL_BG, M_KL_BG, R_KL_BG)
#   │                       │
#   │          NET           │
#   │                       │
#  KP6 ─────── KP7 ─────── KP8      ← Near kitchen    (L_KL_FG, M_KL_FG, R_KL_FG)
#   │           │           │
#  KP11─────── KP10─────── KP9      ← Near baseline   (L_BL_FG, M_BL_FG, R_BL_FG)
#
KP_NAMES = [
    'L_BL_BG', 'M_BL_BG', 'R_BL_BG',   # 0-2  far  baseline
    'R_KL_BG', 'M_KL_BG', 'L_KL_BG',   # 3-5  far  kitchen  (R→L order)
    'L_KL_FG', 'M_KL_FG', 'R_KL_FG',   # 6-8  near kitchen
    'R_BL_FG', 'M_BL_FG', 'L_BL_FG',   # 9-11 near baseline (R→L order)
]

SKELETON = [
    # Far baseline
    (0, 1), (1, 2),
    # Far sidelines (baseline → kitchen)
    (0, 5), (2, 3),
    # Far kitchen line (note: KP5=left, KP4=center, KP3=right)
    (5, 4), (4, 3),
    # Far center service line
    (1, 4),
    # X diagonals — far-left service box
    (0, 4), (1, 5),
    # X diagonals — far-right service box
    (1, 3), (2, 4),
    # Sidelines + center through net zone
    (5, 6), (3, 8), (4, 7),
    # Near kitchen line
    (6, 7), (7, 8),
    # Near center service line
    (7, 10),
    # X diagonals — near-left service box
    (6, 10), (7, 11),
    # X diagonals — near-right service box
    (7, 9), (8, 10),
    # Near sidelines (kitchen → baseline)
    (6, 11), (8, 9),
    # Near baseline
    (11, 10), (10, 9),
]

# Colours: cyan = baseline rows, orange = kitchen rows
KP_COLORS = [
    (0, 255, 255),   # 0  far-left  baseline
    (0, 255, 255),   # 1  far-center baseline
    (0, 255, 255),   # 2  far-right baseline
    (0, 180, 255),   # 3  far-right kitchen
    (0, 180, 255),   # 4  far-center kitchen
    (0, 180, 255),   # 5  far-left  kitchen
    (0, 180, 255),   # 6  near-left  kitchen
    (0, 180, 255),   # 7  near-center kitchen
    (0, 180, 255),   # 8  near-right kitchen
    (0, 255, 255),   # 9  near-right baseline
    (0, 255, 255),   # 10 near-center baseline
    (0, 255, 255),   # 11 near-left  baseline
]

REFINED_COLOR  = (0, 255, 0)      # bright green dot = Hough-refined
SKELETON_COLOR = (0, 255, 0)      # green skeleton lines
KP_RADIUS      = 5
SKEL_THICK     = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# ─────────────────────────────────────────────


def setup_gpu():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU : {gpu_name}  ({vram:.1f} GB VRAM)")
        return "cuda"
    print("WARNING: CUDA not found — running on CPU (will be slow)")
    return "cpu"


def get_output_path(video_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    return str(Path(output_dir) / f"{stem}_court_kp.mp4")


def yolo_to_kp_dicts(kps_xy, kps_conf):
    """Convert YOLO numpy arrays to the dict-list format the refiner expects."""
    return [
        {
            'id':   i,
            'name': KP_NAMES[i],
            'x':    float(kps_xy[i][0]),
            'y':    float(kps_xy[i][1]),
            'conf': float(kps_conf[i]),
        }
        for i in range(len(kps_xy))
    ]


def draw_results(frame, raw_kps, smoothed_kps):
    """
    Draw skeleton from smoothed positions, then keypoint dots.
    Raw (YOLO-only) positions are shown as hollow rings.
    Refined + smoothed positions are shown as filled circles.
    """
    n = len(smoothed_kps)

    # Build lookup: id → dict
    raw_map      = {kp['id']: kp for kp in raw_kps}
    smoothed_map = {kp['id']: kp for kp in smoothed_kps}

    # ── Skeleton from smoothed positions ──────────────────────────────
    for i, j in SKELETON:
        kp_i = smoothed_map.get(i)
        kp_j = smoothed_map.get(j)
        if kp_i is None or kp_j is None:
            continue
        if kp_i['conf'] < KP_CONF_THRESH or kp_j['conf'] < KP_CONF_THRESH:
            continue
        pt1 = (int(kp_i['x']), int(kp_i['y']))
        pt2 = (int(kp_j['x']), int(kp_j['y']))
        cv2.line(frame, pt1, pt2, SKELETON_COLOR, SKEL_THICK, cv2.LINE_AA)

    # ── Keypoint dots ─────────────────────────────────────────────────
    for kp in smoothed_kps:
        idx  = kp['id']
        conf = kp['conf']
        if conf < KP_CONF_THRESH:
            continue

        color = KP_COLORS[idx] if idx < len(KP_COLORS) else (255, 255, 255)
        sx, sy = int(kp['x']), int(kp['y'])

        # Raw YOLO position — hollow ring (so you can see the shift)
        raw = raw_map.get(idx)
        if raw and raw['conf'] >= KP_CONF_THRESH:
            rx, ry = int(raw['x']), int(raw['y'])
            cv2.circle(frame, (rx, ry), KP_RADIUS + 2, color, 1, cv2.LINE_AA)

        # Smoothed/refined position — filled dot
        dot_color = REFINED_COLOR if kp.get('refined') else color
        cv2.circle(frame, (sx, sy), KP_RADIUS, dot_color, -1, cv2.LINE_AA)

        # Index label
        cv2.putText(frame, str(idx), (sx + 7, sy - 7),
                    FONT, 0.4, dot_color, 1, cv2.LINE_AA)


def run_detection(video_path, model_path, output_dir, conf, iou, imgsz):
    device = setup_gpu()

    # ── Load model ──
    print(f"\nLoading model: {model_path}")
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    model = YOLO(model_path)
    print("Model loaded ✓")

    # ── Post-processing pipeline ──
    refiner  = CourtKeypointRefiner(court_color=COURT_COLOR, verbose=False)
    smoother = TemporalSmoother()
    if refiner.thinning_available:
        print("Zhang-Suen thinning: enabled (opencv-contrib found)")
    else:
        print("Zhang-Suen thinning: disabled (install opencv-contrib-python to enable)")

    # ── Open video ──
    print(f"\nOpening video: {video_path}")
    if not Path(video_path).exists():
        print(f"ERROR: Video not found at {video_path}")
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Could not open video file")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video info: {width}x{height}  {fps:.1f}fps  {total_frames} frames")
    print(f"Duration  : {total_frames/fps:.1f} seconds")

    output_path = get_output_path(video_path, output_dir)
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
    writer      = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"\nOutput will be saved to: {output_path}")
    print("-" * 60)

    # ── Detection loop ──
    frame_idx    = 0
    court_found  = 0
    court_missed = 0
    refined_total = 0
    start_time   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = model.predict(
            source  = frame,
            conf    = conf,
            iou     = iou,
            imgsz   = imgsz,
            device  = device,
            verbose = False,
        )

        result    = results[0]
        keypoints = result.keypoints

        detected = False

        if keypoints is not None and len(keypoints) > 0:
            kp_data = keypoints.data.cpu().numpy()  # (N, 12, 3)

            # Pick detection with highest mean keypoint confidence
            best_idx       = 0
            best_mean_conf = -1.0
            for i in range(len(kp_data)):
                mc = float(kp_data[i, :, 2].mean())
                if mc > best_mean_conf:
                    best_mean_conf = mc
                    best_idx = i

            kps      = kp_data[best_idx]   # (12, 3)
            kps_xy   = kps[:, :2]
            kps_conf = kps[:, 2]

            visible = int((kps_conf >= KP_CONF_THRESH).sum())

            if visible > 0:
                detected = True
                court_found += 1

                # ── Post-processing ──────────────────────────────────
                raw_kps      = yolo_to_kp_dicts(kps_xy, kps_conf)
                refined_kps, log = refiner.refine(frame, raw_kps)
                smoothed_kps = smoother.update(refined_kps)

                n_refined = sum(1 for kp in refined_kps if kp.get('refined'))
                refined_total += n_refined

                # ── Draw bounding box ────────────────────────────────
                if result.boxes is not None and len(result.boxes) > 0:
                    box = result.boxes.xyxy[best_idx].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)
                    label = (f"court  {best_mean_conf:.2f}  "
                             f"kp:{visible}/12  refined:{n_refined}")
                    cv2.putText(frame, label, (x1, y1 - 6),
                                FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                draw_results(frame, raw_kps, smoothed_kps)

        if not detected:
            court_missed += 1
            smoother.reset()  # reset EMA on missed frames
            cv2.circle(frame, (30, 30), 8, (0, 0, 255), -1)
            cv2.putText(frame, "no court", (44, 35),
                        FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # ── HUD overlay ──
        elapsed    = time.time() - start_time
        fps_actual = frame_idx / elapsed if elapsed > 0 else 0
        det_rate   = court_found / frame_idx * 100

        hud_lines = [
            f"Frame : {frame_idx}/{total_frames}",
            f"FPS   : {fps_actual:.1f}",
            f"Det   : {det_rate:.1f}%",
            f"Conf  : {conf}",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(frame, line, (width - 170, 24 + i * 22),
                        FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)

        if frame_idx % 100 == 0 or frame_idx == total_frames:
            eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            print(f"  Frame {frame_idx:5d}/{total_frames}  |  "
                  f"FPS: {fps_actual:5.1f}  |  "
                  f"Detection rate: {det_rate:5.1f}%  |  "
                  f"ETA: {eta:.0f}s")

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Total frames     : {total_frames}")
    print(f"  Court detected   : {court_found}  ({court_found/total_frames*100:.1f}%)")
    print(f"  Not detected     : {court_missed}  ({court_missed/total_frames*100:.1f}%)")
    print(f"  KPs Hough-refined: {refined_total} total across all frames")
    print(f"  Processing time  : {elapsed:.1f}s  ({total_frames/elapsed:.1f} fps avg)")
    print(f"  Output saved     : {output_path}")
    print("=" * 60)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickleball court keypoint detection")
    parser.add_argument("--video",  default=VIDEO_PATH,  help="Path to input video")
    parser.add_argument("--model",  default=MODEL_PATH,  help="Path to model .pt")
    parser.add_argument("--output", default=OUTPUT_DIR,  help="Output folder")
    parser.add_argument("--conf",   default=CONF,   type=float, help="Confidence threshold")
    parser.add_argument("--iou",    default=IOU,    type=float, help="NMS IoU threshold")
    parser.add_argument("--imgsz",  default=IMGSZ,  type=int,   help="Inference image size")
    parser.add_argument("--court-color", default=COURT_COLOR,
                        choices=['green', 'blue'], help="Court surface color for HSV masking")
    args = parser.parse_args()

    run_detection(
        video_path = args.video,
        model_path = args.model,
        output_dir = args.output,
        conf       = args.conf,
        iou        = args.iou,
        imgsz      = args.imgsz,
    )
