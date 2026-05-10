"""
Pickleball Ball Detection — Local Inference
============================================
Runs yolo11s best.pt on a video file and saves output with bounding boxes drawn.

Requirements:
    pip install ultralytics opencv-python

Usage:
    python detect_ball.py
    python detect_ball.py --video "C:/path/to/video.mp4"
    python detect_ball.py --video "C:/path/to/video.mp4" --conf 0.3
"""

import cv2
import torch
import argparse
import time
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────────
# CONFIGURATION — edit these
# ─────────────────────────────────────────────

# Path to your downloaded best.pt from Google Drive
MODEL_PATH  = r"C:\AI\pickleball\model\best.pt"

# Path to your input video
VIDEO_PATH  = r"C:\AI\pickleball\vid_test_model\combination_test_v2.mp4"

# Output folder — output video saved here automatically
OUTPUT_DIR  = r"C:\AI\pickleball\output"

# Detection settings
CONF        = 0.35     # raised from 0.25 — cuts weak false positives (green shirts, scoreboard)
IOU         = 0.45     # NMS IoU threshold — keep default unless boxes overlap weirdly
IMGSZ       = 640      # must match what you trained with

# ── Post-processing filters ──────────────────────────────
# Ball size in pixels at 1920×1080 (tune if your video is different resolution)
BALL_MIN_PX = 8        # smaller than this = noise, not a ball
BALL_MAX_PX = 80       # larger than this = player/clothing, not a ball
# Maximum aspect ratio (width/height) — ball is round so close to 1.0
BALL_MAX_ASPECT = 2.5  # shoe laces, elongated objects get filtered out
# Max ball travel distance between consecutive frames (pixels at original res)
# At 60fps, 150mph ball ≈ 90px/frame; allow 2× margin for fast shots
MAX_TRAVEL_PX = 200

# Visual settings
BOX_COLOR   = (0, 255, 0)       # green box (BGR)
TEXT_COLOR  = (0, 255, 0)       # green text
NO_DET_COLOR= (0, 0, 255)       # red dot when ball not detected
BOX_THICK   = 2
FONT        = cv2.FONT_HERSHEY_SIMPLEX
SHOW_TRAIL  = True              # draw ball position trail across frames
TRAIL_LEN   = 30                # how many frames to keep in trail
TRAIL_COLOR = (0, 165, 255)     # orange trail (BGR)

# ─────────────────────────────────────────────


def setup_gpu():
    """Check CUDA availability and print GPU info."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU : {gpu_name}  ({vram:.1f} GB VRAM)")
        return "cuda"
    else:
        print("WARNING: CUDA not found — running on CPU (will be slow)")
        print("  Make sure you have CUDA-enabled PyTorch:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return "cpu"


def get_output_path(video_path, output_dir):
    """Build output video path: output/<original_name>_detected.mp4"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    return str(Path(output_dir) / f"{stem}_detected.mp4")


def draw_trail(frame, trail):
    """Draw fading trail of previous ball positions."""
    for i, (cx, cy) in enumerate(trail):
        alpha  = (i + 1) / len(trail)          # older points = more transparent
        radius = max(2, int(4 * alpha))
        color  = tuple(int(c * alpha) for c in TRAIL_COLOR)
        cv2.circle(frame, (cx, cy), radius, color, -1)


def run_detection(video_path, model_path, output_dir, conf, iou, imgsz):
    device = setup_gpu()

    # ── Load model ──
    print(f"\nLoading model: {model_path}")
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("  Download best.pt from Google Drive and update MODEL_PATH")
        return
    model = YOLO(model_path)
    print(f"Model loaded ✓  (classes: {model.names})")

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

    # ── Setup output video writer ──
    output_path = get_output_path(video_path, output_dir)
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
    writer      = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"\nOutput will be saved to: {output_path}")
    print("-" * 60)

    # ── Detection loop ──
    frame_idx    = 0
    detected     = 0
    not_detected = 0
    trail        = []          # list of (cx, cy) for trail drawing
    last_cx      = None        # for travel-distance filter
    last_cy      = None
    start_time   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Run inference
        results = model.predict(
            source  = frame,
            conf    = conf,
            iou     = iou,
            imgsz   = imgsz,
            device  = device,
            verbose = False,
        )

        boxes = results[0].boxes

        # ── Filter candidate boxes ────────────────────────
        best_box  = None
        best_conf = -1.0
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy
                bw = x2 - x1
                bh = y2 - y1
                if bw <= 0 or bh <= 0:
                    continue

                # Size filter: reject too small or too large
                if bw < BALL_MIN_PX or bh < BALL_MIN_PX:
                    continue
                if bw > BALL_MAX_PX or bh > BALL_MAX_PX:
                    continue

                # Aspect ratio filter: reject elongated objects (shoe laces etc.)
                aspect = max(bw, bh) / min(bw, bh)
                if aspect > BALL_MAX_ASPECT:
                    continue

                c = float(boxes.conf[i])
                if c > best_conf:
                    best_conf = c
                    best_box  = xyxy

        # Travel distance filter: reject teleporting detections
        if best_box is not None and last_cx is not None:
            x1, y1, x2, y2 = best_box
            cx_new = (x1 + x2) // 2
            cy_new = (y1 + y2) // 2
            dist = ((cx_new - last_cx) ** 2 + (cy_new - last_cy) ** 2) ** 0.5
            if dist > MAX_TRAVEL_PX:
                best_box = None   # too far — likely a false positive

        ball_found = best_box is not None

        if ball_found:
            detected += 1

            x1, y1, x2, y2 = best_box
            conf_val = best_conf
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            last_cx, last_cy = cx, cy

            # Update trail
            trail.append((cx, cy))
            if len(trail) > TRAIL_LEN:
                trail.pop(0)

            # Draw trail
            if SHOW_TRAIL and len(trail) > 1:
                draw_trail(frame, trail)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)

            # Draw center dot
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Draw confidence label
            label = f"ball {conf_val:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), BOX_COLOR, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        else:
            not_detected += 1
            last_cx, last_cy = None, None   # reset travel filter on miss
            trail = []   # reset trail when ball disappears

            # Small red dot in corner = no detection this frame
            cv2.circle(frame, (30, 30), 8, NO_DET_COLOR, -1)
            cv2.putText(frame, "no ball", (44, 35),
                        FONT, 0.5, NO_DET_COLOR, 1, cv2.LINE_AA)

        # ── HUD overlay ──
        elapsed   = time.time() - start_time
        fps_actual = frame_idx / elapsed if elapsed > 0 else 0
        det_rate  = detected / frame_idx * 100

        hud_lines = [
            f"Frame : {frame_idx}/{total_frames}",
            f"FPS   : {fps_actual:.1f}",
            f"Det   : {det_rate:.1f}%",
            f"Conf  : {conf}",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(frame, line, (width - 160, 24 + i * 22),
                        FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)

        # Progress print every 100 frames
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            eta = (total_frames - frame_idx) / fps_actual if fps_actual > 0 else 0
            print(f"  Frame {frame_idx:5d}/{total_frames}  |  "
                  f"FPS: {fps_actual:5.1f}  |  "
                  f"Detection rate: {det_rate:5.1f}%  |  "
                  f"ETA: {eta:.0f}s")

    # ── Cleanup ──
    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Total frames   : {total_frames}")
    print(f"  Ball detected  : {detected}  ({detected/total_frames*100:.1f}%)")
    print(f"  Not detected   : {not_detected}  ({not_detected/total_frames*100:.1f}%)")
    print(f"  Processing time: {elapsed:.1f}s  ({total_frames/elapsed:.1f} fps avg)")
    print(f"  Output saved   : {output_path}")
    print("=" * 60)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickleball ball detection")
    parser.add_argument("--video",  default=VIDEO_PATH,  help="Path to input video")
    parser.add_argument("--model",  default=MODEL_PATH,  help="Path to best.pt")
    parser.add_argument("--output", default=OUTPUT_DIR,  help="Output folder")
    parser.add_argument("--conf",   default=CONF,   type=float, help="Confidence threshold")
    parser.add_argument("--iou",    default=IOU,    type=float, help="NMS IoU threshold")
    parser.add_argument("--imgsz",  default=IMGSZ,  type=int,   help="Inference image size")
    args = parser.parse_args()

    run_detection(
        video_path = args.video,
        model_path = args.model,
        output_dir = args.output,
        conf       = args.conf,
        iou        = args.iou,
        imgsz      = args.imgsz,
    )
