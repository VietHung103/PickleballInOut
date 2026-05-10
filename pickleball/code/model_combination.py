"""
Combined YOLO + TrackNetV5 Pickleball Inference
================================================
Runs both models on the same video and fuses their outputs into one
annotated video.

Pipeline (6 phases):
  1. Frame Buffer   — load all frames into RAM
  2. YOLO Pass      — per-frame bounding-box detection
  3. TrackNetV5 Pass— 3-frame heatmap tracking
  4. Fusion Pass    — combine results per frame
  5. Smoothing      — median filter on fused trajectory
  6. Render Pass    — draw all annotations, write output video

Usage:
  python model_combination.py
  python model_combination.py --video path/to/video.mp4
  python model_combination.py --yolo-conf 0.3 --tn-thresh 0.45
  python model_combination.py --max-frames 300   # test on first N frames
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
from pathlib import Path
from scipy.signal import medfilt
from collections import deque
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

DEFAULT_VIDEO         = r'C:\AI\pickleball\vid\game6\Christopher Haworth v Tama Shimabukuro at the Veolia Atlanta Pickleball Championships.mp4'
DEFAULT_YOLO_PATH     = r'C:\AI\pickleball\model\best_2.pt'
DEFAULT_TRACKNET_PATH = r'C:\AI\pickleball\model\tracknet_plot\best_tracknetv5.pth'
DEFAULT_OUTPUT        = None       # None → auto-named as {stem}_combined.mp4
DEFAULT_DEVICE        = 'cuda'

# YOLO inference
YOLO_CONF             = 0.35
YOLO_IOU              = 0.45
YOLO_IMGSZ            = 1280

# YOLO post-processing filters
YOLO_BALL_MIN_PX      = 8
YOLO_BALL_MAX_PX      = 80
YOLO_MAX_ASPECT       = 2.5
YOLO_MAX_TRAVEL_PX    = 200

# TrackNetV5 input resolution (must match training)
TN_IMG_W              = 512
TN_IMG_H              = 288

# TrackNetV5 post-processing filters
TN_THRESH             = 0.5
TN_BLOB_AREA_MIN      = 4
TN_BLOB_AREA_MAX      = 1500
TN_MAX_TRAVEL_PX      = 200

# Fusion
FUSION_AGREE_DIST_PX  = 80    # max pixel distance for BOTH label

# Trajectory smoothing
SMOOTH_WINDOW         = 5     # median filter kernel (must be odd)
SMOOTH_MAX_GAP        = 3     # interpolate gaps within ±3 frames

# Visual
TRAIL_LEN             = 15
YOLO_BOX_COLOR        = (0, 255, 0)    # BGR green
TN_RING_COLOR         = (255, 255, 0)  # BGR cyan
FUSED_DOT_COLOR       = (0, 255, 255)  # BGR yellow
FUSED_DOT_RADIUS      = 8
TN_RING_RADIUS        = 13
TN_RING_THICKNESS     = 2
HUD_HEIGHT            = 38


# ─────────────────────────────────────────────────────────
# TRACKNETV5 MODEL ARCHITECTURE
# (copied from track_ball.py — must match saved checkpoint exactly)
# ─────────────────────────────────────────────────────────

class MDDLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta  = nn.Parameter(torch.zeros(1))

    def _attention(self, polarity):
        k = 5.0 / (0.45 * torch.tanh(self.alpha).abs() + 1e-6)
        m = 0.6 * torch.tanh(self.beta)
        return torch.sigmoid(k * (polarity.abs() - m))

    def forward(self, frames):
        I_prev = frames[:, 0:3]
        I_curr = frames[:, 3:6]
        I_next = frames[:, 6:9]
        D1     = I_curr - I_prev
        D2     = I_next - I_curr
        D1_g   = D1.mean(dim=1, keepdim=True)
        D2_g   = D2.mean(dim=1, keepdim=True)
        A1 = torch.cat([self._attention(F.relu( D1_g)),
                        self._attention(F.relu(-D1_g))], dim=1)
        A2 = torch.cat([self._attention(F.relu( D2_g)),
                        self._attention(F.relu(-D2_g))], dim=1)
        Xin         = torch.cat([I_prev, A1, I_curr, A2, I_next], dim=1)
        motion_attn = torch.cat([A1, A2], dim=1)
        return Xin, motion_attn


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )
    def forward(self, x): return self.block(x)


class V2Backbone(nn.Module):
    def __init__(self, in_ch=13):
        super().__init__()
        self.enc1       = DoubleConv(in_ch, 64)
        self.pool1      = nn.MaxPool2d(2)
        self.enc2       = DoubleConv(64, 128)
        self.pool2      = nn.MaxPool2d(2)
        self.enc3       = DoubleConv(128, 256)
        self.pool3      = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up3        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3       = DoubleConv(512 + 256, 256)
        self.up2        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2       = DoubleConv(256 + 128, 128)
        self.up1        = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1       = DoubleConv(128 + 64, 64)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return d1


class TSATTHead(nn.Module):
    def __init__(self, in_ch=5, patch=8, dim=128, heads=4, layers=2):
        super().__init__()
        self.patch       = patch
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        # Use TN_IMG_W/H (renamed from IMG_W/H in track_ball.py)
        max_tokens       = (TN_IMG_H // patch) * (TN_IMG_W // patch)
        self.pos_embed   = nn.Parameter(torch.randn(1, max_tokens, dim) * 0.02)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, batch_first=True, norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(nn.Linear(dim, patch * patch))

    def forward(self, x):
        B, C, H, W = x.shape
        tokens = self.patch_embed(x)
        ph, pw = tokens.shape[2], tokens.shape[3]
        tokens = tokens.flatten(2).transpose(1, 2)
        N      = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :N, :]
        tokens = self.transformer(tokens)
        pixels = self.head(tokens)
        pixels = pixels.transpose(1, 2).reshape(B, self.patch * self.patch, ph, pw)
        return F.pixel_shuffle(pixels, self.patch)


class RSTR(nn.Module):
    def __init__(self, feat_ch=64, dropout_p=0.1):
        super().__init__()
        self.draft_conv = nn.Conv2d(feat_ch, 1, kernel_size=1)
        self.tsatt      = TSATTHead(in_ch=5)
        self.dropout    = nn.Dropout2d(p=dropout_p)

    def forward(self, dec_feat, motion_attn):
        draft     = self.draft_conv(dec_feat)
        draft_mdd = torch.cat([draft, motion_attn], dim=1)
        delta     = self.tsatt(draft_mdd)
        return torch.sigmoid(draft + delta)


class TrackNetV5(nn.Module):
    def __init__(self):
        super().__init__()
        self.mdd      = MDDLayer()
        self.backbone = V2Backbone(in_ch=13)
        self.rstr     = RSTR(feat_ch=64)

    def forward(self, frames):
        Xin, motion_attn = self.mdd(frames)
        dec_feat          = self.backbone(Xin)
        return self.rstr(dec_feat, motion_attn)


# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────

def preprocess_frame(frame_bgr):
    """Resize and normalise a BGR frame for TrackNetV5. Returns (3, H, W) float32 tensor."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (TN_IMG_W, TN_IMG_H))
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0


def extract_ball_center(heatmap_tensor, threshold=TN_THRESH):
    """
    Find the largest valid blob in a (1,1,H,W) sigmoid heatmap.
    Returns (cx, cy) in TrackNetV5 model space or None.
    """
    hm     = heatmap_tensor.squeeze().cpu().numpy()
    binary = (hm >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c    = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < TN_BLOB_AREA_MIN or area > TN_BLOB_AREA_MAX:
        return None
    M = cv2.moments(c)
    if M['m00'] == 0:
        return None
    return (M['m10'] / M['m00'], M['m01'] / M['m00'])


def model_to_orig(cx, cy, orig_w, orig_h):
    """Convert TrackNetV5 model-space coords to original video pixel coords."""
    return (int(cx * orig_w / TN_IMG_W), int(cy * orig_h / TN_IMG_H))


def apply_yolo_filters(boxes, last_cx, last_cy):
    """
    Apply size, aspect-ratio, and travel-distance filters to raw YOLO boxes.

    Returns (xyxy_tuple | None, confidence).
    Caller must reset last_cx/last_cy to None when return is None.
    """
    best_box  = None
    best_conf = 0.0

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            if bw < YOLO_BALL_MIN_PX or bh < YOLO_BALL_MIN_PX:
                continue
            if bw > YOLO_BALL_MAX_PX or bh > YOLO_BALL_MAX_PX:
                continue
            if max(bw, bh) / min(bw, bh) > YOLO_MAX_ASPECT:
                continue
            c = float(boxes.conf[i])
            if c > best_conf:
                best_conf = c
                best_box  = tuple(xyxy)

    if best_box is not None and last_cx is not None:
        x1, y1, x2, y2 = best_box
        cx_new = (x1 + x2) // 2
        cy_new = (y1 + y2) // 2
        dist = ((cx_new - last_cx) ** 2 + (cy_new - last_cy) ** 2) ** 0.5
        if dist > YOLO_MAX_TRAVEL_PX:
            return None, 0.0

    return best_box, best_conf


def fuse_detections(yolo_center, tn_center, agree_dist=FUSION_AGREE_DIST_PX):
    """
    Fuse per-frame YOLO and TrackNetV5 results.

    Returns (fused_center | None, source_str).
    source_str is one of: 'BOTH', 'YOLO', 'TRACKNET', 'NONE'.

    When both fire but disagree (dist >= agree_dist): trust TrackNetV5 because
    temporal context makes it more spatially precise. YOLO box is still drawn.
    """
    if yolo_center is not None and tn_center is not None:
        dist = ((yolo_center[0] - tn_center[0]) ** 2 +
                (yolo_center[1] - tn_center[1]) ** 2) ** 0.5
        if dist < agree_dist:
            avg = ((yolo_center[0] + tn_center[0]) // 2,
                   (yolo_center[1] + tn_center[1]) // 2)
            return avg, 'BOTH'
        else:
            return tn_center, 'TRACKNET'
    elif yolo_center is not None:
        return yolo_center, 'YOLO'
    elif tn_center is not None:
        return tn_center, 'TRACKNET'
    else:
        return None, 'NONE'


def smooth_coords(vals, window=SMOOTH_WINDOW):
    """NaN-interpolate gaps then apply a median filter."""
    arr      = np.array(vals, dtype=np.float64)
    nan_mask = np.isnan(arr)
    if nan_mask.all():
        return arr
    arr[nan_mask] = np.interp(
        np.flatnonzero(nan_mask),
        np.flatnonzero(~nan_mask),
        arr[~nan_mask]
    )
    return medfilt(arr, kernel_size=window)


def draw_trail(frame, trail):
    """Draw fading yellow-green ghost circles for the last N fused positions."""
    n = len(trail)
    for i, pt in enumerate(trail):
        if pt is None:
            continue
        t       = (i + 1) / n
        radius  = max(4, int(10 * t))
        opacity = 0.25 + 0.70 * t
        color   = (0, int(180 + 75 * t), int(160 + 95 * t))
        overlay = frame.copy()
        cv2.circle(overlay, pt, radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


def draw_hud(frame, frame_idx, total, source, fps_proc):
    """
    Draw semi-transparent top bar with source label, frame counter, FPS.
    Colour by source: BOTH=bright-green, YOLO=green, TRACKNET=cyan, NONE=red-orange.
    """
    source_colors = {
        'BOTH':     (0, 255, 128),
        'YOLO':     (0, 255, 0),
        'TRACKNET': (255, 255, 0),
        'NONE':     (0, 80, 255),
    }
    color   = source_colors.get(source, (200, 200, 200))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], HUD_HEIGHT), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    fps_str = f"{fps_proc:.1f} fps" if fps_proc > 0 else "-- fps"
    text    = f"[{source}]  Frame {frame_idx+1}/{total}  |  {fps_str}"
    cv2.putText(frame, text, (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────
# MAIN ORCHESTRATION
# ─────────────────────────────────────────────────────────

def run_combined(
    video_path,
    yolo_path,
    tracknet_path,
    output_path,
    device      = DEFAULT_DEVICE,
    yolo_conf   = YOLO_CONF,
    yolo_iou    = YOLO_IOU,
    yolo_imgsz  = YOLO_IMGSZ,
    tn_thresh   = TN_THRESH,
    trail_len   = TRAIL_LEN,
    smooth_window = SMOOTH_WINDOW,
    agree_dist  = FUSION_AGREE_DIST_PX,
    max_frames  = None,
):
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU.")
        device = 'cpu'
    device = torch.device(device)

    if device.type == 'cuda':
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    # ── Load YOLO ────────────────────────────────────────
    print(f"\nLoading YOLO: {yolo_path}")
    if not Path(yolo_path).exists():
        raise FileNotFoundError(f"YOLO model not found: {yolo_path}")
    yolo_model = YOLO(yolo_path)
    print(f"  YOLO loaded ✓  classes: {yolo_model.names}")

    # ── Load TrackNetV5 ──────────────────────────────────
    print(f"Loading TrackNetV5: {tracknet_path}")
    if not Path(tracknet_path).exists():
        raise FileNotFoundError(f"TrackNetV5 checkpoint not found: {tracknet_path}")
    ckpt      = torch.load(tracknet_path, map_location=device)
    tn_model  = TrackNetV5().to(device)
    state     = ckpt.get('model_state', ckpt)   # handles wrapped or raw state_dicts
    tn_model.load_state_dict(state)
    tn_model.eval()
    best_f1   = ckpt.get('best_f1', 'N/A')
    print(f"  TrackNetV5 loaded ✓  best_f1={best_f1}")

    # ── Open video ───────────────────────────────────────
    print(f"\nOpening video: {video_path}")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap     = cv2.VideoCapture(video_path)
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_f = min(total_f, max_frames)
    print(f"  {orig_w}x{orig_h} @ {fps:.1f}fps | {total_f} frames")

    # ── Phase 1: Frame Buffer ────────────────────────────
    print("\nPhase 1: Buffering frames...")
    all_frames = []
    while len(all_frames) < total_f:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
    cap.release()
    N = len(all_frames)
    print(f"  {N} frames buffered.")

    # ── Phase 2: YOLO Pass ───────────────────────────────
    print("\nPhase 2: YOLO inference pass...")
    yolo_results = [None] * N   # (cx, cy) in original pixels
    yolo_boxes   = [None] * N   # (x1,y1,x2,y2) for drawing
    yolo_confs   = [0.0]  * N
    last_cx = last_cy = None

    t_yolo = time.time()
    for i, frame in enumerate(all_frames):
        raw     = yolo_model.predict(
            source=frame, conf=yolo_conf, iou=yolo_iou,
            imgsz=yolo_imgsz, device=str(device), verbose=False
        )
        best_box, best_conf = apply_yolo_filters(raw[0].boxes, last_cx, last_cy)

        if best_box is not None:
            x1, y1, x2, y2    = best_box
            cx, cy             = (x1 + x2) // 2, (y1 + y2) // 2
            yolo_results[i]    = (cx, cy)
            yolo_boxes[i]      = best_box
            yolo_confs[i]      = best_conf
            last_cx, last_cy   = cx, cy
        else:
            last_cx = last_cy = None

        if (i + 1) % 100 == 0 or i == N - 1:
            det = sum(r is not None for r in yolo_results[:i+1])
            print(f"  YOLO {i+1}/{N}  detections: {det} ({det/(i+1)*100:.1f}%)")

    yolo_time = time.time() - t_yolo
    yolo_det  = sum(r is not None for r in yolo_results)
    print(f"  YOLO pass done in {yolo_time:.1f}s | {yolo_det}/{N} detections")

    # ── Phase 3: TrackNetV5 Pass ─────────────────────────
    print("\nPhase 3: TrackNetV5 inference pass...")
    tn_results = [None] * N   # (cx, cy) in original pixels
    last_cx = last_cy = None

    # Pre-process all frames once
    preprocessed = [preprocess_frame(f) for f in all_frames]

    t_tn = time.time()
    with torch.no_grad():
        for i in range(1, N - 1):
            triplet = torch.cat([
                preprocessed[i - 1],
                preprocessed[i],
                preprocessed[i + 1]
            ], dim=0).unsqueeze(0).to(device)          # (1, 9, H, W)

            heatmap = tn_model(triplet)                # (1, 1, TN_IMG_H, TN_IMG_W)
            center  = extract_ball_center(heatmap, tn_thresh)

            if center is not None:
                cx_orig, cy_orig = model_to_orig(*center, orig_w, orig_h)
                if last_cx is not None:
                    dist = ((cx_orig - last_cx) ** 2 + (cy_orig - last_cy) ** 2) ** 0.5
                    if dist > TN_MAX_TRAVEL_PX:
                        center = None

            if center is not None:
                tn_results[i] = (cx_orig, cy_orig)
                last_cx, last_cy = cx_orig, cy_orig
            else:
                last_cx = last_cy = None

            if (i + 1) % 100 == 0 or i == N - 2:
                det = sum(r is not None for r in tn_results[:i+1])
                print(f"  TrackNet {i+1}/{N}  detections: {det} ({det/(i+1)*100:.1f}%)")

    del preprocessed   # free memory before render pass

    tn_time = time.time() - t_tn
    tn_det  = sum(r is not None for r in tn_results)
    print(f"  TrackNetV5 pass done in {tn_time:.1f}s | {tn_det}/{N} detections")

    # ── Phase 4: Fusion Pass ─────────────────────────────
    print("\nPhase 4: Fusing results...")
    fused_results = [None] * N
    fused_sources = ['NONE'] * N

    for i in range(N):
        center, source    = fuse_detections(yolo_results[i], tn_results[i], agree_dist)
        fused_results[i]  = center
        fused_sources[i]  = source

    both_cnt  = fused_sources.count('BOTH')
    yolo_cnt  = fused_sources.count('YOLO')
    tn_cnt    = fused_sources.count('TRACKNET')
    none_cnt  = fused_sources.count('NONE')
    print(f"  BOTH={both_cnt}  YOLO={yolo_cnt}  TRACKNET={tn_cnt}  NONE={none_cnt}")

    # ── Phase 5: Smoothing ───────────────────────────────
    print("\nPhase 5: Smoothing trajectory...")
    raw_xs = [float(r[0]) if r is not None else np.nan for r in fused_results]
    raw_ys = [float(r[1]) if r is not None else np.nan for r in fused_results]

    smooth_xs = smooth_coords(raw_xs, window=smooth_window)
    smooth_ys = smooth_coords(raw_ys, window=smooth_window)

    smoothed = []
    for i in range(N):
        if fused_results[i] is not None:
            smoothed.append((int(smooth_xs[i]), int(smooth_ys[i])))
        else:
            lo     = max(0, i - SMOOTH_MAX_GAP)
            hi     = min(N, i + SMOOTH_MAX_GAP + 1)
            nearby = any(fused_results[j] is not None for j in range(lo, hi) if j != i)
            if nearby and not np.isnan(raw_xs[i]):
                smoothed.append((int(smooth_xs[i]), int(smooth_ys[i])))
            else:
                smoothed.append(None)

    # ── Phase 6: Render Pass ─────────────────────────────
    if output_path is None:
        p           = Path(video_path)
        output_path = str(p.parent / (p.stem + '_combined.mp4'))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))
    print(f"\nPhase 6: Rendering → {output_path}")

    trail   = deque(maxlen=trail_len)
    t_start = time.time()

    for i in range(N):
        frame  = all_frames[i].copy()
        pt     = smoothed[i]
        source = fused_sources[i]
        trail.append(pt)

        # Layer 1: Trail
        draw_trail(frame, list(trail))

        # Layer 2: TrackNetV5 cyan ring
        if tn_results[i] is not None:
            cx, cy = tn_results[i]
            cv2.circle(frame, (cx, cy), TN_RING_RADIUS,
                       TN_RING_COLOR, TN_RING_THICKNESS, cv2.LINE_AA)
            cv2.putText(frame, "TN", (cx + 16, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, TN_RING_COLOR, 1, cv2.LINE_AA)

        # Layer 3: YOLO green bounding box + label
        if yolo_boxes[i] is not None:
            x1, y1, x2, y2 = yolo_boxes[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), YOLO_BOX_COLOR, 2)
            label = f"YOLO {yolo_confs[i]:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 4, y1), YOLO_BOX_COLOR, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Layer 4: Fused yellow dot
        if pt is not None:
            cv2.circle(frame, pt, FUSED_DOT_RADIUS, FUSED_DOT_COLOR, -1, cv2.LINE_AA)

        # Layer 5: HUD
        fps_proc = (i + 1) / (time.time() - t_start)
        draw_hud(frame, i, N, source, fps_proc)

        writer.write(frame)

        if (i + 1) % 100 == 0 or i == N - 1:
            print(f"  Rendered {i+1}/{N} frames", end='\r')

    writer.release()
    total_time = time.time() - t_start

    print(f"\n\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"  Video          : {video_path}")
    print(f"  Frames         : {N}")
    print(f"  BOTH agreed    : {both_cnt}  ({both_cnt/N*100:.1f}%)")
    print(f"  Only YOLO      : {yolo_cnt}  ({yolo_cnt/N*100:.1f}%)")
    print(f"  Only TrackNet  : {tn_cnt}  ({tn_cnt/N*100:.1f}%)")
    print(f"  No detection   : {none_cnt}  ({none_cnt/N*100:.1f}%)")
    print(f"  Render time    : {total_time:.1f}s")
    print(f"  Output saved   : {output_path}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Combined YOLO + TrackNetV5 Pickleball Inference'
    )
    parser.add_argument('--video',      default=DEFAULT_VIDEO,
                        help='Input video path')
    parser.add_argument('--yolo',       default=DEFAULT_YOLO_PATH,
                        help='Path to YOLO best_2.pt checkpoint')
    parser.add_argument('--tracknet',   default=DEFAULT_TRACKNET_PATH,
                        help='Path to TrackNetV5 best_tracknetv5.pth checkpoint')
    parser.add_argument('--output',     default=DEFAULT_OUTPUT,
                        help='Output video path (default: auto-named _combined.mp4)')
    parser.add_argument('--device',     default=DEFAULT_DEVICE,
                        choices=['cuda', 'cpu'],
                        help='Inference device (default: cuda)')
    parser.add_argument('--yolo-conf',  type=float, default=YOLO_CONF,
                        help=f'YOLO confidence threshold (default: {YOLO_CONF})')
    parser.add_argument('--yolo-iou',   type=float, default=YOLO_IOU,
                        help=f'YOLO NMS IoU threshold (default: {YOLO_IOU})')
    parser.add_argument('--tn-thresh',  type=float, default=TN_THRESH,
                        help=f'TrackNetV5 heatmap threshold (default: {TN_THRESH})')
    parser.add_argument('--trail',      type=int,   default=TRAIL_LEN,
                        help=f'Trail length in frames (default: {TRAIL_LEN})')
    parser.add_argument('--smooth',     type=int,   default=SMOOTH_WINDOW,
                        help=f'Median filter window for smoothing (default: {SMOOTH_WINDOW})')
    parser.add_argument('--agree-dist', type=int,   default=FUSION_AGREE_DIST_PX,
                        help=f'Max px distance for BOTH label (default: {FUSION_AGREE_DIST_PX})')
    parser.add_argument('--max-frames', type=int,   default=None,
                        help='Process only first N frames (useful for quick tests)')
    args = parser.parse_args()

    run_combined(
        video_path    = args.video,
        yolo_path     = args.yolo,
        tracknet_path = args.tracknet,
        output_path   = args.output,
        device        = args.device,
        yolo_conf     = args.yolo_conf,
        yolo_iou      = args.yolo_iou,
        yolo_imgsz    = YOLO_IMGSZ,
        tn_thresh     = args.tn_thresh,
        trail_len     = args.trail,
        smooth_window = args.smooth,
        agree_dist    = args.agree_dist,
        max_frames    = args.max_frames,
    )
