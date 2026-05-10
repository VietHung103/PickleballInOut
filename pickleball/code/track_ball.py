"""
TrackNetV5 — Local PC Inference Script
=======================================
Run TrackNetV5 on a real pickleball video on your local machine.

SETUP (run once in terminal):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install opencv-python scipy numpy

  # If you don't have a GPU, use CPU version instead:
  pip install torch torchvision torchaudio

USAGE:
  python inference_local.py --video path/to/video.mp4 --ckpt path/to/best_tracknetv5.pth

  # With all options:
  python inference_local.py \
      --video  input.mp4 \
      --ckpt   best_tracknetv5.pth \
      --output output_annotated.mp4 \
      --thresh 0.5 \
      --trail  8 \
      --device cuda        # or 'cpu' if no GPU
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from scipy.signal import medfilt
from collections import deque

# ─────────────────────────────────────────────────────────
# CONFIG — change defaults here if you don't use CLI args
# ─────────────────────────────────────────────────────────
DEFAULT_VIDEO  = r'C:\AI\pickleball\data\game_6\clip_1\0505.mp4'
DEFAULT_CKPT   = r'C:\AI\pickleball\model\tracknet_plot\best_tracknetv5.pth'
DEFAULT_OUTPUT = None          # None = auto-name next to input video
DEFAULT_THRESH = 0.5
DEFAULT_TRAIL  = 8
DEFAULT_DEVICE = 'cuda'        # 'cuda' or 'cpu'

# Max ball travel between consecutive frames (pixels at original res).
# 150mph ball at 60fps ≈ 90px/frame; 2× margin for fast shots.
MAX_TRAVEL_PX  = 200

# Model input resolution (must match training)
IMG_W, IMG_H = 512, 288

# ─────────────────────────────────────────────────────────
# MODEL DEFINITION (copy of training model — must match exactly)
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
        max_tokens       = (IMG_H // patch) * (IMG_W // patch)
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
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_W, IMG_H))
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0  # 3×H×W


def extract_ball_center(heatmap_tensor, threshold):
    hm     = heatmap_tensor.squeeze().cpu().numpy()
    binary = (hm >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)  # largest blob only

    # Reject blobs that are too large (clothing/scoreboard) or too small (noise).
    # At 512×288, ball radius is roughly 2–15px → area 12–700px².
    area = cv2.contourArea(c)
    if area < 4 or area > 1500:
        return None

    M = cv2.moments(c)
    if M['m00'] == 0:
        return None
    return (M['m10'] / M['m00'], M['m01'] / M['m00'])  # model-space cx, cy


def model_to_orig(cx, cy, orig_w, orig_h):
    return (int(cx * orig_w / IMG_W), int(cy * orig_h / IMG_H))


def smooth_coords(vals, window=5):
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
    """Draw ghost-ball circles that fade from old (small, transparent) to new (large, opaque)."""
    n = len(trail)
    for i, pt in enumerate(trail):
        if pt is None:
            continue
        t       = (i + 1) / n          # 0→1, oldest→newest
        radius  = max(4, int(10 * t))
        opacity = 0.25 + 0.70 * t      # 0.25→0.95
        # Yellow-green in BGR: fades from dim olive to bright yellow-green
        color = (0, int(180 + 75 * t), int(160 + 95 * t))
        overlay = frame.copy()
        cv2.circle(overlay, pt, radius, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)


def draw_detection(frame, cx, cy):
    # Subtle ring to mark the current detected position
    cv2.circle(frame, (cx, cy), 13, (0, 255, 255), 2, cv2.LINE_AA)


def draw_hud(frame, frame_idx, total, detected):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 38), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    status = "BALL DETECTED" if detected else "NO BALL"
    color  = (0, 255, 128)   if detected else (0, 80, 255)
    cv2.putText(frame,
                f"Frame {frame_idx+1}/{total}  |  TrackNetV5 Pickleball  |  {status}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main(args):
    # ── Device ──────────────────────────────────────────
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available — falling back to CPU.")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Load model ───────────────────────────────────────
    print(f"\nLoading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = TrackNetV5().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"✅ Model loaded | Best F1: {ckpt.get('best_f1', 'N/A')}")

    # ── Open video ───────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video  : {args.video}")
    print(f"Size   : {orig_w}×{orig_h} @ {fps:.1f} fps | {total_fr} frames")

    # ── Output path ─────────────────────────────────────
    if args.output is None:
        p            = Path(args.video)
        args.output  = str(p.parent / (p.stem + '_tracknetv5.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (orig_w, orig_h))
    print(f"Output : {args.output}\n")

    # ── Read first two frames ────────────────────────────
    ret, f0 = cap.read()
    ret, f1 = cap.read()
    if not ret:
        raise RuntimeError("Video has fewer than 2 frames.")

    t0 = preprocess_frame(f0)
    t1 = preprocess_frame(f1)

    all_frames = [f0, f1]
    raw_xs, raw_ys = [], []
    results        = []   # None or (cx_orig, cy_orig) per frame
    frame_idx      = 1
    last_cx        = None  # for travel-distance filter
    last_cy        = None

    # ── Inference loop ───────────────────────────────────
    print("Running inference...")
    with torch.no_grad():
        while True:
            ret, f2 = cap.read()
            if not ret:
                break

            t2      = preprocess_frame(f2)
            triplet = torch.cat([t0, t1, t2], dim=0).unsqueeze(0).to(device)
            heatmap = model(triplet)   # 1×1×H×W, predicts for middle frame t1

            center  = extract_ball_center(heatmap, args.thresh)
            if center is not None:
                cx_orig, cy_orig = model_to_orig(*center, orig_w, orig_h)
                # Travel distance filter: reject detections that jump impossibly far
                if last_cx is not None:
                    dist = ((cx_orig - last_cx) ** 2 + (cy_orig - last_cy) ** 2) ** 0.5
                    if dist > MAX_TRAVEL_PX:
                        center = None   # treat as no detection this frame
                if center is not None:
                    raw_xs.append(float(cx_orig))
                    raw_ys.append(float(cy_orig))
                    results.append((cx_orig, cy_orig))
                    last_cx, last_cy = cx_orig, cy_orig
                else:
                    raw_xs.append(np.nan)
                    raw_ys.append(np.nan)
                    results.append(None)
            else:
                raw_xs.append(np.nan)
                raw_ys.append(np.nan)
                results.append(None)
                last_cx, last_cy = None, None

            t0, t1 = t1, t2
            f0, f1 = f1, f2
            all_frames.append(f2)
            frame_idx += 1

            if frame_idx % 100 == 0:
                det = sum(r is not None for r in results)
                pct = det / frame_idx * 100
                print(f"  {frame_idx}/{total_fr} frames | "
                      f"Detections: {det} ({pct:.1f}%)", end='\r')

    cap.release()
    det_total = sum(r is not None for r in results)
    print(f"\n✅ Inference done: {det_total}/{len(results)} detections "
          f"({det_total/max(len(results),1)*100:.1f}%)")

    # ── Smooth trajectory ────────────────────────────────
    smooth_xs = smooth_coords(raw_xs, window=args.smooth)
    smooth_ys = smooth_coords(raw_ys, window=args.smooth)

    MAX_GAP   = 3
    smoothed  = []
    for i, r in enumerate(results):
        if r is not None:
            smoothed.append((int(smooth_xs[i]), int(smooth_ys[i])))
        else:
            lo  = max(0, i - MAX_GAP)
            hi  = min(len(results), i + MAX_GAP + 1)
            nearby = any(x is not None for x in results[lo:i] + results[i+1:hi])
            if nearby and not np.isnan(raw_xs[i]):
                smoothed.append((int(smooth_xs[i]), int(smooth_ys[i])))
            else:
                smoothed.append(None)

    # ── Render output video ──────────────────────────────
    print("Rendering annotated video...")
    all_results = [None] + smoothed + [None]
    n_write     = min(len(all_frames), len(all_results))
    trail       = deque(maxlen=args.trail)

    for i in range(n_write):
        frame  = all_frames[i].copy()
        result = all_results[i]
        trail.append(result)

        draw_trail(frame, list(trail))
        if result is not None:
            draw_detection(frame, *result)
        draw_hud(frame, i, n_write, result is not None)
        writer.write(frame)

        if (i + 1) % 100 == 0:
            print(f"  Written {i+1}/{n_write} frames", end='\r')

    writer.release()
    print(f"\n✅ Done! Output saved to: {args.output}")


# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TrackNetV5 Pickleball Inference'
    )
    parser.add_argument('--video',  default=DEFAULT_VIDEO,
                        help='Path to input video file')
    parser.add_argument('--ckpt',   default=DEFAULT_CKPT,
                        help='Path to best_tracknetv5.pth checkpoint')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help='Path to output video (default: auto-named)')
    parser.add_argument('--thresh', type=float, default=DEFAULT_THRESH,
                        help='Heatmap detection threshold (default: 0.5)')
    parser.add_argument('--trail',  type=int,   default=DEFAULT_TRAIL,
                        help='Trail length in frames (default: 8)')
    parser.add_argument('--smooth', type=int,   default=5,
                        help='Median filter window for smoothing (default: 5)')
    parser.add_argument('--device', default=DEFAULT_DEVICE,
                        choices=['cuda', 'cpu'],
                        help='Device to run on (default: cuda)')
    args = parser.parse_args()
    main(args)