# Court Keypoint Post-Processing — Pickleball In/Out System

## Project Context

This is the post-processing module (M2b) for a pickleball in/out decision system.

**Full pipeline:**
```
Video frame
  → YOLO Pose (yolov8s-pose fine-tuned)     ← already built
      → 12 court keypoints (approximate)
  → Post-Processing (THIS MODULE)            ← build this
      → 12 court keypoints (pixel-precise)
  → Homography H                             ← already built
      → bounce point in real-world cm
  → In/Out decision                          ← already built
```

**Problem being solved:**
YOLO Pose predicts far-court keypoints (KP 0-5) with ~15-30px error due to perspective
foreshortening — far court lines appear very thin. Near-court keypoints (KP 6-11) are
already accurate (~3-5px error). Post-processing uses local Hough refinement to snap
each keypoint to the exact line intersection pixel.

---

## Court Keypoint Schema

```
KP layout (camera is on foreground side, looking at far court):

KP0─────────KP1──────────KP2        ← Far baseline       (L_BL_BG, M_BL_BG, R_BL_BG)
│           │            │
KP5─────────KP4──────────KP3        ← Far kitchen line   (L_KL_BG, M_KL_BG, R_KL_BG)
│                        │
│           NET           │
│                        │
KP6─────────KP7──────────KP8        ← Near kitchen line  (L_KL_FG, M_KL_FG, R_KL_FG)
│           │            │
KP11────────KP10─────────KP9        ← Near baseline      (L_BL_FG, M_BL_FG, R_BL_FG)
```

**Keypoint names and real-world coordinates (cm):**
```
KP ID | Name      | X_cm | Y_cm  | Notes
------|-----------|------|-------|---------------------------
  0   | L_BL_BG   |   0  | 1372  | Far left baseline corner
  1   | M_BL_BG   | 305  | 1372  | Far center baseline
  2   | R_BL_BG   | 610  | 1372  | Far right baseline corner
  3   | R_KL_BG   | 610  |  915  | Far right kitchen corner
  4   | M_KL_BG   | 305  |  915  | Far center kitchen (net)
  5   | L_KL_BG   |   0  |  915  | Far left kitchen corner
  6   | L_KL_FG   |   0  |  457  | Near left kitchen corner
  7   | M_KL_FG   | 305  |  457  | Near center kitchen
  8   | R_KL_FG   | 610  |  457  | Near right kitchen corner
  9   | R_BL_FG   | 610  |    0  | Near right baseline corner
 10   | M_BL_FG   | 305  |    0  | Near center baseline
 11   | L_BL_FG   |   0  |    0  | Near left baseline corner (origin)
```

**Court standard dimensions:**
- Full court width: 610 cm
- Full court length: 1372 cm
- Kitchen line distance from baseline: 457 cm
- Net is at 686 cm (center)

---

## Input / Output Contract

### Input
```python
# frame: numpy array (H, W, 3) BGR — original video frame (640x360 typical)
# raw_keypoints: list of dicts from YOLO Pose detector
[
    {
        'id':   0,              # keypoint index 0-11
        'name': 'L_BL_BG',     # keypoint name
        'x':    304.2,          # pixel x in original frame
        'y':    126.8,          # pixel y in original frame
        'conf': 0.91            # YOLO confidence 0-1
    },
    # ... 11 more
]
```

### Output
```python
# refined_keypoints: same format with improved x, y
[
    {
        'id':              0,
        'name':            'L_BL_BG',
        'x':               298.7,       # refined pixel x
        'y':               122.3,       # refined pixel y
        'conf':            0.91,
        'refined':         True,        # was post-processing applied?
        'refinement_delta': 8.3,        # how many pixels it moved
        'refinement_status': 'success'  # see status codes below
    },
    # ...
]

# refinement_log: dict for debugging
{
    0: 'refined_delta=8.3px',
    1: 'refined_delta=4.1px',
    2: 'hough_failed',
    3: 'rejected_not_white',
    6: 'skipped',           # near-court, skipped by default
    ...
}
```

**Refinement status codes:**
- `success` — Hough found intersection, verified on white pixel, accepted
- `hough_failed` — Hough found fewer than 2 lines in crop
- `rejected_dist` — intersection found but too far from prediction (bad Hough)
- `rejected_not_white` — intersection not on a white pixel (bad detection)
- `skipped` — KP not in apply_to_kps list
- `empty_crop` — crop region was out of frame bounds

---

## Algorithm — Step by Step

### Step 1: Crop around predicted keypoint
- Crop a square region around each predicted keypoint
- Crop size: **120px** for far-court KPs (0-5), **80px** for near-court KPs (6-11)
- Far court gets larger crop because prediction error is larger
- Clamp crop to frame boundaries

### Step 2: Color preprocessing
Remove court surface color, isolate white lines:

```
HSV thresholds for court surface removal:
  Green court: H=35-85,  S=40-255, V=40-255
  Blue court:  H=90-130, S=40-255, V=40-255

HSV thresholds for white line isolation:
  White lines: H=0-180,  S=0-50,   V=170-255

Final mask = white_mask AND NOT court_mask
```

Apply morphological cleanup:
- Dilate kernel (3,3) iterations=1 — connect broken line segments
- Erode  kernel (3,3) iterations=1 — remove noise

### Step 3: Line thickening for far court
Far court lines appear ~1-2px wide due to perspective. Before Hough:
- For KP 0,1,2 (far baseline): dilate with kernel (5,5) iterations=2
- For KP 3,4,5 (far kitchen):  dilate with kernel (4,4) iterations=2
- Near court KPs: no thickening needed

### Step 4: Zhang-Suen thinning
After thickening, thin back to single-pixel-wide lines using Zhang-Suen algorithm.
This gives Hough a cleaner signal.
Use: `cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)`
Requires: opencv-contrib-python

If opencv-contrib not available: skip this step, Hough still works but less precisely.

### Step 5: Hough line detection
```
cv2.HoughLines parameters:
  rho=1
  theta=np.pi/180
  threshold: 10 for far-court KPs, 15 for near-court KPs
  (lower threshold for far court because lines are faint)

Take top 10 detected lines maximum.
```

Separate lines into horizontal and vertical:
- Horizontal: theta between 45° and 135°
- Vertical:   theta outside that range (0°-45° or 135°-180°)

Need at least 1 horizontal AND 1 vertical line. If not → hough_failed.

### Step 6: Find intersection
Use best horizontal + best vertical line.
Solve the linear system:
```
[cos(theta1)  sin(theta1)] [x]   [rho1]
[cos(theta2)  sin(theta2)] [y] = [rho2]
```
Use numpy.linalg.solve. If singular matrix → hough_failed.

Convert from crop-local coordinates to frame coordinates:
```
x_frame = x_local + crop_x1
y_frame = y_local + crop_y1
```

### Step 7: Sanity check — distance gate
If refined point is more than `crop_size * 0.8` pixels from original prediction → reject.
This prevents bad Hough results from moving the keypoint to a completely wrong location.

### Step 8: White pixel verification ⭐ Key step
Check that the refined intersection point lands on a white pixel in the ORIGINAL frame
(not the preprocessed crop). Check a 4px radius neighborhood.

If no white pixel found within radius → rejected_not_white → keep original prediction.

This is the critical verification step from the ML6 paper that prevents post-processing
from making things worse.

### Step 9: Accept or reject
If all checks pass → update x, y with refined coordinates, set refined=True.
Otherwise → keep original YOLO Pose prediction unchanged.

---

## Special Cases per Keypoint

```
KP 0 (L_BL_BG): Far left corner — two lines meet at near-90° angle. 
                 Good for Hough. Apply full pipeline.

KP 1 (M_BL_BG): Far center — T-intersection (3 lines meet).
                 Pick the two most prominent lines. May have ambiguity.

KP 2 (R_BL_BG): Far right corner — mirror of KP 0. Good for Hough.

KP 3 (R_KL_BG): Far right kitchen corner — often partially occluded by net post.
                 If conf < 0.5 from YOLO → skip post-processing entirely.

KP 4 (M_KL_BG): Far center kitchen — net post area. Very unreliable.
                 SKIP post-processing for this keypoint always.
                 Use temporal smoothing fallback instead.

KP 5 (L_KL_BG): Far left kitchen corner — mirror of KP 3.
                 If conf < 0.5 from YOLO → skip post-processing.

KP 6 (L_KL_FG): Near left kitchen — usually very accurate from YOLO.
                 Only apply post-processing if conf < 0.7.

KP 7 (M_KL_FG): Near center kitchen — T-intersection near net.
                 Only apply if conf < 0.7.

KP 8 (R_KL_FG): Near right kitchen — mirror of KP 6.
                 Only apply if conf < 0.7.

KP 9  (R_BL_FG): Near right baseline corner — very accurate, skip always.
KP 10 (M_BL_FG): Near center baseline — very accurate, skip always.
KP 11 (L_BL_FG): Near left baseline corner — very accurate, skip always.
```

**Summary of apply_to_kps by default:**
```python
ALWAYS_REFINE  = [0, 1, 2]           # far baseline — highest error
MAYBE_REFINE   = [3, 5, 6, 7, 8]    # refine only if conf < threshold
NEVER_REFINE   = [4, 9, 10, 11]     # net post or very accurate near-court
```

---

## Temporal Smoothing

Apply AFTER post-processing, across frames. Reduces jitter in homography matrix.

```
Algorithm: Exponential Moving Average (EMA)

smoothed[t] = alpha * refined[t] + (1 - alpha) * smoothed[t-1]

alpha = 0.3 for far-court KPs (0-5)   — more smoothing, higher error
alpha = 0.5 for near-court KPs (6-11) — less smoothing, already accurate

Jump detection:
  If |refined[t] - smoothed[t-1]| > MAX_JUMP_PX → ignore refined[t]
  Use smoothed[t-1] instead (bad detection frame)

MAX_JUMP_PX = 25px for far-court
MAX_JUMP_PX = 15px for near-court
```

Initialize smoothed values from first successful detection frame.

---

## Homography Quality Verification

After computing H from refined keypoints, verify it before using for in/out decision.

```
Verification steps:
1. Warp reference court template onto frame using H
2. Compare warped court lines with white pixel mask of frame
3. Count overlap pixels (hits) and non-overlap (misses)
4. overlap_score = hits / (hits + misses)

Thresholds:
  overlap_score > 0.7  → HIGH confidence   → make in/out call
  overlap_score > 0.4  → MEDIUM confidence → make call but flag as uncertain
  overlap_score < 0.4  → LOW confidence    → skip frame, use last good H

Reference court template:
  640x360 pixel image with white lines on black background
  Representing standard pickleball court proportions
  Lines are 3px wide
```

---

## File Structure to Build

```
post_process/
├── __init__.py
├── refiner.py          ← main class CourtKeypointRefiner
├── hough_utils.py      ← Hough detection helpers
├── color_utils.py      ← HSV preprocessing for green/blue courts
├── temporal.py         ← TemporalSmoother class
├── homography_verify.py← HomographyVerifier class
├── visualize.py        ← debug visualization tools
└── test_postprocess.py ← unit tests with sample frames
```

---

## Class Interface to Implement

```python
class CourtKeypointRefiner:
    def __init__(
        self,
        court_color='green',        # 'green' or 'blue'
        crop_size_far=120,          # crop half-size for far KPs
        crop_size_near=80,          # crop half-size for near KPs
        white_verify_radius=4,      # pixels for white pixel check
        max_shift_ratio=0.8,        # max shift as fraction of crop_size
        enable_thinning=True,       # requires opencv-contrib
        verbose=False
    ):
        ...

    def refine(self, frame_bgr, raw_keypoints):
        """
        Main entry point.
        Returns: (refined_keypoints, refinement_log)
        """
        ...

    def _preprocess(self, crop, is_far):
        """Color masking + thickening + thinning."""
        ...

    def _find_intersection(self, mask, is_far):
        """Hough + line separation + intersection math."""
        ...

    def _verify_white(self, frame, x, y):
        """White pixel verification on original frame."""
        ...


class TemporalSmoother:
    def __init__(
        self,
        alpha_far=0.3,
        alpha_near=0.5,
        max_jump_far=25,
        max_jump_near=15,
        num_keypoints=12
    ):
        ...

    def update(self, refined_keypoints):
        """
        Update smoothed positions with new frame's refined keypoints.
        Returns: smoothed_keypoints (same format as refined_keypoints)
        """
        ...

    def reset(self):
        """Reset smoothing state (e.g. scene cut detected)."""
        ...


class HomographyVerifier:
    def __init__(
        self,
        court_width_cm=610,
        court_length_cm=1372,
        high_conf_threshold=0.7,
        low_conf_threshold=0.4
    ):
        ...

    def verify(self, frame_bgr, H):
        """
        Warp reference court onto frame, compute overlap score.
        Returns: (score, confidence_tier)
        confidence_tier: 'HIGH', 'MEDIUM', 'LOW'
        """
        ...

    def _build_reference_template(self, width, height):
        """Build white-lines-on-black reference court image."""
        ...
```

---

## Integration with Existing Pipeline

```python
# In your inference script (replacing direct YOLO output usage):

from post_process.refiner import CourtKeypointRefiner
from post_process.temporal import TemporalSmoother
from post_process.homography_verify import HomographyVerifier

# Initialize (once)
refiner   = CourtKeypointRefiner(court_color='green', verbose=False)
smoother  = TemporalSmoother()
verifier  = HomographyVerifier()
detector  = CourtKeypointDetectorYOLO('best.pt')

# Per frame
for frame in video_frames:
    # Step 1: YOLO Pose detection
    raw_kps = detector.predict(frame)

    # Step 2: Post-processing refinement
    refined_kps, log = refiner.refine(frame, raw_kps)

    # Step 3: Temporal smoothing
    smoothed_kps = smoother.update(refined_kps)

    # Step 4: Compute homography
    H, inliers = detector.compute_homography(smoothed_kps)

    # Step 5: Verify homography quality
    score, confidence = verifier.verify(frame, H)

    if confidence == 'LOW':
        continue  # skip this frame for in/out decision

    # Step 6: Map bounce point to court coordinates
    bounce_cm = detector.pixel_to_court_cm(bounce_x, bounce_y, H)

    # Step 7: In/out decision
    decision = is_in_bounds(bounce_cm, tolerance_cm=5.0)
```

---

## Dependencies

```
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0   ← for Zhang-Suen thinning
numpy>=1.21.0
ultralytics>=8.0.0             ← already installed
```

Install:
```bash
pip install opencv-contrib-python numpy
```

Note: If opencv-contrib is unavailable (e.g. Colab conflict), the system
gracefully skips the Zhang-Suen thinning step. All other steps still work.

---

## Testing

Test on these specific scenarios:

1. **Clean outdoor green court, all 12 KPs visible** → expect all far KPs to improve
2. **Player occluding KP 3 or 5** → expect graceful fallback to raw YOLO prediction
3. **Shadow across far baseline** → white pixel verification should catch bad Hough
4. **Indoor blue court** → test court_color='blue' path
5. **Camera jitter between frames** → temporal smoother should suppress it
6. **Scene cut or camera angle change** → smoother.reset() should be triggered

For each test, log:
- refinement_delta per KP (should be 5-30px for far court, near zero for near court)
- refinement_status distribution across frames
- homography overlap_score distribution
- Final MPJPE improvement vs raw YOLO output

---

## Expected Improvement

Based on the ML6 tennis paper results and the observed error pattern:

```
Before post-processing:
  Far court MPJPE  (KP 0-2): ~20-30px in image space
  Near court MPJPE (KP 6-11): ~3-5px in image space

After post-processing:
  Far court MPJPE  (KP 0-2): ~5-10px in image space  (2-4x improvement)
  Near court MPJPE (KP 6-11): ~3-5px in image space  (unchanged)

In court coordinates (after homography):
  Before: ~10-15cm error at far court
  After:  ~2-5cm error at far court
  (pickleball line width = 5cm, so this matters for calls near the line)
```
