"""
CourtKeypointRefiner — local Hough-based keypoint refinement.

For each far-court keypoint (KP 0-5) the YOLO prediction is snapped to the
exact court line intersection pixel using:
  1. HSV color masking to isolate white lines
  2. Morphological thickening + Zhang-Suen thinning (if opencv-contrib available)
  3. HoughLines to detect the two crossing lines
  4. Linear-system intersection solve
  5. Distance gate + white-pixel verification before accepting

Near-court KPs (6-11) are already accurate from YOLO and are only refined
when their confidence is low.
"""

import cv2
import math
import numpy as np


# ── Refinement policy ──────────────────────────────────────────────────────
# Based on POST_PROCESS.md: far baseline always has highest error.
# Net-post area (KP4) and very-accurate near-court corners (KP9-11) are skipped.
ALWAYS_REFINE = [0, 1, 2]         # far baseline — highest error
MAYBE_REFINE  = [3, 5, 6, 7, 8]  # refine only when confidence is low
NEVER_REFINE  = [4, 9, 10, 11]   # net post / very accurate near-court

# Confidence thresholds for MAYBE_REFINE groups
CONF_THRESH_FAR_KITCHEN  = 0.5   # KP 3, 5 — skip refinement if conf ≥ this
CONF_THRESH_NEAR_KITCHEN = 0.7   # KP 6, 7, 8 — skip refinement if conf ≥ this

KP_NAMES = [
    'L_BL_BG', 'M_BL_BG', 'R_BL_BG',   # 0-2  far baseline
    'R_KL_BG', 'M_KL_BG', 'L_KL_BG',   # 3-5  far kitchen
    'L_KL_FG', 'M_KL_FG', 'R_KL_FG',   # 6-8  near kitchen
    'R_BL_FG', 'M_BL_FG', 'L_BL_FG',   # 9-11 near baseline
]


class CourtKeypointRefiner:
    """
    Refines YOLO Pose court keypoints using local Hough line detection.

    Args:
        court_color:        'green' or 'blue' — controls HSV court-surface mask
        crop_size_far:      half-size of crop window for far-court KPs (px)
        crop_size_near:     half-size of crop window for near-court KPs (px)
        white_verify_radius: radius for white-pixel neighborhood check (px)
        max_shift_ratio:    max allowed shift as fraction of crop_size
        enable_thinning:    use Zhang-Suen thinning if opencv-contrib is present
        verbose:            print per-keypoint refinement results
    """

    def __init__(
        self,
        court_color='green',
        crop_size_far=120,
        crop_size_near=80,
        white_verify_radius=4,
        max_shift_ratio=0.8,
        enable_thinning=True,
        verbose=False,
    ):
        self.court_color          = court_color
        self.crop_size_far        = crop_size_far
        self.crop_size_near       = crop_size_near
        self.white_verify_radius  = white_verify_radius
        self.max_shift_ratio      = max_shift_ratio
        self.verbose              = verbose

        self.thinning_available = False
        if enable_thinning:
            try:
                _ = cv2.ximgproc.thinning
                self.thinning_available = True
            except AttributeError:
                pass

    # ── Public API ────────────────────────────────────────────────────────

    def refine(self, frame_bgr, raw_keypoints):
        """
        Main entry point.

        Args:
            frame_bgr:      numpy (H, W, 3) BGR frame
            raw_keypoints:  list of dicts with keys id, name, x, y, conf

        Returns:
            (refined_keypoints, refinement_log)
            refined_keypoints: same list with updated x/y and extra fields
            refinement_log:    {kp_id: status_string}
        """
        h, w   = frame_bgr.shape[:2]
        refined = []
        log     = {}

        for kp in raw_keypoints:
            kp_id  = kp['id']
            x_pred = kp['x']
            y_pred = kp['y']
            conf   = kp['conf']

            # ── Policy check ──────────────────────────────────────────
            if kp_id in NEVER_REFINE:
                refined.append({**kp, 'refined': False,
                                 'refinement_delta': 0.0,
                                 'refinement_status': 'skipped'})
                log[kp_id] = 'skipped'
                continue

            if kp_id in [3, 5] and conf >= CONF_THRESH_FAR_KITCHEN:
                refined.append({**kp, 'refined': False,
                                 'refinement_delta': 0.0,
                                 'refinement_status': 'skipped'})
                log[kp_id] = 'skipped'
                continue

            if kp_id in [6, 7, 8] and conf >= CONF_THRESH_NEAR_KITCHEN:
                refined.append({**kp, 'refined': False,
                                 'refinement_delta': 0.0,
                                 'refinement_status': 'skipped'})
                log[kp_id] = 'skipped'
                continue

            # ── Crop ──────────────────────────────────────────────────
            is_far    = kp_id < 6
            half      = (self.crop_size_far if is_far else self.crop_size_near) // 2
            cx, cy    = int(round(x_pred)), int(round(y_pred))
            x1 = max(0, cx - half);  x2 = min(w, cx + half)
            y1 = max(0, cy - half);  y2 = min(h, cy + half)

            if (x2 - x1) < 10 or (y2 - y1) < 10:
                refined.append({**kp, 'refined': False,
                                 'refinement_delta': 0.0,
                                 'refinement_status': 'empty_crop'})
                log[kp_id] = 'empty_crop'
                continue

            crop = frame_bgr[y1:y2, x1:x2]

            # ── Preprocess ────────────────────────────────────────────
            mask = self._preprocess(crop, kp_id)

            # ── Find intersection ──────────────────────────────────────
            result = self._find_intersection(mask, is_far)

            if result is None:
                refined.append({**kp, 'refined': False,
                                 'refinement_delta': 0.0,
                                 'refinement_status': 'hough_failed'})
                log[kp_id] = 'hough_failed'
                continue

            x_local, y_local = result
            x_frame = x_local + x1
            y_frame = y_local + y1

            # ── Distance gate ─────────────────────────────────────────
            delta = math.hypot(x_frame - x_pred, y_frame - y_pred)
            crop_size = half * 2
            if delta > crop_size * self.max_shift_ratio:
                refined.append({**kp, 'refined': False,
                                 'refinement_delta': delta,
                                 'refinement_status': 'rejected_dist'})
                log[kp_id] = f'rejected_dist={delta:.1f}px'
                continue

            # ── White pixel verification ───────────────────────────────
            if not self._verify_white(frame_bgr, x_frame, y_frame):
                refined.append({**kp, 'refined': False,
                                 'refinement_delta': delta,
                                 'refinement_status': 'rejected_not_white'})
                log[kp_id] = 'rejected_not_white'
                continue

            # ── Accept ────────────────────────────────────────────────
            if self.verbose:
                print(f"  KP{kp_id} ({KP_NAMES[kp_id]}): "
                      f"({x_pred:.1f},{y_pred:.1f}) → ({x_frame:.1f},{y_frame:.1f})  "
                      f"Δ={delta:.1f}px")

            refined.append({
                **kp,
                'x':                 float(x_frame),
                'y':                 float(y_frame),
                'refined':           True,
                'refinement_delta':  delta,
                'refinement_status': 'success',
            })
            log[kp_id] = f'refined_delta={delta:.1f}px'

        return refined, log

    # ── Private helpers ───────────────────────────────────────────────────

    def _preprocess(self, crop, kp_id):
        """Color masking → morphological cleanup → optional thickening → optional thinning."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Remove court surface colour
        if self.court_color == 'green':
            court_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        else:  # blue
            court_mask = cv2.inRange(hsv, (90, 40, 40), (130, 255, 255))

        # Isolate white lines
        white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 50, 255))
        mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(court_mask))

        # Basic morphological cleanup
        k3 = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, k3, iterations=1)
        mask = cv2.erode(mask, k3, iterations=1)

        # Thicken far-court lines (appear 1-2 px wide due to perspective)
        if kp_id in [0, 1, 2]:                            # far baseline
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        elif kp_id in [3, 5]:                             # far kitchen corners
            mask = cv2.dilate(mask, np.ones((4, 4), np.uint8), iterations=2)

        # Zhang-Suen thinning for a cleaner Hough input
        if self.thinning_available:
            mask = cv2.ximgproc.thinning(
                mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

        return mask

    def _find_intersection(self, mask, is_far):
        """
        Run HoughLines, separate H/V, solve the 2-line intersection.
        Returns (x_local, y_local) in crop coordinates, or None on failure.
        """
        threshold = 10 if is_far else 15
        lines = cv2.HoughLines(mask, 1, np.pi / 180, threshold)

        if lines is None:
            return None

        lines = lines[:10]  # top 10 strongest lines

        horizontal, vertical = [], []
        for line in lines:
            rho, theta = float(line[0][0]), float(line[0][1])
            deg = math.degrees(theta)
            if 45 <= deg <= 135:
                horizontal.append((rho, theta))
            else:
                vertical.append((rho, theta))

        if not horizontal or not vertical:
            return None

        rho1, theta1 = horizontal[0]
        rho2, theta2 = vertical[0]

        A = np.array([
            [math.cos(theta1), math.sin(theta1)],
            [math.cos(theta2), math.sin(theta2)],
        ])
        b = np.array([rho1, rho2])

        try:
            x, y = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

        h, w = mask.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return None

        return float(x), float(y)

    def _verify_white(self, frame, x, y):
        """Return True if there is at least one white pixel within white_verify_radius of (x, y)."""
        h, w = frame.shape[:2]
        r    = self.white_verify_radius
        x1   = max(0, int(x) - r);  x2 = min(w, int(x) + r + 1)
        y1   = max(0, int(y) - r);  y2 = min(h, int(y) + r + 1)
        if x2 <= x1 or y2 <= y1:
            return False
        patch     = frame[y1:y2, x1:x2]
        hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        white     = cv2.inRange(hsv_patch, (0, 0, 170), (180, 50, 255))
        return bool(white.any())
