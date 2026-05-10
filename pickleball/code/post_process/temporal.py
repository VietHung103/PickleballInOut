import numpy as np


class TemporalSmoother:
    """
    Exponential Moving Average smoother for court keypoints across frames.

    Far-court KPs (0-5) use stronger smoothing (lower alpha) because they
    have higher YOLO prediction error. Near-court KPs (6-11) are already
    accurate so less smoothing is applied.

    Jump detection: if a KP moves more than max_jump_px between frames it
    is likely a bad detection — the previous smoothed value is kept instead.
    """

    def __init__(
        self,
        alpha_far=0.3,
        alpha_near=0.5,
        max_jump_far=25,
        max_jump_near=15,
        num_keypoints=12,
    ):
        self.alpha_far    = alpha_far
        self.alpha_near   = alpha_near
        self.max_jump_far  = max_jump_far
        self.max_jump_near = max_jump_near
        self.num_keypoints = num_keypoints
        self._smoothed = None  # (num_keypoints, 2) float array; None until first frame

    def update(self, refined_keypoints):
        """
        Update smoothed positions with this frame's refined keypoints.
        Returns smoothed_keypoints in the same dict-list format as the input.
        """
        curr = np.full((self.num_keypoints, 2), np.nan)
        for kp in refined_keypoints:
            curr[kp['id']] = [kp['x'], kp['y']]

        if self._smoothed is None:
            self._smoothed = curr.copy()
            return [dict(kp) for kp in refined_keypoints]

        for i in range(self.num_keypoints):
            if np.isnan(curr[i]).any():
                continue
            if np.isnan(self._smoothed[i]).any():
                self._smoothed[i] = curr[i]
                continue

            is_far    = i < 6
            alpha     = self.alpha_far    if is_far else self.alpha_near
            max_jump  = self.max_jump_far if is_far else self.max_jump_near

            if np.linalg.norm(curr[i] - self._smoothed[i]) > max_jump:
                pass  # bad detection frame — keep previous smoothed value
            else:
                self._smoothed[i] = alpha * curr[i] + (1 - alpha) * self._smoothed[i]

        smoothed_kps = []
        for kp in refined_keypoints:
            i = kp['id']
            if not np.isnan(self._smoothed[i]).any():
                smoothed_kps.append({
                    **kp,
                    'x': float(self._smoothed[i][0]),
                    'y': float(self._smoothed[i][1]),
                })
            else:
                smoothed_kps.append(dict(kp))

        return smoothed_kps

    def reset(self):
        """Reset smoothing state (call on scene cuts or camera changes)."""
        self._smoothed = None
