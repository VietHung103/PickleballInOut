import cv2
import os

video_path = r"C:\AI\pickleball\data\game_4\Clip_5\0409(1).mp4"
output_folder = r"C:\AI\pickleball\data\game\frames"

target_fps = None  # None means keep original FPS

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

video_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Raw video FPS: {video_fps:.3f}")
print(f"Total frames: {total_frames}")

if target_fps is None or target_fps >= video_fps:
      target_fps = video_fps

step = video_fps / target_fps

frame_id = 0
save_id = 0
next_save = 0.0

while True:
      ret, frame = cap.read()
      if not ret:
          break

      if frame_id >= round(next_save):
          filename = os.path.join(output_folder, f"{save_id:06d}.jpg")
          cv2.imwrite(filename, frame)
          save_id += 1
          next_save += step

      frame_id += 1

cap.release()
print(f"Extracted {save_id} frames at target FPS {target_fps:.3f}")