# extract_dense_frames.py
import cv2
import os

video_path = "room_walkthorugh.mp4" # path to your room scan video
output_dir = "mapping_images"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_interval = int(fps * 0.5) # 2 frames per second

count = 0
saved = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_interval == 0:
        cv2.imwrite(os.path.join(output_dir, f"frame_{saved:04d}.jpg"), frame)
        saved += 1
    count += 1

cap.release()
print(f"Extracted {saved} dense reference frames to {output_dir}/")