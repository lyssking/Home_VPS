import cv2
import os
import shutil

VIDEO_PATH = "room_tour.mp4"
OUTPUT_DIR = "mapping_images"

# 1. Check if file exists
if not os.path.exists(VIDEO_PATH):
    print(f"❌ Error: '{VIDEO_PATH}' not found in current folder.")
    print(f"Make sure your video is named '{VIDEO_PATH}' and located at: {os.path.abspath('.')}")
    exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Error: OpenCV could not open '{VIDEO_PATH}'. Check file permissions or video codec.")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"📹 Video Loaded: {width}x{height} @ {fps:.1f} FPS | Total frames: {total_frames}")

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample 1 frame every 0.35 seconds
frame_interval = max(1, int(fps * 0.35))

saved = 0
count = 0
scores = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores.append(score)

        # Lowered threshold to 20.0 to accept indoor lighting & phone compression
        if score > 20.0:
            out_name = os.path.join(OUTPUT_DIR, f"keyframe_{saved:04d}.jpg")
            cv2.imwrite(out_name, frame)
            saved += 1

    count += 1

cap.release()

if scores:
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    print(f"📊 Sharpness Analysis -> Avg: {avg_score:.1f} | Max: {max_score:.1f}")

# Fallback: if lighting was very soft and score was under 20, extract by time interval directly
if saved == 0 and total_frames > 0:
    print("⚠️ All frames were below threshold. Extracting standard interval frames...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            out_name = os.path.join(OUTPUT_DIR, f"keyframe_{saved:04d}.jpg")
            cv2.imwrite(out_name, frame)
            saved += 1
        count += 1
    cap.release()

print(f"🎉 Successfully saved {saved} reference keyframes to {OUTPUT_DIR}/")