import cv2
import os

def extract_keyframes(video_path, output_folder, frame_interval=15, blur_threshold=80.0):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only check every Nth frame
        if frame_count % frame_interval == 0:
            # Check for motion blur using the Laplacian variance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            if variance > blur_threshold:
                filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} sharp keyframes into '{output_folder}'.")

# Example usage:
extract_keyframes("room_walkthorugh.mp4", "mapping_images", frame_interval=15)