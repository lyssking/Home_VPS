import base64
import os
from typing import Optional

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="VPS Localizer Server")

# Enable CORS for WebAR clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LocalizeRequest(BaseModel):
    image: Optional[str] = None
    frame: Optional[str] = None
    image_base64: Optional[str] = None

# Database Loading
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "vps_lidar_db.npz")

if not os.path.exists(DB_PATH):
    print(f"⚠️ Warning: Database not found at {DB_PATH}. Run build_vps_database.py first.")
    db_points3D = np.empty((0, 3), dtype=np.float32)
    db_descriptors = np.empty((0, 32), dtype=np.uint8)
else:
    db = np.load(DB_PATH)
    db_points3D = db["points3D"]
    db_descriptors = db["descriptors"]
    print(f"✅ Loaded VPS Database: {len(db_points3D)} spatial anchor points.")

# High-density ORB configuration for mobile video frames
orb = cv2.ORB_create(
    nfeatures=4000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=15,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=7
)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

@app.get("/")
def health_check():
    return {"status": "online", "anchors_loaded": len(db_points3D)}

@app.post("/api/vps/localize")
async def localize(req: LocalizeRequest):
    img_data = req.image or req.frame or req.image_base64
    if not img_data:
        raise HTTPException(status_code=400, detail="Missing base64 image data.")

    # Strip data URL prefix if present
    if "," in img_data:
        img_data = img_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding error: {str(e)}")

    h, w, _ = frame.shape
    keypoints, descriptors = orb.detectAndCompute(frame, None)

    if descriptors is None or len(keypoints) < 10 or len(db_descriptors) == 0:
        return {"matched": False, "inliers": 0, "status": "INSUFFICIENT_FEATURES"}

    # KNN feature matching
    matches = bf.knnMatch(descriptors, db_descriptors, k=2)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.80 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 6:
        return {"matched": False, "inliers": len(good_matches), "status": "LOW_MATCHES"}

    # Prepare 2D and 3D correspondences
    pts_2d = np.float32([keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_3d = np.float32([db_points3D[m.trainIdx] for m in good_matches]).reshape(-1, 1, 3)

    # Approximate camera intrinsics (~60 deg FOV)
    fx = fy = (w / 2.0) / np.tan(np.radians(60.0 / 2.0))
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Solve Perspective-n-Point with RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d,
        pts_2d,
        camera_matrix,
        dist_coeffs,
        reprojectionError=12.0,
        iterationsCount=1000,
        confidence=0.99,
        flags=cv2.SOLVEPNP_SQPNP
    )

    inlier_count = len(inliers) if inliers is not None else 0

    if success and inlier_count >= 6:
        # Convert rotation vector to 3x3 rotation matrix
        R_mat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R_mat[0, 0] ** 2 + R_mat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            rx = np.arctan2(R_mat[2, 1], R_mat[2, 2])
            ry = np.arctan2(-R_mat[2, 0], sy)
            rz = np.arctan2(R_mat[1, 0], R_mat[0, 0])
        else:
            rx = np.arctan2(-R_mat[1, 2], R_mat[1, 1])
            ry = np.arctan2(-R_mat[2, 0], sy)
            rz = 0

        # Calculate estimated camera world position: C = -R^T * t
        cam_world_pos = -np.matrix(R_mat).T * np.matrix(tvec)

        return {
            "matched": True,
            "inliers": inlier_count,
            "status": "LOCKED",
            "pose": {
                "position": {
                    "x": float(cam_world_pos[0, 0]),
                    "y": float(cam_world_pos[1, 0]),
                    "z": float(cam_world_pos[2, 0])
                },
                "rotation": {
                    "x": float(rx),
                    "y": float(ry),
                    "z": float(rz)
                }
            }
        }

    return {"matched": False, "inliers": inlier_count, "status": "SEARCHING"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)