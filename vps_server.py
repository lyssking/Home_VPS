import os
import io
import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from lightglue import SuperPoint

app = FastAPI(title="SuperPoint WebXR VPS Server")

# Allow cross-origin requests from WebAR clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LocalizeRequest(BaseModel):
    image: Optional[str] = None

# 1. Initialize SuperPoint Neural Extractor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Initializing SuperPoint on device: {device}")
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)

def extract_superpoint(img_bgr):
    """Extracts 2D keypoints and 256-dim float32 descriptors from a BGR frame."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        res = extractor.extract(tensor.to(device))
    return res['keypoints'][0].cpu().numpy(), res['descriptors'][0].cpu().numpy()

# 2. Load 3D Point Database
DB_PATH = "vps_lidar_db.npz"
db_points3D = []
db_descriptors = []

if os.path.exists(DB_PATH):
    db = np.load(DB_PATH)
    db_points3D = np.array(db["points3D"], dtype=np.float32)
    db_descriptors = np.array(db["descriptors"], dtype=np.float32)
    print(f"✅ Loaded {len(db_points3D)} spatial anchors from {DB_PATH}")
else:
    print(f"⚠️ Warning: {DB_PATH} not found. Run build_vps_database.py first.")

# 3. Setup Fast FLANN Matcher for Float32 SuperPoint Descriptors
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

@app.get("/")
def health_check():
    return {
        "status": "online",
        "anchors_loaded": len(db_points3D),
        "device": str(device)
    }

@app.post("/api/vps/localize")
async def localize(req: LocalizeRequest):
    if not req.image:
        return {"matched": False, "inliers": 0, "status": "NO_IMAGE"}

    if len(db_descriptors) == 0:
        return {"matched": False, "inliers": 0, "status": "EMPTY_DATABASE"}

    # Decode Base64 JPEG frame from WebAR client
    img_data = req.image.split(",", 1)[1] if "," in req.image else req.image
    img_bytes = base64.b64decode(img_data)
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        return {"matched": False, "inliers": 0, "status": "DECODE_ERROR"}

    h, w, _ = frame.shape
    kps, descs = extract_superpoint(frame)

    if len(kps) < 8:
        return {"matched": False, "inliers": 0, "status": "INSUFFICIENT_FEATURES"}

    # Match query descriptors against the pre-indexed 3D anchors
    matches = flann.knnMatch(descs, db_descriptors, k=2)
    
    # Lowe's ratio test for SuperPoint
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.82 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 6:
        return {"matched": False, "inliers": len(good_matches), "status": "LOW_MATCHES"}

    pts_2d = np.float32([kps[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
    pts_3d = np.float32([db_points3D[m.trainIdx] for m in good_matches]).reshape(-1, 1, 3)

    # Approximate mobile camera intrinsics (60-degree FOV)
    fx = fy = (w / 2.0) / np.tan(np.radians(60.0 / 2.0))
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)

    # Solve Perspective-n-Point via RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, 
        pts_2d, 
        camera_matrix, 
        np.zeros((4, 1), dtype=np.float32),
        reprojectionError=10.0, 
        iterationsCount=1000, 
        confidence=0.99, 
        flags=cv2.SOLVEPNP_SQPNP
    )

    inlier_count = len(inliers) if inliers is not None else 0

    if success and inlier_count >= 6:
        # Convert Rodrigues vector to 3x3 rotation matrix
        R_mat, _ = cv2.Rodrigues(rvec)
        
        # Extract Euler angles (XYZ order)
        sy = np.sqrt(R_mat[0, 0]**2 + R_mat[1, 0]**2)
        singular = sy < 1e-6

        if not singular:
            rx = np.arctan2(R_mat[2, 1], R_mat[2, 2])
            ry = np.arctan2(-R_mat[2, 0], sy)
            rz = np.arctan2(R_mat[1, 0], R_mat[0, 0])
        else:
            rx = np.arctan2(-R_mat[1, 2], R_mat[1, 1])
            ry = np.arctan2(-R_mat[2, 0], sy)
            rz = 0.0

        # Calculate camera center in World space: C_world = -R^T * t
        cam_world = -np.matrix(R_mat).T * np.matrix(tvec)

        return {
            "matched": True,
            "inliers": inlier_count,
            "status": "LOCKED",
            "pose": {
                "position": {
                    "x": float(cam_world[0, 0]),
                    "y": float(cam_world[1, 0]),
                    "z": float(cam_world[2, 0])
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