import os
import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from lightglue import SuperPoint

app = FastAPI(title="SuperPoint WebXR VPS Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LocalizeRequest(BaseModel):
    image: Optional[str] = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Initializing SuperPoint on device: {device}")
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)

def extract_superpoint(img_bgr):
    # 1. Convert to LAB color space to isolate the Lightness (L) channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # 2. Apply CLAHE mathematically to flatten shadows and highlights
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    # 3. Merge back and convert to RGB
    limg = cv2.merge((cl, a_channel, b_channel))
    img_equalized = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    img_rgb = cv2.cvtColor(img_equalized, cv2.COLOR_BGR2RGB)
    
    # 4. Feed the normalized image to SuperPoint
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        res = extractor.extract(tensor.to(device))
    return res['keypoints'][0].cpu().numpy(), res['descriptors'][0].cpu().numpy()

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

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

@app.get("/")
def health_check():
    return {"status": "online", "anchors_loaded": len(db_points3D), "device": str(device)}

@app.post("/api/vps/localize")
async def localize(req: LocalizeRequest):
    if not req.image or len(db_descriptors) == 0:
        return {"matched": False, "inliers": 0, "status": "NO_DATA"}

    img_data = req.image.split(",", 1)[1] if "," in req.image else req.image
    img_bytes = base64.b64decode(img_data)
    frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    if frame is None:
        return {"matched": False, "inliers": 0, "status": "DECODE_ERROR"}

    h, w, _ = frame.shape
    kps, descs = extract_superpoint(frame)

    if len(kps) < 8:
        return {"matched": False, "inliers": 0, "status": "INSUFFICIENT_FEATURES"}

    matches = flann.knnMatch(descs, db_descriptors, k=2)
    good_matches = [m for m, n in matches if len((m, n)) == 2 and m.distance < 0.82 * n.distance]

    if len(good_matches) < 6:
        return {"matched": False, "inliers": len(good_matches), "status": "LOW_MATCHES"}

    pts_2d = np.float32([kps[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
    pts_3d = np.float32([db_points3D[m.trainIdx] for m in good_matches]).reshape(-1, 1, 3)

    fx = fy = (w / 2.0) / np.tan(np.radians(60.0 / 2.0))
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, camera_matrix, np.zeros((4, 1), dtype=np.float32),
        reprojectionError=10.0, iterationsCount=1000, confidence=0.99, flags=cv2.SOLVEPNP_SQPNP
    )

    inlier_count = len(inliers) if inliers is not None else 0

    if success and inlier_count >= 6:
        R_w2c, _ = cv2.Rodrigues(rvec)
        
        # Camera-to-World Transform (OpenCV format: +X Right, +Y Down, +Z Forward)
        R_c2w = R_w2c.T
        T_c2w = -R_w2c.T @ tvec

        M_c2w_cv = np.eye(4, dtype=np.float32)
        M_c2w_cv[:3, :3] = R_c2w
        M_c2w_cv[:3, 3] = T_c2w.flatten()

        # Convert to WebGL coordinate format (+X Right, +Y Up, -Z Forward)
        # Multiply by diag(1, -1, -1, 1) to flip Y and Z axes
        cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
        M_c2w_gl = M_c2w_cv @ cv_to_gl

        # Three.js uses Column-Major format (transpose for flat array)
        matrix_column_major = M_c2w_gl.T.flatten().tolist()

        return {
            "matched": True,
            "inliers": inlier_count,
            "status": "LOCKED",
            "matrix4": matrix_column_major
        }

    return {"matched": False, "inliers": inlier_count, "status": "SEARCHING"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)