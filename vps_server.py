import os
import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from lightglue import SuperPoint

app = FastAPI(title="SuperPoint VPS Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class LocalizeRequest(BaseModel):
    image: Optional[str] = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=3000).eval().to(device)

def extract_superpoint(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        res = extractor.extract(tensor.to(device))
    return res['keypoints'][0].cpu().numpy(), res['descriptors'][0].cpu().numpy()

DB_PATH = "vps_lidar_db.npz"
if os.path.exists(DB_PATH):
    db = np.load(DB_PATH)
    db_points3D = db["points3D"]
    db_descriptors = db["descriptors"]
else:
    db_points3D, db_descriptors = [], []

# L2 Norm for floating-point neural descriptors
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

@app.post("/api/vps/localize")
async def localize(req: LocalizeRequest):
    img_data = req.image.split(",", 1)[1] if "," in req.image else req.image
    frame = cv2.imdecode(np.frombuffer(base64.b64decode(img_data), np.uint8), cv2.IMREAD_COLOR)

    h, w, _ = frame.shape
    kps, descs = extract_superpoint(frame)

    if len(kps) < 10 or len(db_descriptors) == 0:
        return {"matched": False, "inliers": 0, "status": "INSUFFICIENT_FEATURES"}

    matches = bf.knnMatch(descs, db_descriptors, k=2)
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
        reprojectionError=12.0, iterationsCount=1000, confidence=0.99, flags=cv2.SOLVEPNP_SQPNP
    )

    inlier_count = len(inliers) if inliers is not None else 0

    if success and inlier_count >= 6:
        R_mat, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(R_mat[0, 0]**2 + R_mat[1, 0]**2)
        rx = np.arctan2(R_mat[2, 1], R_mat[2, 2]) if sy > 1e-6 else np.arctan2(-R_mat[1, 2], R_mat[1, 1])
        ry = np.arctan2(-R_mat[2, 0], sy)
        rz = np.arctan2(R_mat[1, 0], R_mat[0, 0]) if sy > 1e-6 else 0
        cam_world = -np.matrix(R_mat).T * np.matrix(tvec)

        return {
            "matched": True,
            "inliers": inlier_count,
            "status": "LOCKED",
            "pose": {
                "position": {"x": float(cam_world[0, 0]), "y": float(cam_world[1, 0]), "z": float(cam_world[2, 0])},
                "rotation": {"x": float(rx), "y": float(ry), "z": float(rz)}
            }
        }

    return {"matched": False, "inliers": inlier_count, "status": "SEARCHING"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)