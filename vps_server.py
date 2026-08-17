import base64
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

app = FastAPI(title="Local LiDAR VPS Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "vps_lidar_db.npz"
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(f"Missing {DB_PATH}. Run lift_features_to_lidar.py first.")

data = np.load(DB_PATH)
ref_points3D = data['points3D']
ref_descriptors = data['descriptors']
print(f"Loaded VPS Database: {len(ref_points3D)} spatial anchor points.")

orb = cv2.ORB_create(nfeatures=2000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

class LocalizePayload(BaseModel):
    image_b64: str
    fx: float
    fy: float
    ox: float
    oy: float

@app.post("/api/vps/localize")
async def localize(req: LocalizePayload):
    try:
        img_bytes = base64.b64decode(req.image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if frame is None:
            return {"success": False, "error": "Frame decode failure"}

        keypoints, descriptors = orb.detectAndCompute(frame, None)
        if descriptors is None or len(keypoints) < 8:
            return {"success": False, "error": "Not enough visual texture"}

        matches = matcher.knnMatch(descriptors, ref_descriptors, k=2)
        good_3d = []
        good_2d = []

        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < 0.75 * n.distance:
                    good_3d.append(ref_points3D[m.trainIdx])
                    good_2d.append(keypoints[m.queryIdx].pt)

        if len(good_3d) < 6:
            return {"success": False, "error": "Insufficient landmark matches"}

        pts_3d = np.ascontiguousarray(good_3d, dtype=np.float32).reshape(-1, 3)
        pts_2d = np.ascontiguousarray(good_2d, dtype=np.float32).reshape(-1, 2)

        camera_matrix = np.array([
            [req.fx, 0, req.ox],
            [0, req.fy, req.oy],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, camera_matrix, dist_coeffs,
            reprojectionError=8.0,
            iterationsCount=150,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success or inliers is None or len(inliers) < 6:
            return {"success": False, "error": "PnP RANSAC alignment failed"}

        R, _ = cv2.Rodrigues(rvec)
        R_world = R.T
        t_world = -R_world @ tvec.reshape(3, 1)

        # 4x4 Column-Major Matrix for Three.js
        transform_matrix = [
            float(R_world[0, 0]), float(R_world[1, 0]), float(R_world[2, 0]), 0.0,
            float(R_world[0, 1]), float(R_world[1, 1]), float(R_world[2, 1]), 0.0,
            float(R_world[0, 2]), float(R_world[1, 2]), float(R_world[2, 2]), 0.0,
            float(t_world[0, 0]), float(t_world[1, 0]), float(t_world[2, 0]), 1.0
        ]

        return {
            "success": True,
            "inliers": len(inliers),
            "matrix": transform_matrix
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)