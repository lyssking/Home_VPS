import os
import glob
import json
import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import torch
from lightglue import SuperPoint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIDAR_PATH = os.path.join(SCRIPT_DIR, "models", "room_scan.ply")
MAPPING_DIR = os.path.join(SCRIPT_DIR, "mapping_data")
OUTPUT_DB = os.path.join(SCRIPT_DIR, "vps_lidar_db.npz")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
extractor = SuperPoint(max_num_keypoints=3000).eval().to(device)

def extract_superpoint(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        res = extractor.extract(tensor.to(device))
    return res['keypoints'][0].cpu().numpy(), res['descriptors'][0].cpu().numpy()

print(f"[1/3] Loading 3D scan from {LIDAR_PATH}...")
mesh = trimesh.load(LIDAR_PATH, process=False)
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate([g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)])

extents = mesh.extents
if np.max(extents) > 50.0:
    mesh.apply_scale(0.01)
vertex_tree = cKDTree(mesh.vertices)

json_files = sorted(glob.glob(os.path.join(MAPPING_DIR, "*.json")))
all_3d_points, all_descriptors = [], []

print(f"[2/3] Processing {len(json_files)} precisely logged AR frames...")

for json_path in json_files:
    img_path = json_path.replace(".json", ".jpg")
    if not os.path.exists(img_path): 
        continue

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # ThreeJS is column-major, swap to row-major
    matrix = np.array(data["matrix"]).reshape(4, 4).T 
    T_cam = matrix[:3, 3]
    R_cam = matrix[:3, :3]

    img = cv2.imread(img_path)
    if img is None: continue
    
    h, w, _ = img.shape
    fx = fy = (w / 2.0) / np.tan(np.radians(73.0 / 2.0))
    cx, cy = w / 2.0, h / 2.0

    kps, descs = extract_superpoint(img)
    if len(kps) == 0: continue

    for idx, kp in enumerate(kps):
        u, v = kp[0], kp[1]
        
        # We assume ThreeJS camera space: -Z is forward, Y is up, X is right
        local_ray = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0], dtype=np.float32)
        local_ray /= np.linalg.norm(local_ray)
        world_ray = R_cam @ local_ray

        sample_steps = np.linspace(0.2, 10.0, 50)
        ray_pts = T_cam + np.outer(sample_steps, world_ray)

        distances, indices = vertex_tree.query(ray_pts, k=1)
        min_idx = np.argmin(distances)

        if distances[min_idx] < 0.45:
            all_3d_points.append(mesh.vertices[indices[min_idx]])
            all_descriptors.append(descs[idx])

all_3d_points = np.array(all_3d_points, dtype=np.float32)
all_descriptors = np.array(all_descriptors, dtype=np.float32)

print(f"[3/3] ✅ Saving {len(all_3d_points)} geometric anchors to {OUTPUT_DB}")
np.savez(OUTPUT_DB, points3D=all_3d_points, descriptors=all_descriptors)