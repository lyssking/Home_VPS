import os
import glob
import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
import torch
from lightglue import SuperPoint

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIDAR_PATH = os.path.join(SCRIPT_DIR, "models", "room_scan.ply")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "mapping_images")
OUTPUT_DB = os.path.join(SCRIPT_DIR, "vps_lidar_db.npz")

if not os.path.exists(LIDAR_PATH):
    LIDAR_PATH = os.path.join(SCRIPT_DIR, "room_scan.ply")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading SuperPoint on {device}...")
extractor = SuperPoint(max_num_keypoints=3000).eval().to(device)

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
print(f"[1/3] Loading 3D scan from {LIDAR_PATH}...")
mesh = trimesh.load(LIDAR_PATH, process=False)
if isinstance(mesh, trimesh.Scene):
    meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
    mesh = trimesh.util.concatenate(meshes)

bounds = mesh.bounds
extents = mesh.extents
centroid = mesh.centroid

if np.max(extents) > 50.0:
    mesh.apply_scale(0.001 if np.max(extents) > 500.0 else 0.01)
    bounds, extents, centroid = mesh.bounds, mesh.extents, mesh.centroid

floor_z = bounds[0][2]
T_cam = np.array([centroid[0], centroid[1], floor_z + 1.4], dtype=np.float32)
vertex_tree = cKDTree(mesh.vertices)

image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
print(f"[2/3] Processing {len(image_paths)} reference keyframes...")

all_3d_points, all_descriptors = [], []

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None: continue
    h, w, _ = img.shape
    fx = fy = (w / 2.0) / np.tan(np.radians(60.0 / 2.0))
    cx, cy = w / 2.0, h / 2.0

    kps, descs = extract_superpoint(img)
    if len(kps) == 0: continue

    for idx, kp in enumerate(kps):
        u, v = kp[0], kp[1]
        ray_dir = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float32)
        ray_dir /= np.linalg.norm(ray_dir)

        sample_steps = np.linspace(0.4, max(float(np.max(extents)), 6.0), 50)
        ray_pts = T_cam + np.outer(sample_steps, ray_dir)

        distances, indices = vertex_tree.query(ray_pts, k=1)
        min_idx = np.argmin(distances)

        if distances[min_idx] < 0.45:
            all_3d_points.append(mesh.vertices[indices[min_idx]])
            all_descriptors.append(descs[idx])

all_3d_points = np.array(all_3d_points, dtype=np.float32)
all_descriptors = np.array(all_descriptors, dtype=np.float32)

print(f"[3/3] Saving {len(all_3d_points)} geometric anchors to {OUTPUT_DB}...")
np.savez(OUTPUT_DB, points3D=all_3d_points, descriptors=all_descriptors)