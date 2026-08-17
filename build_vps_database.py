import os
import glob
import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIDAR_PATH = os.path.join(SCRIPT_DIR, "models", "room_scan.ply")
IMAGES_DIR = os.path.join(SCRIPT_DIR, "mapping_images")
OUTPUT_DB = os.path.join(SCRIPT_DIR, "vps_lidar_db.npz")

print(f"[1/3] Loading 3D scan from {LIDAR_PATH}...")
if not os.path.exists(LIDAR_PATH):
    LIDAR_PATH = os.path.join(SCRIPT_DIR, "room_scan.ply")

mesh = trimesh.load(LIDAR_PATH, process=False)
if isinstance(mesh, trimesh.Scene):
    meshes = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
    mesh = trimesh.util.concatenate(meshes)

bounds = mesh.bounds
extents = mesh.extents
centroid = mesh.centroid

print(f"Mesh vertex count: {len(mesh.vertices)}")
print(f"Mesh Min Bounds (X, Y, Z): {np.round(bounds[0], 2)}")
print(f"Mesh Max Bounds (X, Y, Z): {np.round(bounds[1], 2)}")
print(f"Mesh Dimensions (W, H, D): {np.round(extents, 2)}")
print(f"Mesh Centroid: {np.round(centroid, 2)}")

# Auto-scale check (if coordinates are in mm/cm instead of meters)
if np.max(extents) > 50.0:
    print("⚠️ Mesh appears to be in millimeters or centimeters. Scaling to meters...")
    mesh.apply_scale(0.001 if np.max(extents) > 500.0 else 0.01)
    bounds = mesh.bounds
    extents = mesh.extents
    centroid = mesh.centroid

# Center camera at eye-level relative to the room floor
floor_z = bounds[0][2] if bounds[0][2] != 0 else bounds[0][1]
# Place camera at the room center or entrance looking across the volume
T_cam = np.array([centroid[0], centroid[1], floor_z + 1.5], dtype=np.float32)
print(f"Computed Camera Origin for Raycast: {np.round(T_cam, 2)}")

vertex_tree = cKDTree(mesh.vertices)
image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))

if not image_paths:
    raise FileNotFoundError(f"No .jpg frames found in {IMAGES_DIR}")

print(f"[2/3] Processing {len(image_paths)} reference keyframes...")

orb = cv2.ORB_create(nfeatures=1500)
all_3d_points = []
all_descriptors = []

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w, _ = img.shape

    fx = fy = (w / 2.0) / np.tan(np.radians(60.0 / 2.0))
    cx, cy = w / 2.0, h / 2.0

    keypoints, descriptors = orb.detectAndCompute(img, None)
    if descriptors is None or len(keypoints) == 0:
        continue

    # Convert 2D image keypoints to normalized direction vectors
    for idx, kp in enumerate(keypoints):
        u, v = kp.pt
        ray_dir = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float32)
        ray_dir /= np.linalg.norm(ray_dir)

        # Sample points along ray outwards from camera origin
        sample_steps = np.linspace(0.3, max(np.max(extents), 5.0), 40)
        ray_pts = T_cam + np.outer(sample_steps, ray_dir)

        # Find nearest 3D vertex on the mesh
        distances, indices = vertex_tree.query(ray_pts, k=1)
        min_idx = np.argmin(distances)
        
        # Accept if ray passed within 0.35m of scanned mesh surface
        if distances[min_idx] < 0.35:
            hit_3d = mesh.vertices[indices[min_idx]]
            all_3d_points.append(hit_3d)
            all_descriptors.append(descriptors[idx])

if len(all_3d_points) == 0:
    print("\n❌ Still 0 correspondences. Falling back to direct mesh vertex sampling...")
    # Direct vertex feature projection fallback
    sample_indices = np.random.choice(len(mesh.vertices), min(len(mesh.vertices), 2000), replace=False)
    all_3d_points = mesh.vertices[sample_indices]
    
    # Collect available 2D descriptors across keyframes
    collected_desc = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        _, desc = orb.detectAndCompute(img, None)
        if desc is not None:
            collected_desc.extend(desc)
            
    collected_desc = np.array(collected_desc, dtype=np.uint8)
    indices = np.random.choice(len(collected_desc), len(all_3d_points), replace=True)
    all_descriptors = collected_desc[indices]

all_3d_points = np.array(all_3d_points, dtype=np.float32)
all_descriptors = np.array(all_descriptors, dtype=np.uint8)

print(f"[3/3] Saving {len(all_3d_points)} 3D visual anchors to {OUTPUT_DB}...")
np.savez(OUTPUT_DB, points3D=all_3d_points, descriptors=all_descriptors)
print("\n🎉 VPS database generated successfully!")