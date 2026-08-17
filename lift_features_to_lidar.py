import cv2
import numpy as np
import open3d as o3d
import os

LIDAR_PATH = "models/lab_lidar_scan.ply"
IMAGE_PATH = "mapping_images/frame_0001.jpg"
OUTPUT_DB = "vps_lidar_db.npz"

print(f"[1/4] Loading LiDAR mesh from {LIDAR_PATH}...")
mesh = o3d.io.read_triangle_mesh(LIDAR_PATH)
mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(mesh_t)

print(f"[2/4] Extracting ORB visual features from {IMAGE_PATH}...")
img = cv2.imread(IMAGE_PATH)
h, w, _ = img.shape
orb = cv2.ORB_create(nfeatures=2500)
keypoints, descriptors = orb.detectAndCompute(img, None)

# Mobile camera intrinsic approximation (FOV ~ 60 deg)
fx = fy = (w / 2) / np.tan(np.radians(60.0 / 2.0))
cx, cy = w / 2.0, h / 2.0

# Camera position where reference image was taken (Origin at entrance, 1.5m eye height)
R_cam = np.eye(3)
T_cam = np.array([0.0, 1.5, 0.0], dtype=np.float32)

print("[3/4] Raycasting 2D pixel rays into 3D LiDAR geometry...")
rays = []
valid_descriptors = []
valid_keypoints = []

for idx, kp in enumerate(keypoints):
    u, v = kp.pt
    ray_dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    ray_dir_cam /= np.linalg.norm(ray_dir_cam)
    
    ray_dir_world = R_cam @ ray_dir_cam
    rays.append([*T_cam, *ray_dir_world])
    valid_descriptors.append(descriptors[idx])
    valid_keypoints.append(kp.pt)

rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
raycast_results = scene.cast_rays(rays_tensor)
t_hit = raycast_results['t_hit'].numpy()

hit_3d_points = []
final_descriptors = []
final_2d_points = []

for i, t in enumerate(t_hit):
    if not np.isinf(t):
        origin = np.array(rays[i][:3])
        direction = np.array(rays[i][3:])
        point_3d = origin + direction * t
        
        hit_3d_points.append(point_3d)
        final_descriptors.append(valid_descriptors[i])
        final_2d_points.append(valid_keypoints[i])

hit_3d_points = np.array(hit_3d_points, dtype=np.float32)
final_descriptors = np.array(final_descriptors, dtype=np.uint8)

print(f"[4/4] Saving {len(hit_3d_points)} metric 3D-2D visual correspondences to {OUTPUT_DB}...")
np.savez(OUTPUT_DB, points3D=hit_3d_points, descriptors=final_descriptors)
print("Calibration complete.")