#!/usr/bin/env python3
"""
SMPL与RealSense深度图评估
采样200个面向相机的SMPL前表面点，与深度图3D点对比计算误差
"""
import os
import sys
import numpy as np
import cv2
import json
import glob
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_camera_params(intri_path, extri_path, cam_id='04'):
    from easymocap.mytools.camera_utils import read_camera
    cameras = read_camera(intri_path, extri_path)
    cam = cameras[cam_id]
    return cam['K'], cam['R'], cam['T'], cam.get('dist', np.zeros((1, 5)))

def load_smpl_model(model_path):
    from easymocap.bodymodel.smpl import SMPLModel
    return SMPLModel(model_path, device='cpu')

def load_smpl_params(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data[0] if len(data) > 0 else None

def get_smpl_vertices(smpl_model, params):
    import torch
    poses = torch.FloatTensor(params['poses']).reshape(1, -1)
    shapes = torch.FloatTensor(params['shapes']).reshape(1, -1)
    Rh = torch.FloatTensor(params['Rh']).reshape(1, 3)
    Th = torch.FloatTensor(params['Th']).reshape(1, 3)
    
    with torch.no_grad():
        vertices = smpl_model(poses=poses, shapes=shapes, Rh=Rh, Th=Th).cpu().numpy()
        if vertices.ndim == 3:
            vertices = vertices[0]
    return vertices

def transform_to_camera(vertices, R, T):
    return (R @ vertices.T).T + T.reshape(1, 3)

def compute_face_normals(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    return normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)

def select_front_facing_vertices(vertices_cam, smpl_model, num_samples=200):
    faces = smpl_model.faces
    face_centers = np.mean(vertices_cam[faces], axis=1)
    face_normals = compute_face_normals(vertices_cam, faces)
    
    view_dirs = -face_centers
    view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8)
    
    dot_products = np.sum(face_normals * view_dirs, axis=1)
    front_faces = faces[dot_products > 0]
    front_vertex_indices = np.unique(front_faces.flatten())
    
    if len(front_vertex_indices) > num_samples:
        sampled_indices = np.random.choice(front_vertex_indices, num_samples, replace=False)
    else:
        sampled_indices = front_vertex_indices
    
    return sampled_indices

def project_to_image(points_3d, K, dist):
    points_2d, _ = cv2.projectPoints(points_3d.reshape(-1, 1, 3), 
                                      np.zeros(3), np.zeros(3), K, dist)
    return points_2d.reshape(-1, 2)

def read_depth_image(depth_path, depth_scale=0.001):
    """
    读取深度图像并转换为米
    
    Args:
        depth_path: 深度图路径
        depth_scale: RealSense深度单位
    """
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if len(depth.shape) == 3:
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    
    if depth.dtype == np.uint16:
        # RealSense 16-bit深度
        depth = depth.astype(np.float32) * depth_scale
    elif depth.dtype == np.uint8:
        # 8-bit可视化图（不推荐）
        depth = depth.astype(np.float32) / 255.0 * 10.0
    
    return depth

def backproject_depth_to_3d(points_2d, depths, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (points_2d[:, 0] - cx) * depths / fx
    y = (points_2d[:, 1] - cy) * depths / fy
    z = depths
    return np.stack([x, y, z], axis=1)

def evaluate_single_frame(frame_idx, smpl_dir, depth_dir, smpl_model, K, R, T, dist, 
                         num_samples=200, img_width=1280, img_height=720, depth_scale=0.001):
    smpl_path = os.path.join(smpl_dir, f'{frame_idx:06d}.json')
    depth_path = os.path.join(depth_dir, f'{frame_idx:06d}.png')
    
    if not os.path.exists(smpl_path) or not os.path.exists(depth_path):
        return None
    
    params = load_smpl_params(smpl_path)
    if params is None:
        return None
    
    vertices_world = get_smpl_vertices(smpl_model, params)
    vertices_cam = transform_to_camera(vertices_world, R, T)
    
    sampled_indices = select_front_facing_vertices(vertices_cam, smpl_model, num_samples)
    sampled_vertices = vertices_cam[sampled_indices]
    points_2d = project_to_image(sampled_vertices, K, dist)
    
    depth_map = read_depth_image(depth_path, depth_scale)
    
    valid_mask = np.ones(len(points_2d), dtype=bool)
    depths = np.zeros(len(points_2d))
    
    for i, (u, v) in enumerate(points_2d):
        u_int, v_int = int(round(u)), int(round(v))
        if 0 <= u_int < img_width and 0 <= v_int < img_height:
            depths[i] = depth_map[v_int, u_int]
            if depths[i] <= 0 or depths[i] > 10.0:
                valid_mask[i] = False
        else:
            valid_mask[i] = False
    
    points_3d_depth = backproject_depth_to_3d(points_2d[valid_mask], depths[valid_mask], K)
    errors = np.linalg.norm(points_3d_depth - sampled_vertices[valid_mask], axis=1)
    
    return {
        'frame_idx': frame_idx,
        'num_valid': len(errors),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'median_error': np.median(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'errors': errors
    }

def main():
    root_dir = 'mmWave/EasyMocap'
    smpl_dir = os.path.join(root_dir, 'output/detect_triangulate_fitSMPL/smpl')
    depth_dir = os.path.join(root_dir, 'Eval/depth')
    intri_path = os.path.join(root_dir, 'data/examples/my_multiview/intri.yml')
    extri_path = os.path.join(root_dir, 'data/examples/my_multiview/extri.yml')
    model_path = os.path.join(root_dir, 'models/pare/data/body_models/smpl/SMPL_NEUTRAL.pkl')
    output_dir = os.path.join(root_dir, 'Eval')
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = 200
    img_width = 1280
    img_height = 720
    depth_scale = 0.001
    
    print("="*70)
    print("SMPL vs RealSense Depth Evaluation")
    print("="*70)
    
    print("\nLoading SMPL model...")
    smpl_model = load_smpl_model(model_path)
    
    print("Loading camera parameters (Camera 04)...")
    K, R, T, dist = load_camera_params(intri_path, extri_path, cam_id='04')
    print(f"Intrinsic matrix K:\n{K}")
    print(f"Samples per frame: {num_samples}")
    print(f"Image size: {img_width} x {img_height}")
    print(f"Depth scale: {depth_scale} m/count ({depth_scale*1000:.1f} mm/count)")
    
    smpl_files = sorted(glob.glob(os.path.join(smpl_dir, '*.json')))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
    
    smpl_frame_ids = [int(os.path.basename(f).split('.')[0]) for f in smpl_files]
    depth_frame_ids = [int(os.path.basename(f).split('.')[0]) for f in depth_files]
    
    matched_frames = sorted(list(set(smpl_frame_ids) & set(depth_frame_ids)))
    
    print(f"\nFound {len(smpl_files)} SMPL frames, {len(depth_files)} depth images")
    print(f"SMPL range: {min(smpl_frame_ids):06d} - {max(smpl_frame_ids):06d}")
    print(f"Depth range: {min(depth_frame_ids):06d} - {max(depth_frame_ids):06d}")
    print(f"Matched frames: {len(matched_frames)}")
    print(f"Evaluating {len(matched_frames)} frames...")
    
    results = []
    for frame_idx in tqdm(matched_frames, desc="Progress"):
        result = evaluate_single_frame(
            frame_idx, smpl_dir, depth_dir, smpl_model, K, R, T, dist,
            num_samples=num_samples, img_width=img_width, img_height=img_height,
            depth_scale=depth_scale
        )
        if result is not None:
            results.append(result)
    
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    
    if len(results) == 0:
        print("No valid results!")
        return
    
    all_errors = np.concatenate([r['errors'] for r in results])
    mean_errors = [r['mean_error'] for r in results]
    
    print(f"\nSuccessfully evaluated {len(results)} frames")
    print(f"\nOverall statistics (all points):")
    print(f"  Total points:  {len(all_errors)}")
    print(f"  Mean error:    {np.mean(all_errors):.4f} m ({np.mean(all_errors)*1000:.2f} mm)")
    print(f"  Std dev:       {np.std(all_errors):.4f} m ({np.std(all_errors)*1000:.2f} mm)")
    print(f"  Median:        {np.median(all_errors):.4f} m ({np.median(all_errors)*1000:.2f} mm)")
    print(f"  Max error:     {np.max(all_errors):.4f} m ({np.max(all_errors)*1000:.2f} mm)")
    print(f"  Min error:     {np.min(all_errors):.4f} m ({np.min(all_errors)*1000:.2f} mm)")
    
    print(f"\nPer-frame statistics:")
    print(f"  Mean of means: {np.mean(mean_errors):.4f} m ({np.mean(mean_errors)*1000:.2f} mm)")
    print(f"  Std of means:  {np.std(mean_errors):.4f} m ({np.std(mean_errors)*1000:.2f} mm)")
    
    output_json = os.path.join(output_dir, 'evaluation_results.json')
    
    results_json = [{
        'frame_idx': int(r['frame_idx']),
        'num_valid': int(r['num_valid']),
        'mean_error': float(r['mean_error']),
        'std_error': float(r['std_error']),
        'median_error': float(r['median_error']),
        'max_error': float(r['max_error']),
        'min_error': float(r['min_error'])
    } for r in results]
    
    summary = {
        'num_frames': len(results),
        'total_points': int(len(all_errors)),
        'overall': {
            'mean_error_m': float(np.mean(all_errors)),
            'std_error_m': float(np.std(all_errors)),
            'median_error_m': float(np.median(all_errors)),
            'max_error_m': float(np.max(all_errors)),
            'min_error_m': float(np.min(all_errors))
        },
        'per_frame': results_json
    }
    
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_json}")
    
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SMPL vs RealSense Depth Evaluation Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"Frames evaluated: {len(results)}\n")
        f.write(f"Total points: {len(all_errors)}\n")
        f.write(f"Samples per frame: {num_samples}\n\n")
        f.write("Overall error statistics:\n")
        f.write(f"  Mean error:   {np.mean(all_errors):.4f} m ({np.mean(all_errors)*1000:.2f} mm)\n")
        f.write(f"  Std dev:      {np.std(all_errors):.4f} m ({np.std(all_errors)*1000:.2f} mm)\n")
        f.write(f"  Median:       {np.median(all_errors):.4f} m ({np.median(all_errors)*1000:.2f} mm)\n")
        f.write(f"  Max error:    {np.max(all_errors):.4f} m ({np.max(all_errors)*1000:.2f} mm)\n")
        f.write(f"  Min error:    {np.min(all_errors):.4f} m ({np.min(all_errors)*1000:.2f} mm)\n\n")
        f.write("Per-frame statistics (first 10 frames):\n")
        for r in results[:10]:
            f.write(f"  Frame {r['frame_idx']:06d}: {r['mean_error']:.4f} m ({r['num_valid']} valid points)\n")
        if len(results) > 10:
            f.write(f"  ... ({len(results)} frames total)\n")
    
    print(f"Report saved to: {report_path}")
    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()

