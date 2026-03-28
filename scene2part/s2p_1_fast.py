import os
import numpy as np
import pickle
from tqdm import tqdm
import nvdiffrast.torch as dr
import trimesh
import torch
import torch.nn.functional as F
import torchvision
import networkx as nx
from itertools import combinations
import pymeshlab as pml
import subprocess
from scipy.ndimage import binary_erosion
from scipy.sparse import coo_matrix, csr_matrix
import argparse
from PIL import Image
import json
import concurrent.futures
from collections import defaultdict

import random

import trimesh.exchange
import trimesh.exchange.export
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d as o3d

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
#from test_pose import visualize_camera_view

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings
)
#from utilities import compute_projection


random.seed(42)

import kaolin as kal

def rasterize_kaolin(verts, faces, w2c, proj, resolution, sigma_inv=1/(3*(10**(-5))), box_len=0.02, k_num=30):
    
    verts = verts.unsqueeze(0)
    faces = faces.long()
    
    verts_h = F.pad(verts, pad=(0, 1), mode='constant', value=1.0)
    
    verts_camera_h = torch.matmul(verts_h, w2c.t())
    coord_transform = torch.eye(4).to(verts_camera_h.device)
    coord_transform[0, 0] = -1
    coord_transform[2, 2] = -1
    verts_camera_h = verts_camera_h @ coord_transform
    # verts_camera_h[..., [0, 2]] *= -1
    
    verts_camera = verts_camera_h[:, :, :3] / verts_camera_h[:, :, 3:]
    
    verts_clip_h = torch.matmul(verts_camera_h, proj.t())
    verts_clip = verts_clip_h[:, :, :3] / verts_clip_h[:, :, 3:]
    
    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(verts_camera, faces) # (1, F, 3(v), 3(xyz))
    face_vertices_clip = kal.ops.mesh.index_vertices_by_faces(verts_clip, faces) # (1, F, 3(v), 3(xyz))
    
    face_normals = kal.ops.mesh.face_normals(face_vertices_camera)
    
    face_vertices_z = face_vertices_camera[..., 2]
    
    visible_faces = torch.all(face_vertices_z <= 0, dim=-1)
    face_mapping = torch.arange(faces.shape[0]).to(visible_faces.device)[visible_faces.reshape(-1)]
    
    face_vertices_z = face_vertices_z[visible_faces].unsqueeze(0)
    face_vertices_clip = face_vertices_clip[visible_faces].unsqueeze(0)
    face_vertices_camera = face_vertices_camera[visible_faces].unsqueeze(0)
    face_normals = face_normals[visible_faces].unsqueeze(0)
    
    h, w = resolution
    
    rendered_features, soft_mask, face_idx_map = kal.render.mesh.dibr_rasterization(
        height=h,
        width=w,
        face_vertices_z=face_vertices_z,  # Depth values
        face_vertices_image=face_vertices_clip[..., :2],  # Image plane coordinates
        face_features=-face_vertices_camera[..., 2:3],  # Optional: per-vertex features like colors
        face_normals_z=face_normals[..., 2],  # Z-component of face normals
        sigmainv=sigma_inv,
        boxlen=box_len,
        knum=k_num
    )
    
    soft_mask = soft_mask.squeeze(0)
    face_idx_map = face_mapping[face_idx_map.reshape(-1)].reshape(h, w)
    depth_map = rendered_features.reshape(h, w)
    
    return soft_mask, face_idx_map, depth_map
    
def rasterize_texture_kaolin(verts, faces, w2c, proj, resolution):
    soft_mask, face_idx_map, depth_map = rasterize_kaolin(verts, faces, w2c, proj, resolution, sigma_inv=7*10**8)
    valid = face_idx_map >= 0
    triangle_id = face_idx_map.long()
    
    return valid, triangle_id

def rasterize_texture(vertices, faces, projection, glctx, resolution):

    vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(projection, 0, 1)).float().unsqueeze(0)
    rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, resolution=resolution)
    # rast_out = rast_out.flip([1])

    H, W = resolution
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)

    return valid, triangle_id

def rasterize_mesh(
    vertices,
    faces,
    c2w,
    intrinsic,
    image_size,
    device
):
    """
    Rasterize single frame camera pose onto the mesh, return valid mask and triangle_id.

    Args:
        vertices:    (V, 3) Vertex coordinates
        faces:       (F, 3) Triangle vertex indices
        c2w:         (4, 4) Camera to world transformation matrix
        intrinsic:   (3, 3) Camera intrinsic matrix
        image_size:  (H, W) Output image dimensions
        device:      torch.device, automatically selected if None

    Returns:
        valid:       (H, W) bool tensor, whether pixel is on the mesh
        triangle_id: (H, W) int tensor, pixel's corresponding triangle index, -1 if invalid
    """
    # Device selection
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    flip_z = torch.tensor([
        [1, 0,  0, 0],
        [0, 1,  0, 0],
        [0, 0, -1, 0],  # Invert Z axis
        [0, 0,  0, 1]
    ], device=device, dtype=torch.float32)
    
    # Step 2: Calculate world->camera transformation matrix
    w2c = torch.inverse(c2w.to(device))
    
    # Step 3: Combine transformation matrices (apply w2c first, then invert Z axis)
    corrected_w2c = flip_z @ w2c
    
    # Extract rotation and translation
    R = corrected_w2c[:3, :3].unsqueeze(0)  # (1, 3, 3)
    T = corrected_w2c[:3, 3].unsqueeze(0)   # (1, 3)

    # Build Meshes
    verts = vertices.to(device)
    faces_idx = faces.to(device)
    mesh = Meshes(verts=[verts], faces=[faces_idx])

    H, W = image_size

    # Extract focal length and principal point from intrinsic
    fx = intrinsic[0, 0].item()
    fy = intrinsic[1, 1].item()
    cx = intrinsic[0, 2].item()
    cy = intrinsic[1, 2].item()

    # Build PerspectiveCameras
    cameras = PerspectiveCameras(
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        R=R, T=T,
        image_size=((H, W),),
        in_ndc=False,
        device=device
    )

    # Rasterization config: only take the nearest face per pixel
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,  # Perspective correction
        cull_backfaces=False,  # Disable backface culling
        clip_barycentric_coords=False  # Do not clip barycentric coordinates
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    # Execute rasterization
    fragments = rasterizer(mesh)
    pix_to_face = fragments.pix_to_face[0, ..., 0]  # (H, W)

    # Construct output
    valid = pix_to_face >= 0
    triangle_id = torch.where(valid, pix_to_face, -1)

    return valid.cpu(), triangle_id.cpu()

def rasterize_texture_custom(vertices, faces, c2w, K, glctx, resolution):
    """
    vertices: (V,3) world coords
    faces: (F,3)
    c2w: 4x4 camera-to-world matrix
    K: 3x3 intrinsic matrix
    """
    H, W = resolution

    # Compute world-to-camera from provided c2w
    w2c_np = np.linalg.inv(c2w)  # **MODIFIED** invert c2w inside function
    w2c = torch.from_numpy(w2c_np).float().cuda()

    # Transform to homogeneous coordinates
    verts_h = F.pad(vertices, (0,1), mode='constant', value=1.0)  # (V,4)
    # World -> camera
    verts_cam_h = verts_h @ w2c.t()  # (V,4)
    verts_cam = verts_cam_h[:, :3]    # (V,3)

    # Project to image plane
    xy = (K @ verts_cam.t()).t()      # (V,3)
    u = xy[:, 0] / xy[:, 2]
    v = xy[:, 1] / xy[:, 2]

    # Convert to NDC coordinates in [-1,1]
    x_ndc = (u / W) * 2 - 1
    y_ndc = 1 - (v / H) * 2
    z_ndc = verts_cam[:, 2]  # depth

    # Build clip-space vert tensor
    verts_clip = torch.stack([x_ndc, y_ndc, z_ndc, torch.ones_like(z_ndc)], dim=1).unsqueeze(0)

    # Rasterize using nvdiffrast
    rast_out, _ = dr.rasterize(glctx, verts_clip, faces, resolution=resolution)
    valid = (rast_out[..., -1] > 0).reshape(H, W)
    triangle_id = (rast_out[..., -1] - 1).long().reshape(H, W)
    return valid, triangle_id

def filter_mesh_from_faces(keep, mesh_points, faces):

    keep_verts_idxs = faces[keep].reshape(-1)

    keep = np.zeros((mesh_points.shape[0])) > 0
    keep[keep_verts_idxs] = True

    filter_mapping = np.arange(keep.shape[0])[keep]
    filter_unmapping = -np.ones((keep.shape[0]))

    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    mesh_points = mesh_points[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)
    faces = faces[keep_faces]
    face_mapping = np.arange(keep_faces.shape[0])[keep_faces]
    faces[:, 0] = filter_unmapping[faces[:, 0]]
    faces[:, 1] = filter_unmapping[faces[:, 1]]
    faces[:, 2] = filter_unmapping[faces[:, 2]]
    return mesh_points, faces, face_mapping

def filter_mesh_from_vertices(keep, mesh_points, faces, tex_pos):
    filter_mapping = np.arange(keep.shape[0])[keep]
    filter_unmapping = -np.ones((keep.shape[0]))
    filter_unmapping[filter_mapping] = np.arange(filter_mapping.shape[0])
    mesh_points = mesh_points[keep]
    keep_0 = keep[faces[:, 0]]
    keep_1 = keep[faces[:, 1]]
    keep_2 = keep[faces[:, 2]]
    keep_faces = np.logical_and(keep_0, keep_1)
    keep_faces = np.logical_and(keep_faces, keep_2)
    faces = faces[keep_faces]
    if tex_pos is not None:
        tex_pos = tex_pos[keep_faces]
    face_mapping = np.arange(keep_faces.shape[0])[keep_faces]
    faces[:, 0] = filter_unmapping[faces[:, 0]]
    faces[:, 1] = filter_unmapping[faces[:, 1]]
    faces[:, 2] = filter_unmapping[faces[:, 2]]
    return mesh_points, faces, face_mapping, tex_pos

def visualize_rasterization(valid, triangle_id):
    """
    Visualize rasterization results
    :param valid: Valid pixel mask (H, W) boolean array
    :param triangle_id: Triangle ID map (H, W) integer array
    """
    # Convert to numpy array (if torch.Tensor)
    if isinstance(valid, torch.Tensor):
        valid = valid.cpu().numpy()
    if isinstance(triangle_id, torch.Tensor):
        triangle_id = triangle_id.cpu().numpy()
    
    H, W = valid.shape
    # Create RGB image (H, W, 3)
    image = np.zeros((H, W, 3), dtype=np.float32)
    
    if np.any(valid):
        # Get triangle IDs of the valid region
        visible_ids = triangle_id[valid]
        
        # Normalize IDs to [0,1] range
        min_id = np.min(visible_ids)
        max_id = np.max(visible_ids)
        if max_id > min_id:
            norm_ids = (visible_ids - min_id) / (max_id - min_id)
        else:  # Avoid division by 0
            norm_ids = np.zeros_like(visible_ids, dtype=np.float32)
        
        # Map colors using viridis colormap
        colors = cm.viridis(norm_ids)[:, :3]  # Get RGB, ignore alpha channel
        image[valid] = colors
    
    # Visualize
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title('Rasterization Visualization')
    plt.axis('off')
    plt.show()


def visualize_camera_o3d(c2w,
                         intrinsic,
                         ply_path: str,
                         image_size: tuple = (640, 480),
                         near: float = 0.1,
                         far: float = 1.0,
                         scale: float = 0.1):
    """
    Use Open3D to visualize the camera frustum and coordinate axes in the PLY scene.

    Arguments:
      - c2w: (4,4) Camera-to-world transformation matrix, supports numpy array or torch tensor
      - intrinsic: (3,3) Camera intrinsic matrix, supports numpy array or torch tensor
      - ply_path: PLY file path of the scene
      - image_size: (width, height) Image plane size
      - near, far: Near and far plane distances
      - scale: Coordinate axis length scale
    """
    # Convert to numpy
    if hasattr(c2w, 'cpu'):
        c2w_np = c2w.cpu().numpy()
    else:
        c2w_np = np.array(c2w)
    if hasattr(intrinsic, 'cpu'):
        K = intrinsic.cpu().numpy()
    else:
        K = np.array(intrinsic)

    # Load scene
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()

    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    W, H = image_size
    # Four-corner pixel coordinates
    pts_pix = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]])
    inv_K = np.linalg.inv(K)

    # Near and far plane vertices
    pts_cam_near = (inv_K @ pts_pix.T).T * near
    pts_cam_far  = (inv_K @ pts_pix.T).T * far

    # Concatenate and transform to world
    verts_cam = np.vstack([pts_cam_near, pts_cam_far])    # (8,3)
    verts_h = np.hstack([verts_cam, np.ones((8,1))])     # (8,4)
    verts_world = (c2w_np @ verts_h.T).T[:, :3]          # (8,3)

    # Define frustum edge connections
    edges = [(0,1),(1,2),(2,3),(3,0),  # near
             (4,5),(5,6),(6,7),(7,4),  # far
             *[(i, i+4) for i in range(4)]
            ]

    # Build LineSet
    frustum_ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(verts_world),
        lines=o3d.utility.Vector2iVector(edges)
    )
    frustum_ls.colors = o3d.utility.Vector3dVector([[1,0,0] for _ in edges])

    # Camera coordinate axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    axes.transform(c2w_np)

    # Add geometry and render
    vis.add_geometry(frustum_ls)
    vis.add_geometry(axes)
    vis.run()
    vis.destroy_window()

def compute_mvp(c2w: np.ndarray,
                K: np.ndarray,
                width: int,
                height: int,
                near: float,
                far: float) -> np.ndarray:
    """
    Compute OpenGL style MVP matrix.

    Arguments:
        c2w   -- 4x4 camera to world coordinate transformation matrix
        K     -- 3x3 camera intrinsic matrix
        width -- viewport width (pixels)
        height-- viewport height (pixels)
        near  -- near clipping plane
        far   -- far clipping plane

    Returns:
        MVP   -- 4x4 MVP matrix
    """
    # 1. View Matrix: World to camera coordinates
    V = np.linalg.inv(c2w)

    # 2. Expand K to 4x4
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    K4 = np.array([
        [fx,  0, cx, 0],
        [ 0, fy, cy, 0],
        [ 0,  0,  1, 0],
        [ 0,  0,  0, 1]
    ], dtype=np.float32)

    # 3. Construct viewport transformation from pixels to NDC and depth mapping
    #    x_ndc =  2*u/W - 1
    #    y_ndc =  1 - 2*v/H
    #    z_ndc = (f + n)/(n - f) + (2 f n)/(Z_c (n - f))
    T = np.eye(4, dtype=np.float32)
    T[0, 0] =  2.0 / width
    T[1, 1] = -2.0 / height
    T[3, 0] = -1.0
    T[3, 1] =  1.0

    D = np.eye(4, dtype=np.float32)
    D[2, 2] = -2.0 / (far - near)
    D[3, 2] = -(far + near) / (far - near)

    # Projection = D * T * K4
    P = D @ T @ K4

    # 4. MVP
    MVP = P @ V  # Because Model = I
    return MVP

def build_opengl_proj_from_intrinsics(K, W, H, near=0.1, far=3.0):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # Map pixel coordinate system to NDC (-1...+1)
    # x_ndc = (u/cx - 1) etc, after derivation:
    A =  2.0 * fx / W
    B =  2.0 * fy / H
    C =  2.0 * (cx / W) - 1.0
    D = -2.0 * (cy / H) + 1.0

    # Depth mapping
    E = -(far + near) / (far - near)
    F = -2.0 * far * near / (far - near)

    M = torch.zeros((4,4), dtype=torch.float32, device=K.device)
    M[0,0] = A
    M[0,2] = C
    M[1,1] = B
    M[1,2] = D
    M[2,2] = E
    M[2,3] = F
    M[3,2] = -1.0
    return M


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--mesh_path', type=str, required=True)
    parser.add_argument('--num_max_frames', type=int, default=1000)
    parser.add_argument('--num_faces_simplified', type=int, default=80000)
    parser.add_argument('--min_cooccurrence_count', type=int, default=3, help='Minimum required edge updates to be considered valid')
    args = parser.parse_args()
    mask_dir = os.path.join(args.data_dir, 'grounded_sam')
    pose_path = os.path.join(args.data_dir, 'pose_intrinsic_imu_mvp.json')
    save_dir = os.path.join(args.data_dir, 'perception', 'vis_groups')
    image_dir = args.image_dir
    mesh_path = args.mesh_path
    mask_data_dir = os.path.join(mask_dir, 'mask_data')
    mask_save_path = os.path.join(mask_dir, 'separate_vis_gsam.pkl')

    glctx = dr.RasterizeGLContext(output_db=False)
    os.makedirs(save_dir, exist_ok=True)

    H, W = Image.open(os.path.join(image_dir, os.listdir(image_dir)[0])).size[::-1]
    resolution_raw = (H, W)
    while H % 8 != 0 or W % 8 != 0:
        H *= 2; W *= 2
    resolution = (H, W)

    mesh = trimesh.load(mesh_path)
    # mesh_scene = trimesh.Scene(mesh)
    mesh_verts_np = mesh.vertices
    mesh_faces_np = mesh.faces

    m = pml.Mesh(mesh_verts_np, mesh_faces_np)
    ms = pml.MeshSet(); ms.add_mesh(m)
    ms.simplification_quadric_edge_collapse_decimation(targetfacenum=args.num_faces_simplified, optimalplacement=False)
    m = ms.current_mesh()
    mesh_verts_np = m.vertex_matrix(); mesh_faces_np = m.face_matrix()

    mesh_verts = torch.from_numpy(mesh_verts_np).float().cuda().contiguous()
    mesh_faces = torch.from_numpy(mesh_faces_np).int().cuda().contiguous()

    mesh_simpl = trimesh.Trimesh(
        vertices=mesh_verts_np,
        faces=mesh_faces_np,
        process=False
    )

    # create all_mask_dict
    print('create all_mask_dict...')
    stem_list = [fname[:-4] for fname in sorted(os.listdir(mask_data_dir))]

    def _load_stem_masks(stem):
        with open(os.path.join(mask_dir, 'mask', stem + '.json'), 'r') as f:
            mask_json = json.load(f)
        mask_data = pickle.load(open(os.path.join(mask_data_dir, stem + '.pkl'), 'rb'))['mask']
        stem_dict = {}
        for entry in mask_json:
            val = entry['value']
            if val == 0:
                continue
            idx = val - 1
            mask = mask_data[idx].reshape(resolution_raw).cpu().numpy() > 0
            stem_dict[val] = mask
        return stem, stem_dict

    all_mask_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(_load_stem_masks, stem): stem for stem in stem_list}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            stem, stem_dict = fut.result()
            all_mask_dict[stem] = stem_dict
    # save masks with highest protocol to reduce memory
    print('saving all_mask_dict...')
    with open(mask_save_path, 'wb') as f:
        pickle.dump(all_mask_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("all_mask_dict: ", all_mask_dict.keys())

    # load poses
    with open(pose_path, 'r') as f:
        pose_dict = json.load(f)
    stem_list = sorted(k for k in all_mask_dict if k in pose_dict)
    # downsample frames evenly without numpy
    if len(stem_list) > args.num_max_frames:
        step = len(stem_list) // args.num_max_frames
        stem_list = stem_list[::step][:args.num_max_frames]
    print('start processing frames...')
    
    subprocess.run(f"rm {save_dir}/*", shell=True)
    
    # build graph for mesh
    G = nx.Graph()
    num_faces = mesh_faces_np.shape[0]
    # Store compact faces_idxs per mask (instead of O(n^2) expanded pair lists)
    faces_idxs_per_mask = []
    # Batch-accumulate pair indices into sparse matrix (avoids Python for-loop over pairs)
    _FLUSH_LIMIT = 10_000_000
    _batch_rows = []
    _batch_cols = []
    _batch_count = [0]
    _edge_cm = [csr_matrix((num_faces, num_faces), dtype=np.int32)]

    def _flush_edge_batch():
        if _batch_count[0] == 0:
            return
        r = np.concatenate(_batch_rows)
        c = np.concatenate(_batch_cols)
        d = np.ones(len(r), dtype=np.int32)
        _edge_cm[0] = _edge_cm[0] + coo_matrix((d, (r, c)), shape=(num_faces, num_faces)).tocsr()
        _batch_rows.clear()
        _batch_cols.clear()
        _batch_count[0] = 0

    for stem in tqdm(stem_list):
        mask_idxs = sorted(list(all_mask_dict[stem].keys()))
        
        frame_info = pose_dict[stem]
        # c2w = pose["c2w"]
        # mvp = pose["mvp"]
        # camproj = pose["camproj"]

        # adapt scannet++ pose info to DRAWER format
        c2w = np.array(frame_info["aligned_pose"])
        camproj = np.array(frame_info["intrinsic"])
        mvp = np.array(frame_info["mvp"])

        # mvp = compute_mvp(c2w, camproj, resolution[1], resolution[0], near=0.1, far=3)

        # mvp = torch.from_numpy(mvp).cuda().float()
        # visualize_camera_view(mesh_path, c2w, camproj)
        c2w = torch.from_numpy(c2w).cuda().float()
        camproj = torch.from_numpy(camproj).cuda().float()
        mvp = torch.from_numpy(mvp).cuda().float()
        # visualize_camera_o3d(c2w, camproj, mesh_path, image_size=(W,H), near=0.1, far=3.0, scale=0.1)

        # T_world2cam = torch.inverse(c2w)

        # # print("W, H: ", W, H)

        # flip_z = torch.diag(torch.tensor([1,1,-1,1], dtype=T_world2cam.dtype, device=T_world2cam.device))
        # T_eyeGL  = flip_z @ T_world2cam
        # proj4 = build_opengl_proj_from_intrinsics(camproj, W, H, near=0.1, far=3.0)
        # MVP = proj4 @ T_eyeGL
        # P4_tensor = MVP.float().cuda()

        # P4_tensor = compute_projection(c2w, camproj, W, H)

        valid, triangle_id = rasterize_texture(mesh_verts, mesh_faces, mvp, glctx, resolution)
        # valid, triangle_id = rasterize_mesh(mesh_verts, mesh_faces, c2w, camproj, resolution, None)
        # valid, triangle_id = rasterize_texture_custom(mesh_verts, mesh_faces, c2w, camproj, glctx, resolution)
        # print("valid: ", valid, "triangle_id: ", triangle_id)

        # **VISUALIZATION** overlay mask on original image
        # print(f"visualizing {stem} rasterization results...")
        # visualize_rasterization(valid, triangle_id)
        
        
        H, W = resolution[0], resolution[1]
        
        
        for mask_idx in mask_idxs:
            binary_mask = all_mask_dict[stem][mask_idx]

            binary_mask = binary_erosion(binary_mask, iterations=4)

            # binary_mask = torchvision.transforms.functional.resize(torch.from_numpy(binary_mask).unsqueeze(0) > 0, (H, W), torchvision.transforms.InterpolationMode.NEAREST).cpu().numpy()
            binary_mask = torchvision.transforms.functional.resize(torch.from_numpy(binary_mask).unsqueeze(0) > 0, (H, W), torchvision.transforms.InterpolationMode.NEAREST).squeeze(0).cpu().numpy()

            faces_idxs = triangle_id[binary_mask][valid[binary_mask]].cpu().numpy().astype(np.int32)

            # **VISUALIZATION** highlight faces in mesh
            # print("frame: ", stem, "mask_idx: ", mask_idx)
            # face_colors = np.zeros((mesh_simpl.faces.shape[0], 4), dtype=np.uint8)
            # face_colors[:] = [200, 200, 200, 255]        # RGBA: Default light gray
            # face_colors[faces_idxs] = [255, 0, 0, 255]    # RGBA: Highlight red

            # mesh_simpl.visual.face_colors = face_colors
            # mesh_simpl.show()

            assert np.all(faces_idxs >= 0)
            
            # Sort vertices to ensure deterministic ordering
            faces_idxs = np.unique(faces_idxs)
            faces_idxs = np.sort(faces_idxs)

            # Store compact array (NOT expanded O(n^2) pair list)
            faces_idxs_per_mask.append(faces_idxs)

            # Generate pair indices via numpy (same order as combinations)
            n = len(faces_idxs)
            if n >= 2:
                i_idx, j_idx = np.triu_indices(n, k=1)
                _batch_rows.append(faces_idxs[i_idx])
                _batch_cols.append(faces_idxs[j_idx])
                _batch_count[0] += len(i_idx)
                if _batch_count[0] >= _FLUSH_LIMIT:
                    _flush_edge_batch()

            # for u, v in combinations(faces_idxs, 2):
            #     edge = (u, v)
            #     if edge in edge_updates:
            #         edge_updates[edge] += 1
            #     else:
            #         edge_updates[edge] = 1

    _flush_edge_batch()
    edge_count_matrix = _edge_cm[0]
    del _batch_rows, _batch_cols, _edge_cm
    print("edge_count_matrix nnz: ", edge_count_matrix.nnz)

    edge_filtered_updates = defaultdict(int)

    print("pairs_list length: ", len(faces_idxs_per_mask))
    for faces_idxs in tqdm(faces_idxs_per_mask):
        n = len(faces_idxs)
        if n < 2:
            continue
        # Regenerate pair indices (same order as combinations)
        i_idx, j_idx = np.triu_indices(n, k=1)
        u_arr = faces_idxs[i_idx]
        v_arr = faces_idxs[j_idx]

        # Vectorized threshold check via sparse matrix lookup
        counts = np.asarray(edge_count_matrix[u_arr, v_arr]).ravel()
        threshold_mask = counts >= args.min_cooccurrence_count
        u_f = u_arr[threshold_mask]
        v_f = v_arr[threshold_mask]
        pairs_sample = list(zip(u_f.tolist(), v_f.tolist()))

        Max_P = 1000
        if len(pairs_sample) > Max_P:
            pairs_sample = random.sample(pairs_sample, Max_P)

        for u, v in pairs_sample:
            edge_filtered_updates[(u, v)] += 1

                
    cnt = 0
    for edge, weight in tqdm(edge_filtered_updates.items()):
        nvis = 1.0
        G.add_edge(edge[0], edge[1], weight=weight / nvis)
        cnt += 1
        
    print("Total edges in graph: ", cnt)


    potential_meshes = []
    faces_idxs_list = []
    potential_i = 0
    
    res_list = [0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 20.0, 35.0, 50.0, 75.0, 100.0, 200.0, 350.0, 500.0]

    def _process_resolution(res):
        """Run Louvain + mesh extraction for a single resolution. Returns list of (mask_mesh, faces_idxs)."""
        partition = nx.community.louvain_communities(G, seed=42, resolution=res)
        parts = []
        for faces_idxs in partition:
            faces_idxs = np.array(list(faces_idxs)).reshape(-1).astype(np.int32)
            if faces_idxs.shape[0] > 10:
                parts.append(faces_idxs)
        print("res: ", res, "total partition: ", len(parts))

        results = []
        for faces_idxs in parts:
            mesh_mask_verts_np, mesh_mask_faces_np, face_map_0 = filter_mesh_from_faces(faces_idxs, mesh_verts_np, mesh_faces_np)

            faces_idxs = face_map_0
            mask_mesh = trimesh.Trimesh(
                mesh_mask_verts_np,
                mesh_mask_faces_np,
                process=False
            )

            # mask_mesh.show()

            edges = mask_mesh.edges_sorted.reshape((-1, 2))
            components = trimesh.graph.connected_components(edges, min_len=1, engine='scipy')
            largest_cc = np.argmax(np.array([comp.shape[0] for comp in components]).reshape(-1), axis=0)
            keep = np.zeros((mesh_mask_verts_np.shape[0])).astype(np.bool_)
            keep[components[largest_cc].reshape(-1)] = True
            _, _, face_map_1, _ = filter_mesh_from_vertices(keep, mesh_mask_verts_np, mesh_mask_faces_np, None)

            faces_idxs = faces_idxs[face_map_1]
            results.append((mask_mesh, faces_idxs))
        return results

    with concurrent.futures.ThreadPoolExecutor() as executor:
        res_results = list(executor.map(_process_resolution, res_list))

    for res_result in res_results:
        for mask_mesh, faces_idxs in res_result:
            potential_meshes.append(mask_mesh)
            trimesh.exchange.export.export_mesh(
                mask_mesh,
                os.path.join(save_dir, f"mask_mesh_{potential_i:0>3d}.ply")
            )
            faces_idxs_list.append(faces_idxs)
            potential_i += 1
            
    with open(os.path.join(save_dir, f"faces_idxs_list.pkl"), 'wb') as f:
        pickle.dump(faces_idxs_list, f)

