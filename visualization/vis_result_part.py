import open3d as o3d
import sys
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scene_dir', type=str, required=True, help='Path to the output result dir')
args = parser.parse_args()

result_dir = args.scene_dir

# original_mesh_path = f"/home/troye/ssd/datasets/scannet++/data/{scene_id}/scans/mesh_aligned_0.05.ply"
# original_mesh = o3d.io.read_triangle_mesh(original_mesh_path)

mesh_list = []

def create_arrow_from_vector(origin: np.ndarray,
                             vector: np.ndarray,
                             color: tuple = (1.0, 0.0, 0.0),
                             cylinder_radius: float = 0.01,
                             cone_radius: float = 0.02,
                             cylinder_height_ratio: float = 0.8,
                             cone_height_ratio: float = 0.2) -> o3d.geometry.TriangleMesh:
    """
    Create an arrow based on origin and vector.

    Parameters
    ----------
    origin : np.ndarray, shape (3,)
        Arrow origin coordinates.
    vector : np.ndarray, shape (3,)
        Arrow vector (vector pointing from origin to endpoint).
    cylinder_radius : float
        Radius of the cylinder.
    cone_radius : float
        Radius of the cone base.
    cylinder_height_ratio : float
        Ratio of cylinder height to total length (0 < ratio < 1).
    cone_height_ratio : float
        Ratio of cone height to total length (0 < ratio < 1), sum is 1.

    Returns
    -------
    arrow : o3d.geometry.TriangleMesh
        The constructed arrow mesh, ready to be added to the scene.
    """
    # 1. Calculate length and direction
    length = np.linalg.norm(vector)
    if length == 0:
        raise ValueError("Vector length is zero, cannot create arrow.")
    direction = vector / length

    # 2. Divide the length of the cylinder and cone proportionally
    cylinder_height = length * cylinder_height_ratio
    cone_height = length * cone_height_ratio

    # 3. Create default arrow along +Z axis
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height
    )
    arrow.compute_vertex_normals()

    # 4. Calculate rotation matrix to align +Z axis to direction
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        # Case of +Z or -Z
        if np.dot(z_axis, direction) < 0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([1, 0, 0] * np.pi)
        else:
            R = np.eye(3)
    else:
        axis = axis / axis_len
        angle = np.arccos(np.dot(z_axis, direction))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    # 5. Rotate and translate to origin
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(origin)

    arrow.paint_uniform_color(color)

    return arrow

remain_scene_path = os.path.join(result_dir, "remain_scene.ply")
if os.path.exists(remain_scene_path):
    remain_scene = o3d.io.read_triangle_mesh(remain_scene_path)
    if remain_scene.has_vertices():
        mesh_list.append(remain_scene)
    else:
        print(f"remain_scene.ply has no vertices: {remain_scene_path}")
eps = 1e-6
inference_arti_dict = {}

for subdir in os.listdir(result_dir):
    subdir_path = os.path.join(result_dir, subdir)
    # if not subdir.endswith(f'{single}'):
    #     continue
    if os.path.isdir(subdir_path):
        ply_file = os.path.join(subdir_path, "part.ply")
        base_ply_file = os.path.join(subdir_path, "base.ply")
        if os.path.exists(ply_file):
            mesh = o3d.io.read_triangle_mesh(ply_file)
            base_mesh = o3d.io.read_point_cloud(base_ply_file)
            if not mesh.is_empty():
                color = np.random.rand(3,)   # random RGB in [0,1]
                mesh.paint_uniform_color(color)
                mesh_list.append(mesh)
                # mesh_list.append(base_mesh)   
                # if original_mesh.has_vertices():
                #     orig_pts = np.asarray(original_mesh.vertices)
                #     part_pts = np.asarray(mesh.vertices)

                #     # Use KDTree to find points with distance less than eps
                #     part_pcd = o3d.geometry.PointCloud()
                #     part_pcd.points = o3d.utility.Vector3dVector(part_pts)
                #     kdtree = o3d.geometry.KDTreeFlann(part_pcd)

                #     keep_mask = np.ones(len(orig_pts), dtype=bool)
                #     for i, p in enumerate(orig_pts):
                #         [_, idx, dist] = kdtree.search_knn_vector_3d(p, 1)
                #         if dist[0] < eps**2:  # Note that search_knn_vector_3d returns squared distance
                #             keep_mask[i] = False

                #     if not np.all(keep_mask):
                #         original_mesh = original_mesh.select_by_index(np.where(keep_mask)[0])
            else:
                print(f"Failed to load file: {ply_file}")
            
            joint_type_path = os.path.join(subdir_path, 'articulation_type.npy')
            origin_world_path = os.path.join(subdir_path, 'origin_world.npy')
            vector_world_path = os.path.join(subdir_path, 'vector_world.npy')
            if os.path.isfile(origin_world_path) and os.path.isfile(vector_world_path):
                origin_world = np.load(origin_world_path)

                vec = np.load(vector_world_path)
                norm = np.linalg.norm(vec)
                vector_world = vec / norm if norm > 0 else vec
                joint_type = np.load(joint_type_path).item()

                inference_arti_dict[subdir] = {
                    'origin': origin_world,
                    'axis': vector_world,
                    'type': joint_type
                }

# Color palette
palette = np.array([
    [0.90, 0.10, 0.10],
    [0.10, 0.60, 0.95],
    [0.10, 0.75, 0.30],
    [0.95, 0.70, 0.10],
    [0.60, 0.20, 0.80],
    [0.20, 0.80, 0.80],
])

arrow_list = []
for i, (name, info) in enumerate(inference_arti_dict.items()):
    origin = info.get("origin", [0, 0, 0])
    axis   = info.get("axis",   [0, 0, 1])
    joint_type = info.get("type", "unknown")
    # color  = palette[i % len(palette)]
    if joint_type == 'rotation':
        color = [0.8, 0.1, 0.1]  # red
    elif joint_type == 'translation':
        color = [0.0, 0.0, 1.0]  # blue
    # scale=1.0: Strictly use the length of the axis vector itself; increase scale to make it more prominent
    ls = create_arrow_from_vector(origin, axis, color=color)
    if ls is not None:
        arrow_list.append(ls)

frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
o3d.visualization.draw_geometries([frame] + mesh_list + arrow_list)