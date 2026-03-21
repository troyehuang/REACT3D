import open3d as o3d
import numpy as np
import os
import pickle


def find_largest_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    Run RANSAC plane segmentation on the point cloud.
    Returns plane_model and inlier indices.
    """
    model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations)
    return model, inliers


def compute_camera_direction(extrinsic_c2w):
    """
    Given a 4x4 extrinsic_c2w matrix (camera->world), compute the camera's viewing direction
    in world coordinates assuming the camera looks along its local +Z axis.
    Also returns camera position in world coords.
    """
    E = np.array(extrinsic_c2w).reshape(4, 4)
    R_c2w = E[:3, :3]
    t_c2w = E[:3, 3]
    camera_dir = R_c2w[:, 2]
    camera_dir = camera_dir / np.linalg.norm(camera_dir)
    camera_pos = t_c2w
    return camera_dir, camera_pos


def find_largest_vertical_plane(ply_path,
                                extrinsic_c2w,
                                distance_threshold=0.01,
                                ransac_n=3,
                                num_iterations=1000,
                                vertical_threshold_deg=10):
    """
    Load PLY, estimate normals, filter for vertical plane candidates,
    fit the largest plane, then split points into plane_cloud and rest_cloud
    depending on the camera direction and positive normal of the plane.

    extrinsic_c2w: 4x4 camera-to-world matrix

    Returns:
    - plane_cloud: open3d.geometry.PointCloud of plane points
    - plane_model: [a, b, c, d] with normal pointing toward camera
    - rest_cloud: open3d.geometry.PointCloud of points beyond the plane along positive normal
    - camera_dir: normalized viewing direction
    - camera_pos: position in world coords
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        raise ValueError(f"No points in {ply_path}")

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    normals = np.asarray(pcd.normals)

    vertical = np.array([0, 0, 1])
    cos_max = np.sin(np.deg2rad(vertical_threshold_deg))
    mask = np.abs(normals.dot(vertical)) <= cos_max
    candidate_idx = np.where(mask)[0].tolist()
    if len(candidate_idx) < ransac_n:
        return None, None, None, None, None, None
    candidate_cloud = pcd.select_by_index(candidate_idx)

    model, inliers_sub = find_largest_plane(
        candidate_cloud,
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations)
    inliers = [candidate_idx[i] for i in inliers_sub]

    camera_dir, camera_pos = compute_camera_direction(extrinsic_c2w)

    n = np.array(model[:3])
    norm_n = np.linalg.norm(n)
    n_normed = n / norm_n
    d_normed = model[3] / norm_n
    if np.dot(n_normed, camera_dir) < 0:
        n_normed = -n_normed
        d_normed = -d_normed

    pts = np.asarray(pcd.points)
    projections = pts.dot(n_normed)
    inlier_proj = projections[inliers]
    max_inlier_proj = inlier_proj.max()

    rest_idx = np.where(projections > max_inlier_proj)[0].tolist()
    plane_idx = np.where(projections <= max_inlier_proj)[0].tolist()

    plane_cloud = pcd.select_by_index(plane_idx)
    rest_cloud = pcd.select_by_index(rest_idx)

    plane_model = [*n_normed.tolist(), d_normed]
    return plane_cloud, plane_model, rest_cloud, camera_dir, camera_pos, plane_idx

def extract_and_save_plane_mesh(orig_mesh_path, plane_idx, save_path):
    """
    从原始三角网格中提取出 plane_idx 顶点对应的子网格（包含所有三个顶点都在 plane_idx 中的三角面），
    并保存为 PLY。
    """
    # 1. 读入原始 mesh
    mesh = o3d.io.read_triangle_mesh(orig_mesh_path)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    has_colors = False
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        has_colors = True

    # 2. 构建一个快速查表：old_idx -> new_idx
    idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(plane_idx)}

    # 3. 找出所有三个顶点都在 plane_idx 集合中的三角面
    mask = np.all(np.isin(tris, plane_idx), axis=1)
    tris_sel = tris[mask]

    # 4. 重新编号三角面顶点索引
    tris_new = np.array([[idx_map[v] for v in tri] for tri in tris_sel], dtype=np.int32)

    # 5. 构建子网格并保存
    mesh_plane = o3d.geometry.TriangleMesh()
    mesh_plane.vertices = o3d.utility.Vector3dVector(verts[plane_idx])
    mesh_plane.triangles = o3d.utility.Vector3iVector(tris_new)
    mesh_plane.compute_vertex_normals()
    if has_colors:
        mesh_plane.vertex_colors = o3d.utility.Vector3dVector(colors[plane_idx])
    o3d.io.write_triangle_mesh(save_path, mesh_plane)
    print(f"Saved plane mesh: {save_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find and split the largest vertical plane and visualize normal & camera directions.")
    parser.add_argument("--data_dir", "-d", required=True, help="data directory")
    parser.add_argument("--threshold", "-t", type=float, default=0.02,
                        help="Max distance threshold for plane fitting")
    parser.add_argument("--iterations", "-n", type=int, default=1000,
                        help="Number of RANSAC iterations")
    parser.add_argument("-v", "--vertical_thresh", type=float, default=45,
                        help="Max deviation deg from vertical")
    args = parser.parse_args()

    data_dir = args.data_dir
    real_dir = os.path.join(data_dir, 'perception', 'vis_groups_final_mesh', 'real')

    for sub_dir in os.listdir(real_dir):
        sub_path = os.path.join(real_dir, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        
        print(f"Processing {sub_path}...")
        ply_path = os.path.join(sub_path, 'mask_mesh.ply')
        top_k_path = os.path.join(sub_path, 'top_k_list.pkl')
        new_ply_path = os.path.join(sub_path, 'mask_mesh_refined.ply')

        with open(top_k_path, 'rb') as f:
            top_k_list = pickle.load(f)
        extrinsic_c2w = top_k_list[0]["extrinsic"]
        print("frame: ", top_k_list[0]["frame_name"])

        plane_cloud, plane_model, rest_cloud, camera_dir, camera_pos, plane_idx = find_largest_vertical_plane(
            ply_path,
            extrinsic_c2w,
            distance_threshold=args.threshold,
            ransac_n=3,
            num_iterations=args.iterations,
            vertical_threshold_deg=args.vertical_thresh
        )
        
        if plane_cloud is None:
            print(f"No valid plane found in {ply_path}. Skipping...")
            continue

        extract_and_save_plane_mesh(ply_path, plane_idx, new_ply_path)

        plane_cloud.paint_uniform_color([1.0, 0.0, 0.0])  # red
        rest_cloud.paint_uniform_color([0.7, 0.7, 0.7])  # gray

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Plane normal arrow (red)
        n = np.array(plane_model[:3])
        normal_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005,
            cone_radius=0.01,
            cylinder_height=0.2,
            cone_height=0.05)
        # Align
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, n)
        c = np.dot(z_axis, n)
        v_mat = np.array([[    0, -v[2],  v[1]],
                          [ v[2],     0, -v[0]],
                          [-v[1],  v[0],    0]])
        Rn = np.eye(3) + v_mat + v_mat.dot(v_mat) * ((1 - c) / (np.linalg.norm(v)**2 + 1e-8))
        normal_arrow.rotate(Rn, center=np.zeros(3))
        centroid = np.mean(np.asarray(plane_cloud.points), axis=0)
        normal_arrow.translate(centroid)
        normal_arrow.paint_uniform_color([1.0, 0.0, 0.0])

        # Camera direction arrow (green)
        cam_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.005,
            cone_radius=0.01,
            cylinder_height=0.2,
            cone_height=0.05)
        v2 = np.cross(z_axis, camera_dir)
        c2 = np.dot(z_axis, camera_dir)
        v2_mat = np.array([[    0, -v2[2],  v2[1]],
                           [ v2[2],     0, -v2[0]],
                           [-v2[1],  v2[0],    0]])
        Rc = np.eye(3) + v2_mat + v2_mat.dot(v2_mat) * ((1 - c2) / (np.linalg.norm(v2)**2 + 1e-8))
        cam_arrow.rotate(Rc, center=np.zeros(3))
        cam_arrow.translate(camera_pos)
        cam_arrow.paint_uniform_color([0.0, 1.0, 0.0])

        # # Visualize
        # o3d.visualization.draw_geometries([
        #     plane_cloud,
        #     rest_cloud,
        #     axis,
        #     normal_arrow,
        #     cam_arrow
        # ],
        # window_name="Plane & Camera Directions",
        # width=800,
        # height=600)

