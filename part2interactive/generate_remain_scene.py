import os
import open3d as o3d
import numpy as np
from pathlib import Path
import argparse
from utilities import save_time_breakdown
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="PLY path of the scene mesh")
    parser.add_argument("--part_dir", required=True, help="Contains multiple subdirectories, each with part.ply and obb_world.npy")
    parser.add_argument("--vis", action="store_true", help="Visualize each OBB")
    args = parser.parse_args()

    scene_mesh_path = args.scene
    part_mesh_dir = Path(args.part_dir)
    save_path = os.path.join(part_mesh_dir, "remain_scene.ply")
    save_obj_path = os.path.join(part_mesh_dir, "remain_scene.obj")

    # start timer
    start_time = time.time()
    
    # === collect first-level subdirs that contain both files ===
    subdirs = [p for p in part_mesh_dir.iterdir() if p.is_dir()]
    pair_list = []
    for d in sorted(subdirs):
        part_ply = d / "part.ply"
        obb_npy = d / "obb_world.npy"
        if part_ply.exists() and obb_npy.exists():
            pair_list.append((part_ply, obb_npy))
    if not pair_list:
        raise RuntimeError(f"No subdir under {part_mesh_dir} contains both 'part.ply' and 'obb_world.npy'.")

    # === read scene mesh and its vertices ===
    scene = o3d.io.read_triangle_mesh(scene_mesh_path)
    if not scene.has_vertices():
        raise RuntimeError("Scene mesh has no vertices.")
    V = np.asarray(scene.vertices)  # (N,3)

    # point cloud view of scene vertices (for OBB point selection)
    pcd_scene = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V))

    # === accumulate indices to remove (inside any OBB) ===
    to_remove = np.zeros(len(V), dtype=bool)
    obb_count = 0

    for part_ply, obb_npy in pair_list:
        # read OBB corners (8x3)
        corners = np.load(str(obb_npy))
        if corners.ndim != 2 or corners.shape[1] != 3:
            print(f"[WARN] {obb_npy} is not of shape (8,3). Skipping.")
            continue

        # create O3D OBB from the 8 corners
        obb = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(corners.astype(np.float64))
        )

        # ---- visualization for this OBB (optional) ----
        if args.vis:
            # Try to draw OBB as wireframe
            try:
                obb_lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            except AttributeError:
                # Fallback: build LineSet from corners
                p = np.asarray(obb.get_box_points())
                lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
                obb_lines = o3d.geometry.LineSet()
                obb_lines.points = o3d.utility.Vector3dVector(p)
                obb_lines.lines  = o3d.utility.Vector2iVector(lines)
            # color lines
            obb_lines.colors = o3d.utility.Vector3dVector([[1,0,0]] * len(obb_lines.lines))
            o3d.visualization.draw_geometries(
                [scene, obb_lines],
                window_name=f"OBB preview: {Path(obb_npy).parent.name}"
            )

        # get indices of scene points inside this OBB
        idx_in = obb.get_point_indices_within_bounding_box(pcd_scene.points)
        if idx_in:
            to_remove[idx_in] = True
        obb_count += 1

    if not to_remove.any():
        print("No scene vertices fell inside any OBB; saving original scene.")
        o3d.io.write_triangle_mesh(save_path, scene)
        print(f"Saved: {save_path} | vertices={len(scene.vertices)} | faces={len(scene.triangles)}")
        print(f"Processed OBBs: {obb_count}, subdirs: {len(pair_list)}")
        return

    # === remove vertices inside any OBB (faces connected will be dropped automatically) ===
    scene.remove_vertices_by_mask(to_remove)

    o3d.io.write_triangle_mesh(save_path, scene)
    # o3d.io.write_triangle_mesh(save_obj_path, scene)
    print(f"Saved: {save_path} | vertices={len(scene.vertices)} | faces={len(scene.triangles)}")
    print(f"Processed OBBs: {obb_count}, subdirs: {len(pair_list)}")
    print(f"Removed vertices: {int(to_remove.sum())}")

    # end timer
    end_time = time.time()
    duration = end_time - start_time

    # --- TIME BREAKDOWN LOGGING ---
    save_time_breakdown(
        input_dir=part_mesh_dir,
        output_dir=part_mesh_dir,
        module_name="Module 5.2: Remaining Scene Generation",
        start_time=start_time,
        end_time=end_time,
        duration=duration
    )

if __name__ == "__main__":
    main()
