import os
import numpy as np
import open3d as o3d
import time
import viser
import trimesh
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--scene_dir', type=str, required=True, help='Path to the output result dir')
args = parser.parse_args()

# ================= Configuration Paths =================
result_dir = args.scene_dir

# ================= Helper Functions =================
def create_arrow_from_vector(origin: np.ndarray,
                             vector: np.ndarray,
                             color: tuple = (1.0, 0.0, 0.0),
                             cylinder_radius: float = 0.01,
                             cone_radius: float = 0.02,
                             cylinder_height_ratio: float = 0.8,
                             cone_height_ratio: float = 0.2) -> o3d.geometry.TriangleMesh:
    length = np.linalg.norm(vector)
    if length == 0:
        raise ValueError("Vector length is zero, cannot create arrow.")
    direction = vector / length

    cylinder_height = length * cylinder_height_ratio
    cone_height = length * cone_height_ratio

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height
    )
    arrow.compute_vertex_normals()

    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        if np.dot(z_axis, direction) < 0:
            rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle([1, 0, 0] * np.pi)
        else:
            rot_mat = np.eye(3)
    else:
        axis = axis / axis_len
        angle = np.arccos(np.dot(z_axis, direction))
        rot_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    arrow.rotate(rot_mat, center=(0, 0, 0))
    arrow.translate(origin)
    return arrow

# --- Added: Extract low-level logic for pose updates ---
def apply_joint_transform(frame_handle, joint_type, origin, axis, val):
    """Apply the transformation matrix to the Viser Frame handle"""
    if joint_type == 'rotation':
        rot = R.from_rotvec(axis * val)
        scipy_quat = rot.as_quat() # [x, y, z, w]
        wxyz = (scipy_quat[3], scipy_quat[0], scipy_quat[1], scipy_quat[2])
        translation = origin - rot.apply(origin)
        
        frame_handle.position = translation
        frame_handle.wxyz = wxyz
        
    elif joint_type == 'translation':
        translation = axis * val
        frame_handle.position = translation
        frame_handle.wxyz = (1.0, 0.0, 0.0, 0.0)

def make_joint_callback(frame_handle, joint_type, origin, axis):
    def callback(event: viser.GuiEvent):
        val = event.target.value
        # Use the extracted logic
        apply_joint_transform(frame_handle, joint_type, origin, axis, val)
    return callback

def o3d_to_trimesh(o3d_mesh, default_color=(200, 200, 200)):
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    
    if o3d_mesh.has_vertex_colors():
        vertex_colors = (np.asarray(o3d_mesh.vertex_colors) * 255).astype(np.uint8)
    else:
        vertex_colors = np.tile(default_color, (len(vertices), 1)).astype(np.uint8)
        
    return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)


# ================= Main Program =================
def main():
    server = viser.ViserServer()
    print("Viser server started! Please open http://localhost:8080 in your browser.")
    
    # 1. Render Remain Scene (Background)
    remain_scene_path = os.path.join(result_dir, "remain_scene.ply")
    if os.path.exists(remain_scene_path):
        remain_scene = o3d.io.read_triangle_mesh(remain_scene_path)
        if remain_scene.has_vertices():
            remain_trimesh = o3d_to_trimesh(remain_scene)
            server.scene.add_mesh_trimesh(
                name="/scene/remain",
                mesh=remain_trimesh
            )

    gui_folder = server.gui.add_folder("Joint Controls")
    gui_sliders = []
    
    # --- Added: Save information of all movable parts for animation export ---
    parts_info = []

    # 2. Iterate through all Parts
    for subdir in sorted(os.listdir(result_dir)):
        subdir_path = os.path.join(result_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        ply_file = os.path.join(subdir_path, "part.ply")
        base_ply_file = os.path.join(subdir_path, "base.ply")
        inner_box_file = os.path.join(subdir_path, "inner_box.obj")
        
        joint_type_path = os.path.join(subdir_path, 'articulation_type.npy')
        origin_world_path = os.path.join(subdir_path, 'origin_world.npy')
        vector_world_path = os.path.join(subdir_path, 'vector_world.npy')

        if os.path.exists(ply_file):
            mesh = o3d.io.read_triangle_mesh(ply_file)
            base_pcd = o3d.io.read_point_cloud(base_ply_file)
            
            if mesh.is_empty():
                continue

            # --- A. Add Base Point Cloud ---
            if not base_pcd.is_empty():
                points = np.asarray(base_pcd.points)
                if base_pcd.has_colors():
                    colors = (np.asarray(base_pcd.colors) * 255).astype(np.uint8)
                else:
                    colors = np.ones_like(points, dtype=np.uint8) * 120
                
                server.scene.add_point_cloud(
                    name=f"/scene/{subdir}/base",
                    points=points,
                    colors=colors,
                    point_size=0.01
                )

            # --- B. Create Frame for Part and mount Mesh and Inner Box ---
            part_frame = server.scene.add_frame(
                name=f"/scene/{subdir}/part_frame",
                show_axes=False 
            )
            
            # B.1 Mount the main Part Mesh
            part_trimesh = o3d_to_trimesh(mesh)
            server.scene.add_mesh_trimesh(
                name=f"/scene/{subdir}/part_frame/mesh",
                mesh=part_trimesh
            )

            # B.2 Mount Inner Box (if exists)
            if os.path.exists(inner_box_file):
                inner_box_mesh = o3d.io.read_triangle_mesh(inner_box_file)
                if not inner_box_mesh.is_empty():
                    inner_box_trimesh = o3d_to_trimesh(inner_box_mesh, default_color=(100, 100, 100))
                    server.scene.add_mesh_trimesh(
                        name=f"/scene/{subdir}/part_frame/inner_box",
                        mesh=inner_box_trimesh
                    )

            # --- C. Parse joints and establish GUI control interaction ---
            if os.path.isfile(origin_world_path) and os.path.isfile(vector_world_path):
                origin_world = np.load(origin_world_path)
                vec = np.load(vector_world_path)
                norm = np.linalg.norm(vec)
                vector_world = vec / norm if norm > 0 else vec
                joint_type = str(np.load(joint_type_path).item())

                # --- Read Range Max ---
                range_max_path = os.path.join(subdir_path, 'range_max.npy')
                if os.path.isfile(range_max_path):
                    motion_range = float(np.load(range_max_path).item()) if joint_type == 'rotation' else 0.5
                else:
                    motion_range = np.pi / 2 if joint_type == 'rotation' else 0.5

                arrow_color = (0.8, 0.1, 0.1) if joint_type == 'rotation' else (0.0, 0.0, 1.0)
                arrow_mesh = create_arrow_from_vector(origin_world, vector_world, color=arrow_color)
                
                # uncomment this if you want to see the axis of the joint
                # server.scene.add_mesh_simple(
                #     name=f"/scene/{subdir}/arrow",
                #     vertices=np.asarray(arrow_mesh.vertices),
                #     faces=np.asarray(arrow_mesh.triangles),
                #     color=tuple((np.array(arrow_color) * 255).astype(int))
                # )

                with gui_folder:
                    if joint_type == 'rotation':
                        slider = server.gui.add_slider(
                            f"Rot: {subdir}", min=0.0, max=motion_range, step=0.01, initial_value=0.0
                        )
                    else:  
                        slider = server.gui.add_slider(
                            f"Trans: {subdir}", min=0.0, max=motion_range, step=0.01, initial_value=0.0
                        )
                        
                    slider.on_update(make_joint_callback(part_frame, joint_type, origin_world, vector_world))
                    gui_sliders.append(slider)
                    
                    # --- Added: Record physical and animation info for this part ---
                    parts_info.append({
                        'frame': part_frame,
                        'type': joint_type,
                        'origin': origin_world,
                        'axis': vector_world,
                        'max_val': motion_range # Set the upper limit for the animation amplitude
                    })

    # 3. Add Reset Button
    with gui_folder:
        reset_btn = server.gui.add_button("Reset All Joints")
        @reset_btn.on_click
        def _(_):
            for s in gui_sliders:
                s.value = 0.0

        # --- Added: Add export animation button ---
        export_btn = server.gui.add_button("Export Animation (.viser)")
        @export_btn.on_click
        def _(_):
            print("Starting serialization... Generating animation.")
            serializer = server.get_scene_serializer()
            num_frames = 100
            
            # Loop to generate animation frames
            for t in range(num_frames):
                # Use a periodic function for smooth reciprocating animation
                phase = (t / num_frames) * 2 * np.pi
                
                for info in parts_info:
                    # Calculate the joint value for the current frame (moves between 0 and max_val)
                    val = (1.0 - np.cos(phase)) / 2.0 * info['max_val']
                    # Apply low-level transform logic to update this frame
                    apply_joint_transform(info['frame'], info['type'], info['origin'], info['axis'], val)
                
                # Insert sleep to control the exported animation speed (e.g., 30fps)
                serializer.insert_sleep(1.0 / 30.0)

            # Save the generated animation file
            data = serializer.serialize()
            save_path = Path(result_dir) / "recording.viser"
            save_path.write_bytes(data)
            print(f"Animation saved successfully to: {save_path}")
            
            # Restore scene after export
            for s in gui_sliders:
                s.value = 0.0
            for info in parts_info:
                apply_joint_transform(info['frame'], info['type'], info['origin'], info['axis'], 0.0)


    # Keep the main thread running
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()