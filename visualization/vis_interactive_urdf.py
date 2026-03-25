"""
Interactive URDF scene visualizer using viser.

Parses URDF files from the given scene directory, loads meshes, and provides
GUI sliders for manipulating revolute and prismatic joints in real time.

Usage:
    python vis_interactive.py --scene_dir 7f4d173c9c_ours
"""

import os
import xml.etree.ElementTree as ET
import argparse
import time

import numpy as np
import trimesh
import viser
import viser.transforms as tf


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------

def parse_urdf(urdf_path: str, meshes_dir: str) -> dict:
    """Parse a single URDF file and extract links, joints, and mesh paths.

    Returns a dict with keys:
        - name: robot name
        - links: {link_name: [list of visual dicts]}
        - joints: [list of joint dicts]
    Each visual dict has: mesh_path, origin_xyz, origin_rpy.
    Each joint dict has: name, type, parent, child, origin_xyz, axis, lower, upper.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # --- links ---
    links: dict[str, list[dict]] = {}
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name")
        visuals = []
        for vis in link_elem.findall("visual"):
            origin = vis.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                if origin.get("xyz"):
                    xyz = np.array([float(v) for v in origin.get("xyz").split()])
                if origin.get("rpy"):
                    rpy = np.array([float(v) for v in origin.get("rpy").split()])

            geom = vis.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is None:
                continue

            # Resolve the mesh path
            raw_path = mesh_elem.get("filename", "")
            if "package://" in raw_path:
                # Legacy package:// format — extract relative path after scene name
                parts = raw_path.split("/")
                scene_name = os.path.basename(meshes_dir)
                try:
                    idx = parts.index(scene_name)
                    rel = "/".join(parts[idx + 1 :])
                except ValueError:
                    rel = parts[-2] + "/" + parts[-1]
                mesh_path = os.path.join(meshes_dir, rel)
            else:
                # Relative path — resolve relative to URDF file directory
                urdf_dir = os.path.dirname(urdf_path)
                mesh_path = os.path.normpath(os.path.join(urdf_dir, raw_path))

            visuals.append({
                "mesh_path": mesh_path,
                "origin_xyz": xyz,
                "origin_rpy": rpy,
            })
        links[link_name] = visuals

    # --- joints ---
    joints = []
    for joint_elem in root.findall("joint"):
        jtype = joint_elem.get("type", "fixed")
        jname = joint_elem.get("name", "")

        parent = joint_elem.find("parent").get("link")
        child = joint_elem.find("child").get("link")

        origin_xyz = np.zeros(3)
        origin_elem = joint_elem.find("origin")
        if origin_elem is not None and origin_elem.get("xyz"):
            origin_xyz = np.array([float(v) for v in origin_elem.get("xyz").split()])

        axis = np.array([1.0, 0.0, 0.0])
        axis_elem = joint_elem.find("axis")
        if axis_elem is not None and axis_elem.get("xyz"):
            axis = np.array([float(v) for v in axis_elem.get("xyz").split()])
            norm = np.linalg.norm(axis)
            if norm > 0:
                axis = axis / norm

        lower, upper = 0.0, 0.0
        limit_elem = joint_elem.find("limit")
        if limit_elem is not None:
            lower = float(limit_elem.get("lower", "0"))
            upper = float(limit_elem.get("upper", "0"))

        joints.append({
            "name": jname,
            "type": jtype,
            "parent": parent,
            "child": child,
            "origin_xyz": origin_xyz,
            "axis": axis,
            "lower": lower,
            "upper": upper,
        })

    return {
        "name": root.get("name", "robot"),
        "links": links,
        "joints": joints,
    }


# ---------------------------------------------------------------------------
# Mesh loading helpers
# ---------------------------------------------------------------------------

def load_mesh(mesh_path: str) -> trimesh.Trimesh | None:
    """Load a mesh file, returning a single Trimesh (merged if Scene)."""
    if not os.path.isfile(mesh_path):
        print(f"  [WARN] Mesh not found: {mesh_path}")
        return None
    try:
        loaded = trimesh.load(mesh_path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = loaded.dump(concatenate=True)
        return loaded
    except Exception as e:
        print(f"  [WARN] Failed to load {mesh_path}: {e}")
        return None


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation: axis (unit vector) + angle → 3×3 rotation matrix."""
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def wxyz_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to quaternion [w, x, y, z]."""
    return tf.SO3.from_matrix(R).wxyz


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive URDF scene visualizer")
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="Path to scene directory (e.g. 7f4d173c9c_ours)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    scene_dir = args.scene_dir
    scene_name = os.path.basename(scene_dir.rstrip("/"))
    urdfs_dir = os.path.join(scene_dir, f"{scene_name}_urdfs")
    meshes_dir = os.path.join(scene_dir, f"{scene_name}_meshes")

    assert os.path.isdir(urdfs_dir), f"URDFs dir not found: {urdfs_dir}"
    assert os.path.isdir(meshes_dir), f"Meshes dir not found: {meshes_dir}"

    # Collect URDF files
    urdf_files = sorted([
        f for f in os.listdir(urdfs_dir) if f.endswith(".urdf")
    ])
    print(f"Found {len(urdf_files)} URDF files in {urdfs_dir}")

    # Parse all URDFs
    parsed_urdfs = []
    for fname in urdf_files:
        fpath = os.path.join(urdfs_dir, fname)
        parsed = parse_urdf(fpath, meshes_dir)
        parsed["_fname"] = fname
        parsed_urdfs.append(parsed)

    # Identify articulated joints (non-fixed)
    articulated = []
    for parsed in parsed_urdfs:
        for joint in parsed["joints"]:
            if joint["type"] in ("revolute", "prismatic"):
                articulated.append((parsed, joint))

    print(f"Found {len(articulated)} articulated joints")

    # ------------------------------------------------------------------
    # Setup viser server
    # ------------------------------------------------------------------
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Viser server starting at http://localhost:{args.port}")

    # ------------------------------------------------------------------
    # Add static meshes (base links and remain_scene)
    # ------------------------------------------------------------------
    print("Loading meshes...")
    # Track mesh handles for articulated parts (to update transforms)
    part_handles: dict[str, list] = {}  # joint_name -> list of mesh handles

    for parsed in parsed_urdfs:
        fname = parsed["_fname"]
        joints = parsed["joints"]

        # Find which links are children of articulated joints
        articulated_child_links = set()
        for j in joints:
            if j["type"] in ("revolute", "prismatic"):
                articulated_child_links.add(j["child"])

        # Add static (non-articulated) link meshes
        for link_name, visuals in parsed["links"].items():
            if link_name == "world":
                continue
            if link_name in articulated_child_links:
                continue  # Will be added as articulated below

            for vi, vis_info in enumerate(visuals):
                mesh = load_mesh(vis_info["mesh_path"])
                if mesh is None:
                    continue
                server.scene.add_mesh_trimesh(
                    name=f"/static/{fname}/{link_name}/{vi}",
                    mesh=mesh,
                )

        # Add articulated part meshes (initially at rest position)
        for joint in joints:
            if joint["type"] not in ("revolute", "prismatic"):
                continue
            child_link = joint["child"]
            if child_link not in parsed["links"]:
                continue

            handles = []
            for vi, vis_info in enumerate(parsed["links"][child_link]):
                mesh = load_mesh(vis_info["mesh_path"])
                if mesh is None:
                    continue
                handle = server.scene.add_mesh_trimesh(
                    name=f"/articulated/{joint['name']}/{vi}",
                    mesh=mesh,
                )
                handles.append((handle, vis_info))
            part_handles[joint["name"]] = handles

    # ------------------------------------------------------------------
    # Add GUI sliders for each articulated joint
    # ------------------------------------------------------------------
    print("Creating GUI controls...")
    gui_sliders: dict[str, viser.GuiInputHandle] = {}

    for parsed, joint in articulated:
        jname = joint["name"]
        jtype = joint["type"]
        lower = joint["lower"]
        upper = joint["upper"]

        label = jname
        if jtype == "revolute":
            # Display in degrees for user-friendliness
            slider = server.gui.add_slider(
                label=f"🔄 {label}",
                min=np.degrees(lower),
                max=np.degrees(upper),
                step=0.5,
                initial_value=0.0,
            )
        else:  # prismatic
            slider = server.gui.add_slider(
                label=f"↔ {label}",
                min=lower,
                max=upper,
                step=0.001,
                initial_value=0.0,
            )
        gui_sliders[jname] = (slider, joint, jtype)

    # Add a reset button
    reset_button = server.gui.add_button("Reset All Joints")

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def update_joint(jname: str, value: float, joint: dict, jtype: str):
        """Compute and apply the transform for a single joint.

        Mesh vertices are in world coordinates. The URDF transform chain is:
            T(joint_origin) * JointMotion * T(-joint_origin) * vertex
        which means rotation/translation happens around the joint origin.

        For viser (R_node * vertex + t_node), this gives:
          revolute:  R_node = R(axis, θ),  t_node = origin - R @ origin
          prismatic: R_node = I,           t_node = d * axis
        """
        origin = joint["origin_xyz"]
        axis = joint["axis"]
        handles = part_handles.get(jname, [])

        if jtype == "revolute":
            angle = np.radians(value)
            R = rotation_matrix_from_axis_angle(axis, angle)
            wxyz = wxyz_from_rotation_matrix(R)
            # Rotation around joint origin: t = origin - R * origin
            translation = origin - R @ origin
            for handle, vis_info in handles:
                handle.wxyz = wxyz
                handle.position = translation
        elif jtype == "prismatic":
            # Translation along axis (joint origin cancels out)
            translation = axis * value
            for handle, vis_info in handles:
                handle.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
                handle.position = translation

    # Register callbacks for each slider
    for jname, (slider, joint, jtype) in gui_sliders.items():
        # Use default argument capture to bind current values
        def make_callback(jn, jt, jd):
            def cb(event: viser.GuiEvent) -> None:
                update_joint(jn, event.target.value, jd, jt)
            return cb

        slider.on_update(make_callback(jname, jtype, joint))

    # Reset button callback
    @reset_button.on_click
    def _(_) -> None:
        for jname, (slider, joint, jtype) in gui_sliders.items():
            slider.value = 0.0
            update_joint(jname, 0.0, joint, jtype)

    print("Ready! Open the viser URL in your browser to interact.")

    # Keep the server alive
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
