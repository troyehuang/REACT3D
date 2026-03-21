import numpy as np
import os
from pathlib import Path
import shutil
import argparse

# this file is to generate URDF for a whole object


def transform_coordinates_open3d_to_urdf(point):
    x, y, z = point
    return np.array([z, x, y])

def generate_urdf_interactive(folder_path, output_urdf_path, obj_name):
    folder_path = Path(folder_path)
    # subfolders = [d for d in folder_path.iterdir() if d.is_dir() and not d.name.endswith("not_valid")]
    
    urdf_parts = []

    urdf_parts.append("""
    <link name="world"/>""")
    
    # for i, subfolder in enumerate(subfolders):
    origin_path = folder_path / "origin_world.npy"
    vector_path = folder_path / "vector_world.npy"
    base_obj_path = folder_path / "base.obj"
    part_obj_path = folder_path / "part.dae"
    articulation_type_path = folder_path / "articulation_type.npy"
    inner_box_path = folder_path / "inner_box.obj"


    if not (origin_path.exists() and vector_path.exists() and base_obj_path.exists() and part_obj_path.exists() and articulation_type_path.exists()):
        print(f"skip {folder_path}, lack of necessary files.")
        return
    
    origin = np.load(origin_path)
    vector = np.load(vector_path)
    
    vector = vector / np.linalg.norm(vector)

    joint_type = "revolute"
    lower_limit = "0"
    upper_limit = "2.0944"  # Default rotation angle limit
    if articulation_type_path.exists():
        articulation_type = np.load(articulation_type_path)
        # If saved as byte type, convert to str
        if isinstance(articulation_type, (bytes, np.bytes_)):
            articulation_type = articulation_type.decode("utf-8")
        # If saved as a numpy array, convert to string (assuming only one element)
        if isinstance(articulation_type, np.ndarray):
            articulation_type = str(articulation_type.item())
        # Check type
        print(f"articulation_type: {articulation_type}")
        if articulation_type.lower() == "translation":
            joint_type = "prismatic"
            # Set the limits for prismatic joint from 0 to 0.5
            lower_limit = "0"
            upper_limit = "0.5"
    
    # Generate base link and its fixed joint
    urdf_parts.append(f"""
<link name="{obj_name}_base">
    <visual>
        <geometry>
            <mesh filename="package://sp/{base_obj_path}" scale="1 1 1"/>
        </geometry>
    </visual>
</link>

<joint name="{obj_name}_joint_base" type="fixed">
    <parent link="world"/>
    <child link="{obj_name}_base"/>
</joint>
    """)
    
    # Generate joint to connect base and part link
    urdf_parts.append(f"""
<joint name="{obj_name}_joint" type="{joint_type}">
    <parent link="{obj_name}_base"/>
    <child link="{obj_name}_part"/>
    <origin xyz="{origin[0]} {origin[1]} {origin[2]}"/>
    <axis xyz="{vector[0]} {vector[1]} {vector[2]}"/>
    <limit lower="{lower_limit}" upper="{upper_limit}" effort="100.0" velocity="1.0"/>
</joint>
    """)
    
    # Generate part link, merging visual of part and inner_box (if inner_box.obj exists)
    if inner_box_path.exists():
        urdf_parts.append(f"""
<link name="{obj_name}_part">
    <!-- part.dae -->
    <visual>
        <origin xyz="{-origin[0]} {-origin[1]} {-origin[2]}" rpy="0 0 0" />
        <geometry>
            <mesh filename="package://sp/{part_obj_path}" scale="1 1 1"/>
        </geometry>
    </visual>
    <!-- inner_box.obj -->
    <visual>
        <origin xyz="{-origin[0]} {-origin[1]} {-origin[2]}" rpy="0 0 0" />
        <geometry>
            <mesh filename="package://sp/{inner_box_path}" scale="1 1 1"/>
        </geometry>
    </visual>
</link>
        """)
    else:
        # If inner_box file does not exist, only use part.dae
        urdf_parts.append(f"""
<link name="{obj_name}_part">
    <visual>
        <origin xyz="{-origin[0]} {-origin[1]} {-origin[2]}" rpy="0 0 0" />
        <geometry>
            <mesh filename="package://sp/{part_obj_path}" scale="1 1 1"/>
        </geometry>
    </visual>
</link>
        """)
    
    urdf_content = f"""<?xml version="1.0"?>
<robot name="generated_robot">
""" + "\n".join(urdf_parts) + """
</robot>
    """
    
    with open(output_urdf_path, "w") as file:
        file.write(urdf_content)
    
    print(f"URDF generated: {output_urdf_path}")

def generate_urdf_static(folder_path, output_urdf_path):
    folder = Path(folder_path)
    #dae_files = sorted(folder.glob("*.dae"))
    dae_files = [folder]

    if not dae_files:
        print("could not find any dae files in the folder.")
        return

    urdf_parts = []
    
    urdf_parts.append('    <link name="world"/>')
    
    for i, dae_file in enumerate(dae_files):
        stem = dae_file.stem  # File name, excluding extension
        link_name = f"link_{i}"

        # Path to position.npy with the same name
        position_file = folder / f"{stem}_position.npy"
        if position_file.exists():
            position = np.load(position_file)
            x, y, z = position.tolist()
        else:
            print(f"Warning: position file not found for {dae_file.name}, using 0 0 0.")
            x, y, z = 0.0, 0.0, 0.0

        urdf_parts.append(f"""
    <link name="{link_name}">
        <visual>
            <geometry>
                <mesh filename="package://sp/{dae_file}" scale="1 1 1"/>
            </geometry>
            
            <material name="red_material">
                <color rgba="0.5 0.5 0.5 1.0"/>
            </material>
        </visual>
    </link>
        """)
        urdf_parts.append(f"""
    <joint name="{link_name}_joint_{i}" type="fixed">
        <parent link="world"/>
        <child link="{link_name}"/>
        <origin xyz="{x} {y} {z}" rpy="0 0 0"/>
    </joint>
        """)
    
    urdf_content = f"""<?xml version="1.0"?>
<robot name="static_objects_robot">
{''.join(urdf_parts)}
</robot>
"""
    with open(output_urdf_path, "w") as f:
        f.write(urdf_content)
    
    print(f"URDF generated: {output_urdf_path}")


parser = argparse.ArgumentParser(description="Process scene and URDF output paths.")
# parser.add_argument("--scene_id", type=str, default="40aec5fffa", help="Scene ID")
parser.add_argument("--folder_path", type=str, default=None, help="Folder path for scene output")
parser.add_argument("--output_path", type=str, default=None, help="Output path for URDF ROS")
args = parser.parse_args()

# 40aec5fffa
# scene_id = args.scene_id
folder_path = args.folder_path
output_path = args.output_path

if os.path.exists(output_path):
        shutil.rmtree(output_path)
os.makedirs(output_path, exist_ok=True)

# Iterate over the first-level subfolders in folder_path
for subfolder in os.listdir(folder_path):
    obj_name = subfolder.replace(" ", "_")
    subfolder_path = os.path.join(folder_path, subfolder)
    # Check if it is a directory and not named "unopenable"
    if not os.path.isdir(subfolder_path) and subfolder != "remain_scene.dae":
        continue
    
    # Build the URDF file output path with .urdf extension
    output_file = os.path.join(output_path, f"{obj_name}.urdf")
    
    if subfolder == "remain_scene.dae":
        output_file = os.path.join(output_path, "remain_scene.urdf")
        generate_urdf_static(subfolder_path, output_file)
    else:
        generate_urdf_interactive(subfolder_path, output_file, obj_name)




