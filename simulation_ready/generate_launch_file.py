#!/usr/bin/env python3
import os
import argparse

parser = argparse.ArgumentParser(
    description="Generate a ROS launch file with a group for each URDF file."
)
parser.add_argument(
    "--urdf_dir",
    type=str,
    required=True,
    help="Directory containing URDF files (e.g., ./urdf/40aec5fffa/)"
)
parser.add_argument(
    "--output",
    type=str,
    default="generated.launch",
    help="Output launch file (default: generated.launch)"
)
args = parser.parse_args()

# Check if urdf_dir is a valid directory
if not os.path.isdir(args.urdf_dir):
    print(f"Error: {args.urdf_dir} is not a valid directory.")
    exit(1)

scene_id = os.path.basename(os.path.normpath(args.urdf_dir))

# Get all .urdf files in the directory
urdf_files = [f for f in os.listdir(args.urdf_dir) if f.endswith(".urdf")]

# Build launch file content
launch_content = "<launch>\n\n"
for urdf_file in urdf_files:
    # Remove extension and replace spaces with underscores as group name
    group_name = os.path.splitext(urdf_file)[0].replace(" ", "_")
    # Assume the URDF file path format used in the launch file is $(find sp)/urdf/scene_id/<urdf_file>
    group_block = f'  <group ns="{group_name}">\n'
    group_block += f'    <param name="robot_description" textfile="$(find sp)/urdf/{scene_id}/{urdf_file}" />\n'
    
    if urdf_file != "remain_scene.urdf":
        group_block += '    <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" output="screen" />\n'
        group_block += '    <node name="joint_state_publisher_gui" pkg="joint_state_publisher_gui" type="joint_state_publisher_gui" output="screen" />\n'
    else:
        group_block += (
            '    <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher" output="screen">\n'
            '      <remap from="tf"       to="/tf"/>\n'
            '      <remap from="tf_static" to="/tf_static"/>\n'
            '    </node>\n'
            '    <node name="link_mover_remain_scene" pkg="sp" type="link_mover_node.py" output="screen">\n'
            '      <param name="link_regex" value="^link_[0-9]+$"/>\n'
            '      <param name="parent_frame" value="world"/>\n'
            '    </node>\n'
        )
    group_block += "  </group>\n\n"
    launch_content += group_block

# Add global RViz node
launch_content += f'  <node name="rviz" pkg="rviz" type="rviz" args="-d $(find sp)/rviz/{scene_id}.rviz" output="screen" />\n'
launch_content += "</launch>\n"

# Write to output file
with open(args.output, "w") as f:
    f.write(launch_content)

print("Launch file generated:", args.output)

# if __name__ == '__main__':
#     main()
