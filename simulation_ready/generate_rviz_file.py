#!/usr/bin/env python3
import argparse
import xml.etree.ElementTree as ET
import yaml

def parse_launch_groups(launch_file):
    """
    Parse the launch file and extract all ns attributes from <group> tags.
    Returns a list of namespace strings.
    """
    tree = ET.parse(launch_file)
    root = tree.getroot()
    groups = []
    # Find all <group> elements and get their ns attribute
    for group in root.findall("group"):
        ns = group.get("ns")
        if ns:
            groups.append(ns)
    return groups

def generate_rviz_config(groups):
    """
    Generate a complete RViz configuration dictionary based on a template,
    and add one RobotModel display for each group. Each RobotModel's "Robot Description"
    is set to "/<group>/robot_description".
    """
    # Base RViz configuration template
    config = {
        "Panels": [
            {
                "Class": "rviz/Displays",
                "Help Height": 78,
                "Name": "Displays",
                "Property Tree Widget": {
                    "Expanded": [
                        "/Global Options1",
                        "/Status1"
                        '/InteractiveMarkers1'
                    ],
                    "Splitter Ratio": 0.5
                },
                "Tree Height": 555
            },
            {
                "Class": "rviz/Selection",
                "Name": "Selection"
            },
            {
                "Class": "rviz/Tool Properties",
                "Expanded": [
                    "/2D Pose Estimate1",
                    "/2D Nav Goal1",
                    "/Publish Point1"
                ],
                "Name": "Tool Properties",
                "Splitter Ratio": 0.5886790156364441
            },
            {
                "Class": "rviz/Views",
                "Expanded": [
                    "/Current View1"
                ],
                "Name": "Views",
                "Splitter Ratio": 0.5
            },
            {
                "Class": "rviz/Time",
                "Experimental": False,
                "Name": "Time",
                "SyncMode": 0,
                "SyncSource": ""
            }
        ],
        "Preferences": {
            "PromptSaveOnExit": True
        },
        "Toolbars": {
            "toolButtonStyle": 2
        },
        "Visualization Manager": {
            "Class": "",
            "Displays": [
                # Add a Grid display item by default
                {
                    "Alpha": 0.5,
                    "Cell Size": 1,
                    "Class": "rviz/Grid",
                    "Color": "160; 160; 164",
                    "Enabled": True,
                    "Line Style": {
                        "Line Width": 0.03,
                        "Value": "Lines"
                    },
                    "Name": "Grid",
                    "Normal Cell Count": 0,
                    "Offset": {"X": 0, "Y": 0, "Z": 0},
                    "Plane": "XY",
                    "Plane Cell Count": 10,
                    "Reference Frame": "<Fixed Frame>",
                    "Value": True
                },
                {
                    "Alpha": 1,
                    "Class": "rviz/InteractiveMarkers",
                    "Enable Transparency": True,
                    "Enabled": True,
                    "Name": "InteractiveMarkers",
                    "Show Axes": False,
                    "Show Descriptions": True,
                    "Show Visual Aids": False,
                    "Update Topic": "/remain_scene/multi_link_mover/update",
                    "Value": True
                }
            ],
            "Enabled": True,
            "Global Options": {
                "Background Color": "48; 48; 48",
                "Default Light": True,
                "Fixed Frame": "world",
                "Frame Rate": 30
            },
            "Name": "root",
            "Tools": [
                {"Class": "rviz/Interact", "Hide Inactive Objects": True},
                {"Class": "rviz/MoveCamera"},
                {"Class": "rviz/Select"},
                {"Class": "rviz/FocusCamera"},
                {"Class": "rviz/Measure"},
                {
                    "Class": "rviz/SetInitialPose",
                    "Theta std deviation": 0.2617993950843811,
                    "Topic": "/initialpose",
                    "X std deviation": 0.5,
                    "Y std deviation": 0.5
                },
                {"Class": "rviz/SetGoal", "Topic": "/move_base_simple/goal"},
                {
                    "Class": "rviz/PublishPoint",
                    "Single click": True,
                    "Topic": "/clicked_point"
                }
            ],
            "Value": True,
            "Views": {
                "Current": {
                    "Class": "rviz/Orbit",
                    "Distance": 4.754905700683594,
                    "Enable Stereo Rendering": {
                        "Stereo Eye Separation": 0.06,
                        "Stereo Focal Distance": 1,
                        "Swap Stereo Eyes": False,
                        "Value": False
                    },
                    "Focal Point": {
                        "X": -0.2173389047384262,
                        "Y": 0.7964558601379395,
                        "Z": 0.71034175157547
                    },
                    "Focal Shape Fixed Size": True,
                    "Focal Shape Size": 0.05,
                    "Invert Z Axis": False,
                    "Name": "Current View",
                    "Near Clip Distance": 0.01,
                    "Pitch": 0.06539880484342575,
                    "Target Frame": "<Fixed Frame>",
                    "Value": "Orbit (rviz)",
                    "Yaw": 0.7553973197937012
                },
                "Saved": "~"
            }
        },
        "Window Geometry": {
            "Displays": {"collapsed": False},
            "Height": 846,
            "Width": 1200,
            "X": 228,
            "Y": 32
        }
    }

    # Add a RobotModel display item for each group
    for group in groups:
        robot_display = {
            "Alpha": 1,
            "Class": "rviz/RobotModel",
            "Collision Enabled": False,
            "Enabled": True,
            "Links": {
                "All Links Enabled": True,
                "Expand Joint Details": False,
                "Expand Link Details": False,
                "Expand Tree": False,
                "Link Tree Style": "Links in Alphabetic Order"
            },
            "Name": f"RobotModel_{group}",
            "Robot Description": f"/{group}/robot_description",
            "TF Prefix": "",
            "Update Interval": 0,
            "Value": True,
            "Visual Enabled": True
        }
        config["Visualization Manager"]["Displays"].append(robot_display)

    return config

parser = argparse.ArgumentParser(
    description="Generate an RViz config file based on group ns attributes in a ROS launch file."
)
parser.add_argument("--launch_file", type=str, required=True, help="Path to the launch file")
parser.add_argument("--output", type=str, default="generated.rviz", help="Output RViz config file name (default: generated.rviz)")
args = parser.parse_args()

groups = parse_launch_groups(args.launch_file)
if not groups:
    print("No group namespaces found in the launch file.")
    exit(1)

rviz_config = generate_rviz_config(groups)

with open(args.output, "w") as f:
    yaml.dump(rviz_config, f, default_flow_style=False, allow_unicode=True)

print("RViz config file generated:", args.output)

