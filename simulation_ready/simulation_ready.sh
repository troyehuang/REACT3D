#!/bin/bash

source /home/troye/miniconda3/etc/profile.d/conda.sh

CODE_DIR="/home/troye/ssd/REACT3D"
data_dir=/home/troye/ssd/REACT3D/data/realscan_test

scene_name=$(basename $data_dir)

conda activate react3d

# Create target directories
mkdir -p ${data_dir}/urdf/${scene_name}
mkdir -p ${data_dir}/launch
mkdir -p ${data_dir}/rviz

python generate_urdf_scene_ros.py --folder_path ${data_dir}/scene_output_filtered --output_path ${data_dir}/urdf/${scene_name}

python generate_launch_file.py --urdf_dir ${data_dir}/urdf/${scene_name} --output ${data_dir}/launch/${scene_name}.launch

python generate_rviz_file.py --launch_file ${data_dir}/launch/${scene_name}.launch --output ${data_dir}/rviz/${scene_name}.rviz