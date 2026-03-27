#!/bin/bash

CODE_DIR="/cluster/project/cvg/students/zhahuang/REACT3D"
data_dir=/cluster/project/cvg/students/zhahuang/DRAWER/data_revision/40aec5fffa_new_1

conda activate react3d

python inference.py --scene-folder ${data_dir} --output ${data_dir}/scene_output --model opdm_rgb.pth

python filter_duplicates.py --input_dir ${data_dir}/scene_output --output_dir ${data_dir}/scene_output_filtered --iou_threshold 0.5

python generate_remain_scene.py --scene ${data_dir}/mesh_aligned_0.05.ply --part_dir ${data_dir}/scene_output_filtered

