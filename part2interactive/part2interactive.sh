#!/bin/bash

CODE_DIR="/cluster/project/cvg/students/zhahuang/REACT3D"
data_dir=/cluster/project/cvg/students/zhahuang/DRAWER/data_revision/realscan_test

conda activate opdm

python inference_mesh.py --folder ${data_dir}/perception/vis_groups_final_mesh --output ${data_dir}/scene_output --scene-mesh ${data_dir}/mesh_aligned_0.05.ply

python filter_duplicates.py --input_dir ${data_dir}/scene_output --output_dir ${data_dir}/scene_output_filtered --iou_threshold 0.5

python generate_remain_scene.py --scene ${data_dir}/mesh_aligned_0.05.ply --part_dir ${data_dir}/scene_output_filtered

