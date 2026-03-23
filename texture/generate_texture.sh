#!/bin/bash

CODE_DIR="/cluster/project/cvg/students/zhahuang/REACT3D"
data_dir=/cluster/project/cvg/students/zhahuang/DRAWER/data_revision/realscan_test

conda activate react3d

# it might take a long time for big scene, change the texture resolution to make it faster
python convert_texture.py --folder ${data_dir}/scene_output_filtered --texture-size 256

blender --background --python blender_glb_to_dae.py -- --folder ${data_dir}/scene_output_filtered