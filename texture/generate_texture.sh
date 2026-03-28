#!/bin/bash

CODE_DIR="/cluster/project/cvg/students/zhahuang/REACT3D"
data_dir=/cluster/project/cvg/students/zhahuang/DRAWER/data_revision/40aec5fffa_new_1

conda activate react3d

# it might take a long time for big scene, change the texture resolution to make it faster
python convert_texture.py --folder ${data_dir}/scene_output_filtered --texture-size 128 --remain-texture-size 512

blender --background --python blender_glb_to_dae.py -- --folder ${data_dir}/scene_output_filtered