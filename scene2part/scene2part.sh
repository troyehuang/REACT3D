#!/bin/bash

CODE_DIR="/cluster/project/cvg/students/zhahuang/REACT3D"

conda activate react3d

data_dir=/cluster/project/cvg/students/zhahuang/DRAWER/data_revision/realscan_test

# ----------------------ram------------------------
cd ${CODE_DIR}/ram++

python inference_ram_plus.py --image_dir ${data_dir}/images_2 --pretrained ram_plus_swin_large_14m.pth

cd ${CODE_DIR}/scene2part

#------------------------openable tags filter--------------------
conda activate llava

python filter_ram_llava.py --data_dir ${data_dir}

PROMPT_FILE=${data_dir}/sam_prompt.txt
if [[ ! -f ${PROMPT_FILE} ]]; then
  echo "Prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi
TEXT_PROMPT="$(cat "${PROMPT_FILE}")"

#-----------------------perception---------------------
conda activate react3d

python generate_intrinsic_mvp.py ${data_dir}/pose_intrinsic_imu.json

cd ${CODE_DIR}/grounded_sam

python grounded_sam_detect_doors.py   --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py   --grounded_checkpoint groundingdino_swint_ogc.pth   --sam_checkpoint sam_vit_h_4b8939.pth   --input_dir ${data_dir}/images_2/   --output_dir ${data_dir}/grounded_sam   --box_threshold 0.3   --text_threshold 0.25   --text_prompt "${TEXT_PROMPT}"   --device "cuda"

cd ${CODE_DIR}/scene2part

python percept_stage1_local_low.py --data_dir ${data_dir}     --image_dir ${data_dir}/images_2     --mesh_path ${data_dir}/mesh_aligned_0.05.ply     --num_max_frames 2000     --num_faces_simplified 80000

python percept_stage2.py  --data_dir ${data_dir}

python percept_stage3_local_new.py     --data_dir ${data_dir}     --image_dir ${data_dir}/images_2 --top_k 5

python percept_stage4_local.py     --data_dir ${data_dir}

rm -r ${data_dir}/perception/vis_groups_gpt4_api

cp -r ${data_dir}/perception/vis_groups_handle_note ${data_dir}/perception/vis_groups_gpt4_api

python percept_stage6_local.py     --data_dir ${data_dir}   --mesh_path ${data_dir}/mesh_aligned_0.05.ply --image_dir ${data_dir}/images_2 --depth_dir ${data_dir}/depth

python percept_stage7_local.py -d ${data_dir}
