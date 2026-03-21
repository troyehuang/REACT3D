import json
import torch
import sys
import numpy as np
from utilities import compute_projection

pose_path = sys.argv[1]
save_path = pose_path.replace('.json', '_mvp.json')

with open(pose_path, 'r') as f:
    pose_dict = json.load(f)

W0, H0 = 1920, 1440
W1, H1 = 960, 720
#W0, H0 = 1440, 1920
#W1, H1 = 720, 960
sx, sy = W1 / W0, H1 / H0

for key, entry in pose_dict.items():
    
    aligned_pose_np = np.array(entry["aligned_pose"], dtype=np.float32)  # 4x4
    K_np = np.array(entry["intrinsic"], dtype=np.float32)               # 3x3

    K_np = K_np.copy()
    K_np[0, 0] *= sx  # fx
    K_np[1, 1] *= sy  # fy
    K_np[0, 2] *= sx  # cx
    K_np[1, 2] *= sy  # cy

    entry["intrinsic"] = K_np.tolist()

    aligned_pose = torch.from_numpy(aligned_pose_np).cuda().float()
    intrinsic = torch.from_numpy(K_np).cuda().float()
    mvp_tensor = compute_projection(aligned_pose, intrinsic, W=W1, H=H1)

    entry["mvp"] = mvp_tensor.detach().cpu().tolist()

with open(save_path, 'w') as f:
    json.dump(pose_dict, f, indent=4)
