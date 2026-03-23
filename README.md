# REACT3D: Recovering Articulations for Interactive Physical 3D Scenes


<div align="center">
<a href="https://react3d.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2510.11340" target="_blank" rel="noopener noreferrer"> <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF"></a>
<!-- <a href="https://drive.google.com/file/d/1PuDanf0JnpyCRtiurPIESG80pU_nEBIj/view"> <img src="https://img.shields.io/badge/Demo-blue" alt="Demo"></a> -->

<p>
    <a href="https://troyehuang.github.io/">Zhao Huang</a>   
    ·
    <a href="https://boysun045.github.io/boysun-website/">Boyang Sun</a>
    · 
    <a href="https://alexdelitzas.github.io/">Alexandros Delitzas</a>
    ·
    <a href="https://www.linkedin.com/in/chen-jiaqi/">Jiaqi Chen</a>
    ·
    <a href="https://people.inf.ethz.ch/pomarc/">Marc Pollefeys</a>
</p>

<img src="./assets/teaser.png" alt="[Teaser Figure]" style="zoom:80%;" />
</div>

Interactive 3D scenes are increasingly vital for embodied intelligence, yet existing datasets remain limited due to the labor-intensive process of annotating part segmentation, kinematic types, and motion trajectories. We present REACT3D, a scalable zero-shot framework that converts static 3D scenes into simulation-ready interactive replicas with consistent geometry, enabling direct use in diverse downstream tasks.

## Setup

### Installation

Clone the repository and install necessary dependencies：

```bash
git clone https://github.com/troyehuang/REACT3D.git --recursive
cd REACT3D

conda create -n react3d python==3.10
conda activate react3d

export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9"

pip install ninja
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

# groundingDINO
set BUILD_WITH_CUDA=True
set CUDA_HOME=<your_path>
set AM_I_DOCKER=False

cd grounded_sam/GroundingDINO
python setup.py build
python setup.py install

# scene2part
cd ../..
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

pip install -r requirements.txt

# part2interactive
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

cd opdformer/mask2former/modeling/pixel_decoder/ops
python setup.py build install

```

Install LLaVA:
```bash
conda create -n llava python==3.10
conda activate llava

git clone https://github.com/xinyu1205/recognize-anything.git
cd LLaVA
pip install --upgrade pip
pip install -e .
```

Install checkpoints:
```bash
cd REACT3D

# ram++ checkpoint
cd ram++
wget --no-check-certificate https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth

# grounded sam checkpoint
cd ../grounded_sam
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# opdm checkpoint
cd ../part2interactive
wget --no-check-certificate https://huggingface.co/3dlg-hcvc/opdmulti-motion-state-rgb-model/resolve/main/pytorch_model.pth -O {REACT3D_dir}/part2interactive/opdm_rgb.pth
```

### Data Preparation
We evaluate our work on [ScanNet++](https://github.com/scannetpp/scannetpp) augmented by [Articulated3D](https://github.com/insait-institute/USDNet) and [MultiScan](https://github.com/smartscenes/multiscan) datasets. For convenience, a small subset of generated interactive scenes and correspoonding static input scenes is provided [here]().

To use custom data, please follow the format in "data" folder to process your own scenes. Make sure the your data format is the same as the example data.

The `data` folder is organized as follows:
```bash
data
|---scene_name
|   |---images_2
|   |---mesh_aligned_0.05.ply
|   |---pose_intrinsic_imu.json
|   |---depth
```

## Quick Start

To run the script on a specific scene, use:

```bash
cd REACT3D

cd scene2part
bash scene2part.sh # remember to change the paths

cd ../part2interactive
bash part2interactive.sh # remember to change the paths

cd ../texture
bash generate_texture.sh

cd ../simulation_ready
bash simulation_ready.sh

```



## Acknowledgements
Our work is based on [OPDMulti](https://github.com/3dlg-hcvc/OPDMulti) and [DRAWER](https://github.com/xiahongchi/DRAWER). We thank the authors for their great work and open-sourcing the code.

## Citation
```
@ARTICLE{11434845,
  author={Huang, Zhao and Sun, Boyang and Delitzas, Alexandros and Chen, Jiaqi and Pollefeys, Marc},
  journal={IEEE Robotics and Automation Letters}, 
  title={REACT3D: Recovering Articulations for Interactive Physical 3D Scenes}, 
  year={2026},
  volume={},
  number={},
  pages={1-8},
  keywords={Three-dimensional displays;Joints;Geometry;Image reconstruction;Estimation;Solid modeling;Point cloud compression;Foundation models;Biological system modeling;Accuracy;Semantic scene understanding;object detection;segmentation and categorization;RGB-D perception},
  doi={10.1109/LRA.2026.3674028}
}
```
