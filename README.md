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

<img src="./assets/teaser.jpg" alt="[Teaser Figure]" style="zoom:80%;" />
</div>

Interactive 3D scenes are increasingly vital for embodied intelligence, yet existing datasets remain limited due to the labor-intensive process of annotating part segmentation, kinematic types, and motion trajectories. We present REACT3D, a scalable zero-shot framework that converts static 3D scenes into simulation-ready interactive replicas with consistent geometry, enabling direct use in diverse downstream tasks.

## Setup

### Installation
Clone the repository and install necessary dependencies：

```bash
git clone https://github.com/troyehuang/REACT3D.git --recursive
conda create -n react3d python==3.10
conda activate react3d

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

pip install -r requirements.txt

cd opdformer/mask2former/modeling/pixel_decoder/ops
python setup.py build install

// opdm checkpoint
wget --no-check-certificate https://huggingface.co/3dlg-hcvc/opdmulti-motion-state-rgb-model/resolve/main/pytorch_model.pth -O {REACT3D_dir}/part2interactive/opdm_rgb.pth


```

### Data Preparation
We evaluate our work on [ScanNet++](https://github.com/scannetpp/scannetpp) augmented by [Articulated3D](https://github.com/insait-institute/USDNet) and [MultiScan](https://github.com/smartscenes/multiscan) datasets. For convenience, a small subset of generated interactive scenes and correspoonding static input scenes is provided [here]().

To use custom data, please follow the instructions in xxx to process your own scenes.

The `xxx` folder is organized as follows:
```bash
xxx
|---xxx
|   |---xxx
|   |---xxx
|   |---...
|---xxx
```

## Quick Start

To run the script on a specific scene, use:

```bash
cd REACT3D


cd part2interactive
python inference.py --scene-folder ${data_dir} --output ${data_dir}/scene_output --model opdm_rgb.pth
python filter_duplicates.py --input_dir ${data_dir}/scene_output/ --output_dir ${data_dir}/final_output/ --iou_threshold 0.5
python generate_remain_scene.py --scene ${data_dir}/mesh_aligned_0.05.ply --part_dir ${data_dir}/final_output/
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
