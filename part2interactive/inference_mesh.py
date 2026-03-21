"""
inference.py
------------
Provides functionality to run the OPDMulti model on an input image, independent of dataset and ground truth, and 
visualize the output. Large portions of the code originate from get_prediction.py, rgbd_to_pcd_vis.py, 
evaluate_on_log.py, and other related files. The primary goal was to create a more standalone script which could be 
converted more easily into a public demo, thus the goal was to sever most dependencies on existing ground truth or 
datasets.

Example usage:
python inference.py \
    --rgb path/to/59-4860.png \
    --depth path/to/59-4860_d.png \
    --model path/to/model.pth \
    --output path/to/output_dir
"""

import argparse
import logging
import os
import time
import cv2
import shutil
from PIL import Image
from typing import Any
from collections import defaultdict
import pickle, csv
from scipy.ndimage import binary_fill_holes

import imageio
import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
from detectron2 import engine, evaluation
from detectron2.modeling import build_model
from detectron2.config import get_cfg, CfgNode
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import instances
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import (
    Visualizer,
    ColorMode,
    _create_text_labels,
    GenericMask,
)
from numpy.linalg import norm
from copy import deepcopy

from huggingface_hub import hf_hub_download

from mask2former import (
    add_maskformer2_config,
    add_motionnet_config,
)
from utilities import (
    prediction_to_json,
    camera_to_image, rotatePoint,
    circlePoints,
    transform_origin_and_vector_to_world,
    group_files_by_id,
    group_real_files_by_id,
    group_pickle_and_mesh_by_id,
    fit_2d_point_onto_3d_pointcloud,
    find_nearest_mask_point,
    rotate_pcd,
    pcd_to_mesh_poisson,
    transform_mesh_coord_open3d_to_urdf,
    transform_coordinates_open3d_to_urdf,
    corners_to_mesh,
    save_mesh_with_mtl,
    vertex_colors_to_texture,
    generate_texture_from_mesh,
    transform_object_pcd_to_world,
    transform_obj_mesh_to_world,
    copy_all,
    process_obb,
    transform_obb_to_world,
    coverage_ratio,
    fill_open_holes,
    recursive_path_replace,
    save_image_bbx,
    save_time_breakdown,
    load_scaled_intrinsics,
)
from visualization import (
    draw_line,
    generate_rotation_visualization,
    generate_translation_visualization,
    generate_articulation_visualization,
    generate_mesh_articulation_visualization,
    batch_trim,
    project_3d_to_2d,
    visualize_articulation_in_scene,
    vis_mesh_base_part,
)

HF_MODEL_PATH = {"repo_id": "3dlg-hcvc/opdmulti-motion-state-rgb-model", "filename": "pytorch_model.pth"}

# import based on torch version. Required for model loading. Code is taken from fvcore.common.checkpoint, in order to
# replicate model loading without the overhead of setting up an OPDTrainer

TORCH_VERSION: tuple[int, ...] = tuple(int(x) for x in torch.__version__.split(".")[:2])
if TORCH_VERSION >= (1, 11):
    from torch.ao import quantization
    from torch.ao.quantization import FakeQuantizeBase, ObserverBase
elif (
    TORCH_VERSION >= (1, 8)
    and hasattr(torch.quantization, "FakeQuantizeBase")
    and hasattr(torch.quantization, "ObserverBase")
):
    from torch import quantization
    from torch.quantization import FakeQuantizeBase, ObserverBase

# TODO: find a global place for this instead of in many places in code
TYPE_CLASSIFICATION = {
    0: "rotation",
    1: "translation",
}

ARROW_COLOR_ROTATION = [0, 1, 0]  # green
ARROW_COLOR_TRANSLATION = [0, 0, 1]  # blue

K_REALSENSE = [
    923.08227539,
    0.,
    632.93695068,
    0.,
    922.37591553,
    345.53430176,
    0.,
    0.,
    1.,
]

K_EXAMPLE = [
    214.85935872395834,
    0.0,
    0.0,
    0.0,
    214.85935872395834,
    0.0,
    125.90160319010417,
    95.13726399739583,
    1.0,
],

K_NOW = [
    480.,
    0.0,
    0.0,
    0.0,
    480.,
    0.0,
    639.5,
    359.5,
    1.0,
],

K_256 = [
    480.,
    0.0,
    0.0,
    0.0,
    480.,
    0.0,
    127.5,
    95.5,
    1.0,
],

K_SAME = [
    214.85935872395834,
    0.0,
    0.0,
    0.0,
    214.85935872395834,
    0.0,
    127.5,
    95.5,
    1.0,
],

K_MULTISCAN = [
    798.0402221679688,
    0.0,
    0.0,
    0.0,
    798.0402221679688,
    0.0,
    472.57562255859375,
    362.7891540527344,
    1.0,
],

K_MULTISCAN_256 = [
    212.81072591145834,  # fx' = 798.04... / 3.75
    0.0,
    0.0,
    0.0,
    212.81072591145834,  # fy' = 798.04... / 3.75
    0.0,
    126.020166015625,    # cx' = 472.57... / 3.75
    96.7437744140625,    # cy' = 362.78... / 3.75
    1.0,
],

def get_parser() -> argparse.ArgumentParser:
    """
    Specfy command-line arguments.

    The primary inputs to the script should be the image paths (RGBD) and camera intrinsics. Other arguments are
    provided to facilitate script testing and model changes. Run file with -h/--help to see all arguments.

    :return: parser for extracting command-line arguments
    """
    parser = argparse.ArgumentParser(description="Inference for OPDMulti")
    # The main arguments which should be specified by the user
    parser.add_argument(
        "--rgb",
        dest="rgb_image",
        metavar="FILE",
        help="path to RGB image file on which to run model",
    )
    parser.add_argument(
        "--depth",
        dest="depth_image",
        metavar="FILE",
        help="path to depth image file on which to run model",
    )
    parser.add_argument(
        "--extrinsics",
        dest="extrinsics_npy",
        metavar="FILE",
        help="path to extrinsics image file on which to run model",
    )
    parser.add_argument(
        "--folder",
        dest="folder",
        metavar="INPUT FOLDER",
        help="path to the folder containing RGB, depth and extrinsic files",
    )
    parser.add_argument(
        "--scene-mesh",
        dest="scene_mesh",
        help="path to the scene mesh file (.ply)",
    )
    parser.add_argument(  # FIXME: might make more sense to make this a path
        "-i",
        "--intrinsics",
        nargs=9,
        # default=K_SAME,
        default=K_MULTISCAN_256,
        dest="intrinsics",
        help="camera intrinsics matrix, as a list of values",
    )

    # optional parameters for user to specify
    parser.add_argument(
        "-n",
        "--num-samples",
        default=10,
        dest="num_samples",
        metavar="NUM",
        help="number of sample states to generate in visualization",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        dest="crop",
        help="crop whitespace out of images for visualization",
    )

    # local script development arguments
    parser.add_argument(
        "-m",
        "--model",
        default="/home/troye/ssd/opdm/pytorch_model.pth",  # FIXME: set a good default path
        dest="model",
        metavar="FILE",
        help="path to model file to run",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/coco/instance-segmentation/swin/opd_v1_real.yaml",
        metavar="FILE",
        dest="config_file",
        help="path to config file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output",  # FIXME: set a good default path
        dest="output",
        help="path to output directory in which to save results",
    )
    parser.add_argument(
        "--num-processes",
        default=1,
        dest="num_processes",
        help="number of processes per machine. When using GPUs, this should be the number of GPUs.",
    )
    parser.add_argument(
        "-s",
        "--score-threshold",
        default=0.1,
        type=float,
        dest="score_threshold",
        help="threshold between 0.0 and 1.0 by which to filter out bad predictions",
    )
    parser.add_argument(
        "--input-format",
        default="RGB",
        dest="input_format",
        help="input format of image. Must be one of RGB, RGBD, or depth",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        dest="vis",
        help="whether to visualize the intermediate results",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="flag to require code to use CPU only",
    )

    return parser


def setup_cfg(args: argparse.Namespace) -> CfgNode:
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add model configurations
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_motionnet_config(cfg)
    cfg.merge_from_file(args.config_file)

    # set additional config parameters
    cfg.MODEL.WEIGHTS = args.model
    cfg.OBJ_DETECT = False  # TODO: figure out if this is needed, and parameterize it
    cfg.MODEL.MOTIONNET.VOTING = "none"
    # Output directory
    cfg.OUTPUT_DIR = args.output
    cfg.MODEL.DEVICE = "cpu" if args.cpu else "cuda"

    cfg.MODEL.MODELATTRPATH = None

    # Input format
    cfg.INPUT.FORMAT = args.input_format
    if args.input_format == "RGB":
        cfg.MODEL.PIXEL_MEAN = cfg.MODEL.PIXEL_MEAN[0:3]
        cfg.MODEL.PIXEL_STD = cfg.MODEL.PIXEL_STD[0:3]
    elif args.input_format == "depth":
        cfg.MODEL.PIXEL_MEAN = cfg.MODEL.PIXEL_MEAN[3:4]
        cfg.MODEL.PIXEL_STD = cfg.MODEL.PIXEL_STD[3:4]
    elif args.input_format == "RGBD":
        pass
    else:
        raise ValueError("Invalid input format")

    cfg.freeze()
    engine.default_setup(cfg, args)

    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="opdformer")
    return cfg


def format_input(rgb_path: str) -> list[dict[str, Any]]:
    """
    Read and format input image into detectron2 form so that it can be passed to the model.

    :param rgb_path: path to RGB image file
    :return: list of dictionaries per image, where each dictionary is of the form
        {
            "file_name": path to RGB image,
            "image": torch.Tensor of dimensions [channel, height, width] representing the image
        }
    """
    image = imageio.imread(rgb_path).astype(np.float32)
    
    if image.shape[-1] == 4: 
        image = image[..., :3] 

    image_tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))  # dim: [channel, height, width]
    return [{"file_name": rgb_path, "image": image_tensor}]


def load_model(model: nn.Module, checkpoint: Any) -> None:
    """
    Load weights from a checkpoint.

    The majority of the function definition is taken from the DetectionCheckpointer implementation provided in
    detectron2. While not all of this code is necessarily needed for model loading, it was ported with the intention
    of keeping the implementation and output as close to the original as possible, and reusing the checkpoint class here
    in isolation was determined to be infeasible.

    :param model: model for which to load weights
    :param checkpoint: checkpoint contains the weights.
    """

    def _strip_prefix_if_present(state_dict: dict[str, Any], prefix: str) -> None:
        """If prefix is found on all keys in state dict, remove prefix."""
        keys = sorted(state_dict.keys())
        if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
            return

        for key in keys:
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    checkpoint_state_dict = checkpoint.pop("model")

    # convert from numpy to tensor
    for k, v in checkpoint_state_dict.items():
        if not isinstance(v, np.ndarray) and not isinstance(v, torch.Tensor):
            raise ValueError("Unsupported type found in checkpoint! {}: {}".format(k, type(v)))
        if not isinstance(v, torch.Tensor):
            checkpoint_state_dict[k] = torch.from_numpy(v)

    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching.
    _strip_prefix_if_present(checkpoint_state_dict, "module.")

    # workaround https://github.com/pytorch/pytorch/issues/24139
    model_state_dict = model.state_dict()
    incorrect_shapes = []
    for k in list(checkpoint_state_dict.keys()):  # state dict is modified in loop, so list op is necessary
        if k in model_state_dict:
            model_param = model_state_dict[k]
            # Allow mismatch for uninitialized parameters
            if TORCH_VERSION >= (1, 8) and isinstance(model_param, nn.parameter.UninitializedParameter):
                continue
            shape_model = tuple(model_param.shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                has_observer_base_classes = (
                    TORCH_VERSION >= (1, 8)
                    and hasattr(quantization, "ObserverBase")
                    and hasattr(quantization, "FakeQuantizeBase")
                )
                if has_observer_base_classes:
                    # Handle the special case of quantization per channel observers,
                    # where buffer shape mismatches are expected.
                    def _get_module_for_key(model: torch.nn.Module, key: str) -> torch.nn.Module:
                        # foo.bar.param_or_buffer_name -> [foo, bar]
                        key_parts = key.split(".")[:-1]
                        cur_module = model
                        for key_part in key_parts:
                            cur_module = getattr(cur_module, key_part)
                        return cur_module

                    cls_to_skip = (
                        ObserverBase,
                        FakeQuantizeBase,
                    )
                    target_module = _get_module_for_key(model, k)
                    if isinstance(target_module, cls_to_skip):
                        # Do not remove modules with expected shape mismatches
                        # them from the state_dict loading. They have special logic
                        # in _load_from_state_dict to handle the mismatches.
                        continue

                incorrect_shapes.append((k, shape_checkpoint, shape_model))
                checkpoint_state_dict.pop(k)

    model.load_state_dict(checkpoint_state_dict, strict=False)


def predict(model: nn.Module, inp: list[dict[str, Any]]) -> list[dict[str, instances.Instances]]:
    """
    Compute model predictions.

    :param model: model to run on input
    :param inp: input, in the form
        {
            "image_file": path to image,
            "image": float32 torch.tensor of dimensions [channel, height, width] as RGB/RGBD/depth image
        }
    :return: list of detected instances and predicted openable parameters
    """
    with torch.no_grad(), evaluation.inference_context(model):
        out = model(inp)
    return out

def arrow_back_project(rgb_image, origin, axis_vector, intrinsic_matrix, mask, output_dir, visible_points_mask, vis):

    # pcd_center = pcd.get_center()
    rgb_image = cv2.imread(rgb_image)

    # mask
    mask_color = np.array([255, 0, 0], dtype=np.uint8)
    colored_mask = np.zeros_like(rgb_image)
    colored_mask[mask == 1] = mask_color

    # visualize visible points mask
    v_rgb_image = rgb_image.copy()
    visible_color = np.array([0, 255, 0], dtype=np.uint8)
    v_colored_mask = np.zeros_like(v_rgb_image)
    v_colored_mask[visible_points_mask == 1] = visible_color
    alpha = 0.5
    v_rgb_image[visible_points_mask == 1] = (1 - alpha) * v_rgb_image[visible_points_mask == 1] + alpha * v_colored_mask[visible_points_mask == 1]
    
    rgb_image[mask == 1] = (1 - alpha) * rgb_image[mask == 1] + alpha * colored_mask[mask == 1]

    # alpha = 0.5
    # rgb_image[mask == 1] = (1 - alpha) * rgb_image[mask == 1] + alpha * colored_mask[mask == 1]
    

    endpoint = origin + axis_vector * 0.4
    origin_2d = project_3d_to_2d(origin, intrinsic_matrix)
    endpoint_2d = project_3d_to_2d(endpoint, intrinsic_matrix)
    vector_2d = np.array(endpoint_2d) - np.array(origin_2d)

    # refine the origin point
    origin_2d = find_nearest_mask_point(visible_points_mask, origin_2d)
    endpoint_2d = np.array(origin_2d) + vector_2d
    endpoint_2d = int(endpoint_2d[0]), int(endpoint_2d[1])

    # print("origin_2d: ", origin_2d)
    # print("endpoint_2d: ", endpoint_2d)
    
    image_with_arrow = v_rgb_image.copy()

    # Determine image boundaries
    h, w, _ = image_with_arrow.shape
    min_x = min(origin_2d[0], endpoint_2d[0])
    max_x = max(origin_2d[0], endpoint_2d[0])
    min_y = min(origin_2d[1], endpoint_2d[1])
    max_y = max(origin_2d[1], endpoint_2d[1])
    
    # Check if the arrow extends beyond the image boundaries
    if min_x < 0 or min_y < 0 or max_x >= w or max_y >= h:
        # Calculate the new dimensions
        new_w = max(max_x + 10, w)  # Add a margin of 10 pixels
        new_h = max(max_y + 10, h)
        offset_x = max(-min_x, 0)
        offset_y = max(-min_y, 0)
        
        # Create a larger canvas with a white background
        new_image = np.ones((new_h + offset_y, new_w + offset_x, 3), dtype=np.uint8) * 255
        
        # Place the original image on the canvas
        new_image[offset_y:offset_y + h, offset_x:offset_x + w] = image_with_arrow
        image_with_arrow = new_image
        
        # Update the mask
        new_mask = np.zeros_like(new_image[:, :, 0])
        new_mask[offset_y:offset_y + h, offset_x:offset_x + w] = visible_points_mask
        visible_points_mask = new_mask
        
        # Adjust the origin and endpoint positions
        origin_2d = (origin_2d[0] + offset_x, origin_2d[1] + offset_y)
        endpoint_2d = (endpoint_2d[0] + offset_x, endpoint_2d[1] + offset_y)

    # draw circle and arrow
    cv2.circle(image_with_arrow, origin_2d, 5, (0, 255, 0), -1)
    cv2.arrowedLine(image_with_arrow, origin_2d, endpoint_2d, (0, 0, 255), 2, tipLength=0.05)

    output_filename = f"{output_dir}/2d_output.png"
    cv2.imwrite(output_filename, image_with_arrow)

    if vis:
        cv2.imshow("mask image", rgb_image)
        cv2.imshow("Visible Points Mask with Arrow", image_with_arrow)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return origin_2d

def vis_2d_pred(img_path, origin, axis_vector, intrinsic_matrix, pred_type, is_real, origin_color, axis_color):
    origin = origin.tolist()
    axis_vector = axis_vector.tolist()

    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    img = img[:, :, :3]
    cv_in = img.copy()
    cv_in = cv2.cvtColor(cv_in, cv2.COLOR_BGR2RGB)

    vis = Visualizer(
                cv_in,
                scale=2,
                instance_mode=ColorMode.IMAGE,  # remove the colors of unsegmented pixels
            )

    line_length = 0.5
    pred_origin_4d = origin + [1]
    pred_origin_2d = camera_to_image(pred_origin_4d, is_real, intrinsic_matrix)
    pred_axis = axis_vector
    pred_axis = list(pred_axis / norm(pred_axis))
    pred_new_point = list(np.array(origin) + line_length * np.array(pred_axis))
    pred_new_point = pred_new_point + [1]
    pred_new_point = camera_to_image(pred_new_point, is_real, intrinsic_matrix)
    
    # Visualize the predicted axis
    pred_arrow_p0 = rotatePoint(
        pred_new_point[0] - pred_origin_2d[0],
        pred_new_point[1] - pred_origin_2d[1],
        30,
        0.1,
    )
    pred_arrow_p1 = rotatePoint(
        pred_new_point[0] - pred_origin_2d[0],
        pred_new_point[1] - pred_origin_2d[1],
        -30,
        0.1,
    )
    pred_circle_p = circlePoints(pred_axis, 0.1, 50)
    pred_circle_p = line_length * pred_circle_p + np.repeat(
        np.asarray(origin)[:, np.newaxis], 50, axis=1
    )
    pred_circle_p = pred_circle_p.transpose()
    pred_circle_p_2d = np.asarray([camera_to_image(p, is_real, intrinsic_matrix) for p in pred_circle_p])
    # text_y_offset = 1 if (new_point[1]-origin_2d[0][1]) > 0 else -1
    # self.draw_text("axis_error: {:.3f}".format(axis_diff), (origin_2d[0][0], origin_2d[0][1]-20*text_y_offset), color="tan")
    vis.draw_line(
        [pred_origin_2d[0], pred_new_point[0]],
        [pred_origin_2d[1], pred_new_point[1]],
        color=axis_color,
        linewidth=2,
    )
    vis.draw_line(
        [pred_new_point[0] - pred_arrow_p0[0], pred_new_point[0]],
        [pred_new_point[1] - pred_arrow_p0[1], pred_new_point[1]],
        color=axis_color,
        linewidth=2,
    )
    vis.draw_line(
        [pred_new_point[0] - pred_arrow_p1[0], pred_new_point[0]],
        [pred_new_point[1] - pred_arrow_p1[1], pred_new_point[1]],
        color=axis_color,
        linewidth=2,
    )
    vis.draw_polygon(
        pred_circle_p_2d, color=axis_color, edge_color=axis_color, alpha=0.0
    )
    if pred_type == 0:
        # self.draw_text("origin_error: {:.3f}".format(origin_diff), (origin_2d[0][0], origin_2d[0][1]-10*text_y_offset), color="c")
        vis.draw_circle(pred_origin_2d, color=origin_color, radius=5)
    
    return vis.output


# def main(
#     cfg: CfgNode,
#     rgb_image: str,
#     intrinsics: list[float],
#     num_samples: int,
#     crop: bool,
#     score_threshold: float,
# ) -> None:
#     """
#     Main inference method.

#     :param cfg: configuration object
#     :param rgb_image: local path to RGB image
#     :param depth_image: local path to depth image
#     :param intrinsics: camera intrinsics matrix as a list of 9 values
#     :param num_samples: number of sample visualization states to generate
#     :param crop: if True, images will be cropped to remove whitespace before visualization
#     :param score_threshold: float between 0 and 1 representing threshold at which to filter instances based on score
#     """
#     logger = logging.getLogger("detectron2")

#     # setup data
#     logger.info("Loading image.")
#     inp = format_input(rgb_image)

#     # setup model
#     logger.info("Loading model.")
#     model = build_model(cfg)
#     weights = torch.load(cfg.MODEL.WEIGHTS, map_location=torch.device("cpu"))
#     if "model" not in weights:
#         weights = {"model": weights}
#     load_model(model, weights)

#     # run model on data
#     logger.info("Running model.")
#     prediction = predict(model, inp)[0]  # index 0 since there is only one image
#     pred_instances = prediction["instances"]
#     # print("+++++++++++++++++++++++++++++++++++++++")
#     # print(prediction)

#     # log results
#     image_id = os.path.splitext(os.path.basename(rgb_image))[0]
#     pred_dict = {"image_id": image_id}
#     instances = pred_instances.to(torch.device("cpu"))
#     pred_dict["instances"] = prediction_to_json(instances, image_id)
#     torch.save(pred_dict, os.path.join(cfg.OUTPUT_DIR, f"{image_id}_prediction.pth"))
#     print("+++++++++++++++++++++++++++++++++++++++")
#     # print(pred_dict)
    

#     # select best prediction to visualize
#     score_ranking = np.argsort([-1 * pred_instances[i].scores.item() for i in range(len(pred_instances))])
#     score_ranking = [idx for idx in score_ranking if pred_instances[int(idx)].scores.item() > score_threshold]
#     if len(score_ranking) == 0:
#         logging.warning("The model did not predict any moving parts above the score threshold.")
#         return

#     for idx in score_ranking:  # iterate through all best predictions, by score threshold
#         pred = pred_instances[int(idx)]  # take highest predicted one
#         logger.info("Rendering prediction for instance %d", int(idx))
#         output_dir = os.path.join(cfg.OUTPUT_DIR, str(idx))
#         os.makedirs(output_dir, exist_ok=True)

#         # extract predicted values for visualization
#         mask = np.squeeze(pred.pred_masks.cpu().numpy())  # dim: [height, width]
#         origin = pred.morigin.cpu().numpy().flatten()  # dim: [3, ]
#         axis_vector = pred.maxis.cpu().numpy().flatten()  # dim: [3, ]
#         # print("origin: ", origin)
#         # print("axis_vector: ", axis_vector)

#         pred_type = TYPE_CLASSIFICATION.get(pred.mtype.item())
#         range_min = 0 - pred.mstate.cpu().numpy()
#         range_max = pred.mstatemax.cpu().numpy() - pred.mstate.cpu().numpy()

#         # process visualization
#         color = o3d.io.read_image(rgb_image)
#         # depth = o3d.io.read_image(depth_image)
        
#         # depth_np = np.load(depth_image)
#         # depth = o3d.geometry.Image(depth_np)

#         # depth_array = depth_np
#         # print(f"Depth value range: {depth_array.min()} - {depth_array.max()}")

#         # depth_scale = 1.
#         # rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=depth_scale, convert_rgb_to_intensity=False)
#         color_np = np.asarray(color)
#         height, width = color_np.shape[:2]

#         # generate intrinsics
#         intrinsic_matrix = np.reshape(intrinsics, (3, 3), order="F")
#         intrinsic_obj = o3d.camera.PinholeCameraIntrinsic(
#             width,
#             height,
#             intrinsic_matrix[0, 0],
#             intrinsic_matrix[1, 1],
#             intrinsic_matrix[0, 2],
#             intrinsic_matrix[1, 2],
#         )

#         # Convert the RGBD image to a point cloud
#         # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_obj)

#         arrow_back_project(rgb_image, origin, axis_vector, intrinsic_matrix, mask)

#         # vis = o3d.visualization.Visualizer()
#         # vis.create_window(visible=True)

#         # Add the translated geometries
#         # vis.add_geometry(pcd)
#         # vis.run()
#         # vis.destroy_window()

#         # Create a LineSet to visualize the direction vector
#         # axis_arrow = draw_line(origin, axis_vector + origin, depth_scale, depth_array, K_EXAMPLE, K_NOW)
#         # axis_arrow.paint_uniform_color(ARROW_COLOR_ROTATION)


#         # if USE_GT:
#         #     anno_path = f"/localhome/atw7/projects/opdmulti/data/data_demo_dev/59-4860.json"
#         #     part_id = 32

#         #     # get annotation for the frame
#         #     import json

#         #     with open(anno_path, "r") as f:
#         #         anno = json.load(f)

#         #     articulations = anno["articulation"]
#         #     for articulation in articulations:
#         #         if articulation["partId"] == part_id:
#         #             range_min = articulation["rangeMin"] - articulation["state"]
#         #             range_max = articulation["rangeMax"] - articulation["state"]
#         #             break

#         # if pred_type == "rotation":
#         #     generate_rotation_visualization(
#         #         pcd,
#         #         axis_arrow,
#         #         mask,
#         #         axis_vector,
#         #         origin,
#         #         range_min,
#         #         range_max,
#         #         num_samples,
#         #         output_dir,
#         #     )
#         # elif pred_type == "translation":
#         #     generate_translation_visualization(
#         #         pcd,
#         #         axis_arrow,
#         #         mask,
#         #         axis_vector,
#         #         range_min,
#         #         range_max,
#         #         num_samples,
#         #         output_dir,
#         #     )
#         # else:
#         #     raise ValueError(f"Invalid motion prediction type: {pred_type}")
        
#         # generate_articulation_visualization(
#         #     pcd,
#         #     axis_arrow,
#         #     mask,
#         #     axis_vector,
#         #     origin,
#         #     output_dir,
#         # )

#         if pred_type:
#             if crop:  # crop images to remove shared extraneous whitespace
#                 output_dir_cropped = f"{output_dir}_cropped"
#                 if not os.path.isdir(output_dir_cropped):
#                     os.makedirs(output_dir_cropped)
#                 batch_trim(output_dir, output_dir_cropped, identical=True)
#                 # create_gif(output_dir_cropped, num_samples)
#             else:  # leave original dimensions of image as-is
#                 # create_gif(output_dir, num_samples)
#                 pass

def main(
    cfg: CfgNode,
    folder: str,
    scene_mesh: str,
    intrinsics: list[float],
    num_samples: int,
    crop: bool,
    score_threshold: float,
    vis: bool,
) -> None:
    """
    Main inference method.

    :param cfg: configuration object
    :param rgb_image: local path to RGB image
    :param depth_image: local path to depth image
    :param intrinsics: camera intrinsics matrix as a list of 9 values
    :param num_samples: number of sample visualization states to generate
    :param crop: if True, images will be cropped to remove whitespace before visualization
    :param score_threshold: float between 0 and 1 representing threshold at which to filter instances based on score
    """
    logger = logging.getLogger("detectron2")

    scene_id = folder.split("/")[-3]
    scene_pcd = o3d.io.read_point_cloud(scene_mesh)

    file_groups = group_pickle_and_mesh_by_id(folder + "/real")
    # print(file_groups)

    scene_output_dir = os.path.join(cfg.OUTPUT_DIR, folder.split("/")[-3])
    print(scene_output_dir)
    if os.path.exists(scene_output_dir):
        shutil.rmtree(scene_output_dir)
    os.makedirs(scene_output_dir, exist_ok=True)

    # copy_all(folder + "/unopenable", scene_output_dir + "/unopenable")

    bounding_box_list = []
    
    for label_id, file_group in file_groups.items():
        print("---------------------+++++++++++++++++++++++++++++---------------------")
        print("label_id: ", label_id)
        # if label_id != "mesh_607":
        #     continue

        pickle_path = file_group["pickle_path"]
        mask_mesh_path = file_group["mask_mesh_path"]
        mask_mesh = o3d.io.read_point_cloud(mask_mesh_path)
        mask_mesh_mesh = o3d.io.read_triangle_mesh(mask_mesh_path)
        print("mask_mesh: ", mask_mesh_path)

        with open(pickle_path, 'rb') as f:
            top_k_list = pickle.load(f)
            original_top_k_list = deepcopy(top_k_list)
            # top_k_list = top_k_list[0]  # get the first element of the list
            # top_k_list = recursive_path_replace(top_k_list, f"/cluster/project/cvg/students/zhahuang/DRAWER/data/{scene_id}/perception/vis_groups_final_mesh", folder)
            try:
                print("111111111111111111111111  " +scene_id)
                # top_k_list = recursive_path_replace(
                #     top_k_list,
                #     f"/cluster/project/cvg/students/zhahuang/DRAWER/data_new/{scene_id}/perception/vis_groups_final_mesh",
                #     folder
                # )
                
                top_k_list = recursive_path_replace(
                    top_k_list,
                    f"/cluster/project/cvg/students/zhahuang/DRAWER/data_revision/{scene_id}/perception/vis_groups_final_mesh",
                    folder
                )
            except Exception as e:
                print(f"[Warn] primary path replace failed: {e}, trying fallback...")
                top_k_list = recursive_path_replace(
                    original_top_k_list,
                    f"/cluster/project/cvg/students/zhahuang/DRAWER/data_new/{scene_id}/perception/vis_groups_final_mesh",
                    folder
                )
            best_k = top_k_list[0]  # get the best k frame

        print("len(top_k_list): ", len(top_k_list))
        success = False
        for k_frame in top_k_list:
            if success:
                break

            rgb_image = k_frame["rgb_path"]
            print("rgb_image_path: ", rgb_image)
            depth_np = k_frame["depth"]
            extrinsic = k_frame["extrinsic"]
            
            visible_points_mask = k_frame["mask"]
            visible_points_mask = (visible_points_mask > 0).astype(np.uint8)
            # visible_points_mask = fill_open_holes(visible_points_mask)

            # print("visible_points_mask: ", visible_points_mask)
            # print("visible_points_mask: ", visible_points_mask.shape)

            # setup data
            logger.info("Loading image.")
            inp = format_input(rgb_image)

            # setup model
            logger.info("Loading model.")
            model = build_model(cfg)
            weights = torch.load(cfg.MODEL.WEIGHTS, map_location=torch.device("cpu"))
            if "model" not in weights:
                weights = {"model": weights}
            load_model(model, weights)

            # run model on data
            logger.info("Running model.")
            prediction = predict(model, inp)[0]  # index 0 since there is only one image
            pred_instances = prediction["instances"]
            # print("+++++++++++++++++++++++++++++++++++++++")
            # print(prediction)

            # log results
            # image_id = os.path.splitext(os.path.basename(rgb_image))[0]
            # pred_dict = {"image_id": image_id}
            # instances = pred_instances.to(torch.device("cpu"))
            # pred_dict["instances"] = prediction_to_json(instances, image_id)
            # torch.save(pred_dict, os.path.join(cfg.OUTPUT_DIR, f"{image_id}_prediction.pth"))
            # print("+++++++++++++++++++++++++++++++++++++++")
            # print(pred_dict)

            output_dir = os.path.join(scene_output_dir, label_id)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            # select best prediction to visualize
            score_ranking = np.argsort([-1 * pred_instances[i].scores.item() for i in range(len(pred_instances))])
            score_ranking = [idx for idx in score_ranking if pred_instances[int(idx)].scores.item() > score_threshold]
            if len(score_ranking) == 0:
                logging.warning("The model did not predict any moving parts above the score threshold.")

            pred_idx = 0
            for idx in score_ranking:  # iterate through all best predictions, by score threshold
                pred = pred_instances[int(idx)]  # take highest predicted one
                print("Predict Fields: ")
                print(pred.get_fields().keys())
                print("pred.scores: ", pred.scores)
                logger.info("Rendering prediction for instance %d", int(pred_idx))
                # curr_output_dir = os.path.join(output_dir, str(pred_idx))
                curr_output_dir = output_dir
                os.makedirs(curr_output_dir, exist_ok=True)

                # extract predicted values for visualization
                mask = np.squeeze(pred.pred_masks.cpu().numpy())  # dim: [height, width]
                origin = pred.morigin.cpu().numpy().flatten()  # dim: [3, ]
                axis_vector = pred.maxis.cpu().numpy().flatten()  # dim: [3, ]
                print("origin: ", origin)
                print("axis_vector: ", axis_vector)

                pred_type = TYPE_CLASSIFICATION.get(pred.mtype.item())
                range_min = 0 - pred.mstate.cpu().numpy()
                range_max = pred.mstatemax.cpu().numpy() + pred.mstate.cpu().numpy()

                # process visualization
                color = o3d.io.read_image(rgb_image)
                # depth = o3d.io.read_image(depth_image)
                # depth_np = np.array(depth)
                depth = o3d.geometry.Image(depth_np)

                # depth_array = depth_np
                # print(f"Depth value range: {depth_array.min()} - {depth_array.max()}")

                depth_scale = 1.
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=depth_scale, depth_trunc=10.0, convert_rgb_to_intensity=False)
                color_np = np.asarray(color)
                height, width = color_np.shape[:2]

                # generate intrinsics
                intrinsic_matrix = np.reshape(intrinsics, (3, 3), order="F")
                intrinsic_obj = o3d.camera.PinholeCameraIntrinsic(
                    width,
                    height,
                    intrinsic_matrix[0, 0],
                    intrinsic_matrix[1, 1],
                    intrinsic_matrix[0, 2],
                    intrinsic_matrix[1, 2],
                )

                # Convert the RGBD image to a point cloud
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_obj)

                # visualize pcd
                # vis = o3d.visualization.Visualizer()
                # vis.create_window(visible=True)
                # vis.add_geometry(pcd)
                # vis.run()
                # vis.destroy_window()
                
                print("rgb_image_path2: ", rgb_image)
                try:
                    origin_2d = arrow_back_project(rgb_image, origin, axis_vector, intrinsic_matrix, mask, curr_output_dir, visible_points_mask, vis)

                except MemoryError:
                    print(f"[Skip] {label_id} encountered (MemoryError)")
                    continue
                except Exception as e:
                    print(f"[Skip] {label_id} projection failed, reason: {e}. Skipping this instance.")
                    continue

                mask_coverage_ratio = coverage_ratio(visible_points_mask, mask)
                if mask_coverage_ratio < 0.6:
                    print(f"Mask coverage ratio is too low: {mask_coverage_ratio}")
                    continue
                
                mask = visible_points_mask

                # world to local
                invert_extrinsic = np.linalg.inv(extrinsic)
                local_mask_mesh = transform_object_pcd_to_world(deepcopy(mask_mesh), extrinsic)
                local_mask_mesh_mesh = transform_obj_mesh_to_world(deepcopy(mask_mesh_mesh), extrinsic)
                axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

                # print(f"Visualizing local mask mesh and axis... {mask_mesh_path}")
                # vis = o3d.visualization.Visualizer()
                # vis.create_window(visible=True)

                # # 2. Add coordinate axes to the window
                # vis.add_geometry(local_mask_mesh_mesh)
                # vis.add_geometry(axis_pcd)
                
                # vis.run()
                # vis.destroy_window()

                # get best point cloud
                best_color = o3d.io.read_image(rgb_image)
                best_depth = o3d.geometry.Image(depth_np)
                best_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(best_color, best_depth, depth_scale=depth_scale, depth_trunc=10.0, convert_rgb_to_intensity=False)
                best_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(best_rgbd_image, intrinsic_obj)

                origin_3d = fit_2d_point_onto_3d_pointcloud(origin_2d, pcd, intrinsic_matrix, depth_np, depth_scale=1.)
                if origin_3d is None:
                    continue

                # Create a LineSet to visualize the direction vector
                axis_arrow = draw_line(origin_3d, axis_vector + origin_3d, depth_scale)
                if pred_type == "rotation":
                    axis_arrow.paint_uniform_color(ARROW_COLOR_ROTATION)
                elif pred_type == "translation":
                    axis_arrow.paint_uniform_color(ARROW_COLOR_TRANSLATION)

                object_combined_pcd, box_pcd, cuboid_world, translated_arrow, translated_axis_vector, translated_origin_3d, obb, origin_3d, axis_vector = generate_mesh_articulation_visualization(
                    pcd,
                    axis_arrow,
                    mask,
                    axis_vector,
                    origin,
                    curr_output_dir,
                    intrinsic_matrix,
                    depth_np,
                    origin_3d,
                    extrinsic,
                    pred_type,
                    scene_pcd,
                    local_mask_mesh,
                    best_k,
                    best_pcd,
                    vis,
                    range_max,
                    local_mask_mesh_mesh
                )
                
                if box_pcd is None:
                    print("No box point cloud found, skipping this instance.")
                    continue
                # bounding_box_list = process_obb(bounding_box_list, curr_output_dir, np.array(cuboid_world["corners"]))
                # print(cuboid_world["corners"])
                # print(transform_obb_to_world(obb, extrinsic))
                


                # from obj local to world (whole scene) coordinates
                origin_world, axis_vector_world = transform_origin_and_vector_to_world(origin_3d, axis_vector, np.linalg.inv(extrinsic))
                obb_world = transform_obb_to_world(obb, extrinsic)

                # saving data
                pred_score = pred.scores.cpu().numpy()
                np.save(f"{curr_output_dir}/score.npy", pred_score)
                np.save(f"{curr_output_dir}/range_max.npy", range_max)
                np.save(f"{curr_output_dir}/origin_local.npy", translated_origin_3d)        # articulation origin in local o3d space
                np.save(f"{curr_output_dir}/vector_local.npy", translated_axis_vector)      # articulation vector in local o3d space
                np.save(f"{curr_output_dir}/origin_world.npy", origin_world)                # articulation origin in world/scene o3d space
                np.save(f"{curr_output_dir}/vector_world.npy", axis_vector_world)           # articulation vector in world/scene o3d space
                np.save(f"{curr_output_dir}/articulation_type.npy", pred_type)              
                np.save(f"{curr_output_dir}/mask.npy", mask)
                np.save(f"{curr_output_dir}/obb_world.npy", obb_world)  # mask coverage ratio

                save_image_bbx(rgb_image, mask, curr_output_dir)

                # transform pcd from local to world coordinates
                box_pcd_world_o3d = deepcopy(box_pcd)
                box_pcd_world_o3d = transform_object_pcd_to_world(box_pcd_world_o3d, np.linalg.inv(extrinsic))
                o3d.io.write_point_cloud(f"{curr_output_dir}/base.ply", box_pcd_world_o3d)
                shutil.copy2(mask_mesh_path, f"{curr_output_dir}/part.ply")

                # box_pcd_world = transform_object_pcd_to_world(box_pcd, extrinsic)
                # part_pcd_world = transform_object_pcd_to_world(mask_pcd, extrinsic)
                
                # create mesh
                # part_mesh = pcd_to_mesh_poisson(part_pcd_world)
                part_mesh = o3d.io.read_triangle_mesh(mask_mesh_path)
                box_mesh, inner_box_mesh = corners_to_mesh(cuboid_world, pred_type)

                # texture = vertex_colors_to_texture(part_mesh, 1024)
                # Image.fromarray(texture).save(f"{curr_output_dir}/part.png")

                base_obj_path = f"{curr_output_dir}/base.obj"
                part_obj_path = f"{curr_output_dir}/part.obj"
                if inner_box_mesh:
                    inner_box_obj_path = f"{curr_output_dir}/inner_box.obj"
                    save_mesh_with_mtl(inner_box_obj_path, inner_box_mesh, color=cuboid_world["color"])
                save_mesh_with_mtl(base_obj_path, box_mesh, color=cuboid_world["color"])
                save_mesh_with_mtl(part_obj_path, part_mesh)
                # generate_texture_from_mesh(part_mesh, 1024, part_obj_path)
                

                # o3d.io.write_triangle_mesh(f"{curr_output_dir}/base.obj", box_mesh)         # base mesh in URDF space
                # o3d.io.write_triangle_mesh(f"{curr_output_dir}/part.obj", part_mesh)        # part mesh in URDF space
                # mask_pcd.paint_uniform_color([1, 0, 0])
                # o3d.visualization.draw_geometries([mask_pcd, part_mesh], window_name="Mesh Visualization")
                
                # visualize URDF result in o3d space
                # part_mesh = transform_mesh_coord_open3d_to_urdf(part_mesh)
                # box_mesh = transform_mesh_coord_open3d_to_urdf(box_mesh)
                # if inner_box_mesh:
                #     inner_box_mesh = transform_mesh_coord_open3d_to_urdf(inner_box_mesh)
                # # translated_arrow = transform_mesh_coord_open3d_to_urdf(translated_arrow)

                # translated_origin_3d = transform_coordinates_open3d_to_urdf(origin_world)
                # translated_axis_vector = transform_coordinates_open3d_to_urdf(axis_vector_world)
                # arrow_tmp = draw_line(translated_origin_3d, translated_axis_vector + translated_origin_3d, depth_scale)
                # vis_mesh_base_part(box_mesh, part_mesh, arrow_tmp, inner_box_mesh)

                # visualize articulation result in whole scene
                # scene_pcd = o3d.io.read_triangle_mesh("/home/troye/ssd/datasets/scannet++/data/40aec5fffa/scans/mesh_aligned_0.05.ply")
                # axis_arrow_world = visualize_articulation_in_scene(scene_pcd, origin_world, axis_vector_world, pred_type)

                # object_combined_pcd_transformed =  transform_object_pcd_to_world(object_combined_pcd, extrinsic)
                # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
                # o3d.visualization.draw_geometries([ axis_arrow_world, coordinate_frame], window_name="Mesh Visualization")

                if pred_type:
                    if crop:  # crop images to remove shared extraneous whitespace
                        output_dir_cropped = f"{curr_output_dir}_cropped"
                        if not os.path.isdir(output_dir_cropped):
                            os.makedirs(output_dir_cropped)
                        batch_trim(curr_output_dir, output_dir_cropped, identical=True)
                        # create_gif(output_dir_cropped, num_samples)
                    else:  # leave original dimensions of image as-is
                        # create_gif(curr_output_dir, num_samples)
                        pass
                
                pred_idx += 1
                success = True
                break
            
        
    # print("Bounding box list len: ", len(bounding_box_list))
    # print(bounding_box_list)

    # for box_dict in bounding_box_list:
    #     if box_dict["valid"] == True:
    #         continue
        
    #     new_name = box_dict["path"] + "_not_valid"
    #     os.rename(box_dict["path"], new_name)


if __name__ == "__main__":
    # parse arguments
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    scene_ids = ["realscan_landscape_format"]

    # scene_ids = [ 
    #     "scene_00093_00", "scene_00096_00"
    # ]

    # scene_ids = ["scene_00006_00","scene_00022_00", "scene_00030_01", "scene_00065_00", "scene_00086_00", "scene_00089_00", 
    # ]

    # scene_ids = ["c0f5742640",
    # ]

    for sid in scene_ids:
        print("Processing scene: ", sid)
        # args.scene_mesh = f"/home/troye/ssd/datasets/scannet++/data/{sid}/scans/mesh_aligned_0.05.ply"
        args.scene_mesh = f"/home/troye/ssd/Zhao_SP/scene2obj/segmentation_data/{sid}/mesh_aligned_0.05.ply"
        args.folder = f"/home/troye/ssd/Zhao_SP/scene2obj/segmentation_data/{sid}/perception/vis_groups_final_mesh"
        intrinsic_json_path = f"/home/troye/ssd/Zhao_SP/scene2obj/segmentation_data/{sid}/pose_intrinsic_imu_mvp.json"

        current_intrinsics = load_scaled_intrinsics(intrinsic_json_path, scale=3.75)
        if current_intrinsics is None:
            current_intrinsics = K_SAME

        scene_output_dir = os.path.join(cfg.OUTPUT_DIR, args.folder.split("/")[-3])
        start_time = time.time()
        # run main code
        engine.launch(
            main,
            args.num_processes,
            args=(
                cfg,
                # args.rgb_image,
                # args.depth_image,
                # args.extrinsics_npy,
                args.folder,
                args.scene_mesh,
                current_intrinsics,
                args.num_samples,
                args.crop,
                args.score_threshold,
                args.vis,
            ),
        )
        end_time = time.time()
        duration = end_time - start_time
        print(f"Module 4: Articulated Objects Generation -- {duration:.2f} seconds")

        # --- TIME BREAKDOWN LOGGING ---
        save_time_breakdown(
            input_dir=scene_output_dir,
            output_dir=scene_output_dir,
            module_name="Module 4: Articulated Objects Generation",
            start_time=start_time,
            end_time=end_time,
            duration=duration
        )