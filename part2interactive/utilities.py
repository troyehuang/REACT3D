import numpy as np
import pycocotools.mask as mask_util
from detectron2.structures import BoxMode
import math
import re
import os
import shutil
from collections import defaultdict
from scipy.spatial import KDTree
from PIL import Image
from scipy.spatial import ConvexHull, HalfspaceIntersection

from detectron2.utils.visualizer import (
    Visualizer,
    ColorMode,
    _create_text_labels,
    GenericMask,
)
import cv2
import open3d as o3d
import time, csv, json


# MotionNet: based on instances_to_coco_json and relevant codes in densepose
def prediction_to_json(instances, img_id: str):
    """
    Args:
        instances (Instances): the output of the model
        img_id (str): the image id in COCO

    Returns:
        list[dict]: the results in densepose evaluation format
    """
    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    # Prediction for MotionNet
    # mtype = instances.mtype.squeeze(axis=1).tolist()

    # 2.0.3
    if instances.has("pdim"):
        pdim = instances.pdim.tolist()
    if instances.has("ptrans"):
        ptrans = instances.ptrans.tolist()
    if instances.has("prot"):
        prot = instances.prot.tolist()

    mtype = instances.mtype.tolist()
    morigin = instances.morigin.tolist()
    maxis = instances.maxis.tolist()
    mstate = instances.mstate.tolist()
    mstatemax = instances.mstatemax.tolist()
    if instances.has("mextrinsic"):
        mextrinsic = instances.mextrinsic.tolist()

    # if motionstate:
    #     mstate = instances.mstate.tolist()

    # MotionNet has masks in the annotation
    # use RLE to encode the masks, because they are too large and takes memory
    # since this evaluator stores outputs of the entire dataset
    rles = [mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0] for mask in instances.pred_masks]
    for rle in rles:
        # "counts" is an array encoded by mask_util as a byte-stream. Python3's
        # json writer which always produces strings cannot serialize a bytestream
        # unless you decode it. Thankfully, utf-8 works out (which is also what
        # the pycocotools/_mask.pyx does).
        rle["counts"] = rle["counts"].decode("utf-8")

    results = []
    for k in range(len(instances)):
        if instances.has("pdim"):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
                "segmentation": rles[k],
                "pdim": pdim[k],
                "ptrans": ptrans[k],
                "prot": prot[k],
                "mtype": mtype[k],
                "morigin": morigin[k],
                "maxis": maxis[k],
                "mstate": mstate[k],
                "mstatemax": mstatemax[k],
            }
        elif instances.has("mextrinsic"):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
                "segmentation": rles[k],
                "mtype": mtype[k],
                "morigin": morigin[k],
                "maxis": maxis[k],
                "mextrinsic": mextrinsic[k],
                "mstate": mstate[k],
                "mstatemax": mstatemax[k],
            }
        else:
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
                "segmentation": rles[k],
                "mtype": mtype[k],
                "morigin": morigin[k],
                "maxis": maxis[k],
                "mstate": mstate[k],
                "mstatemax": mstatemax[k],
            }
        # if motionstate:
        #     result["mstate"] = mstate[k]
        results.append(result)
    return results

def getFocalLength(FOV, height, width=None):
    # FOV is in radius, should be vertical angle
    if width == None:
        f = height / (2 * math.tan(FOV / 2))
        return f
    else:
        fx = height / (2 * math.tan(FOV / 2))
        fy = fx / height * width
        return (fx, fy)
    
def camera_to_image(point, is_real=False, intrinsic_matrix=None):
    point_camera = np.array(point)
    # Calculate the camera intrinsic parameters (they are fixed in this project)
    if not is_real:
        # Below is for the MoionNet synthetic dataset intrinsic
        FOV = 50
        img_width = img_height = 256
        fx, fy = getFocalLength(FOV / 180 * math.pi, img_height, img_width)
        cy = img_height / 2
        cx = img_width / 2
        x = point_camera[0] * fx / (-point_camera[2]) + cx
        y = -(point_camera[1] * fy / (-point_camera[2])) + cy
    else:
        # Below is the for MotionREAL dataset 
        point_2d = np.dot(intrinsic_matrix, point_camera[:3])
        x = point_2d[0] / point_2d[2]
        y = point_2d[1] / point_2d[2]

    return (x, y)

def rotatePoint(x, y, angle, scale):
    rad = np.pi * angle / 180
    x2 = np.cos(rad) * x - np.sin(rad) * y
    y2 = np.sin(rad) * x + np.cos(rad) * y
    return [x2 * scale, y2 * scale]

def rotation_from_vectors(source, dest):
    a, b = (source / np.linalg.norm(source)).reshape(3), (
        dest / np.linalg.norm(dest)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rmat = np.eye(3) + kmat + np.matmul(kmat, kmat) * ((1 - c) / (s ** 2))
    return rmat

def circlePoints(axis, radius=0.5, num=50):
    angles = np.linspace(0, 2 * np.pi, num, endpoint=False)
    x_vec = np.cos(angles) * radius
    y_vec = np.sin(angles) * radius
    z_vec = np.zeros_like(x_vec) + 0.5
    points = np.stack((x_vec, y_vec, z_vec), axis=0)
    rot = rotation_from_vectors(np.array([0, 0, 1]), np.asarray(axis))
    points = np.matmul(rot, points)
    return points

def camera_to_world(point, extrinsic):
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]

    return R.T @ (point - T)

# origin and axis_vector are not rotated
def transform_origin_and_vector_to_world(origin, axis_vector, extrinsic):
    origin_world = camera_to_world(origin, extrinsic)
    axis_end_camera = origin + axis_vector
    axis_end_world = camera_to_world(axis_end_camera, extrinsic)
    axis_vector_world = axis_end_world - origin_world

    return origin_world, axis_vector_world

# pcd is not rotated
def transform_object_pcd_to_world(pcd, extrinsic):
    # to homogenous coordinates
    points = np.asarray(pcd.points)
    N = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((N, 1))))  # transform to homogeneous coordinates (Nx4)
    
    # inv matrix
    extrinsic_inv = np.linalg.inv(extrinsic)  # 4x4 extrinsic inverse matrix
    
    # back to euclidean coordinates
    transformed_points = (extrinsic_inv @ homogeneous_points.T).T[:, :3]  # transform to Nx3
    
    pcd.points = o3d.utility.Vector3dVector(transformed_points)
    
    return pcd


def transform_obj_mesh_to_world(mesh: o3d.geometry.TriangleMesh, extrinsic: np.ndarray, inplace: bool = False):
    M = mesh if inplace else o3d.geometry.TriangleMesh(mesh)  # Copy

    # Prepare vertex homogeneous coordinates
    verts = np.asarray(M.vertices)
    if verts.size == 0:
        return M

    ones = np.ones((verts.shape[0], 1), dtype=verts.dtype)
    homo = np.hstack([verts, ones])

    # Use inverse extrinsic matrix (consistent with pcd logic)
    T_inv = np.linalg.inv(extrinsic)

    # Vertex coordinate transformation
    verts_tf = (T_inv @ homo.T).T[:, :3]
    M.vertices = o3d.utility.Vector3dVector(verts_tf)

    # Normal transformation: only use linear part and normalize
    if M.has_vertex_normals():
        linear = T_inv[:3, :3]

        # If extrinsic is purely rigid, linear is sufficient directly here;
        # For safety, extract the closest rotation matrix using SVD to avoid stretching affecting normals.
        U, _, Vt = np.linalg.svd(linear)
        R = U @ Vt

        norms = np.asarray(M.vertex_normals)
        norms_tf = (R @ norms.T).T
        norms_tf /= (np.linalg.norm(norms_tf, axis=1, keepdims=True) + 1e-12)
        M.vertex_normals = o3d.utility.Vector3dVector(norms_tf)
    else:
        # If no normals, recommend recomputing
        M.compute_vertex_normals()

    return M

def transform_cuboid_corners_to_world(corners, extrinsic):
    corners = np.array(corners, dtype=np.float64)
    
    # to homogenous coordinates
    N = corners.shape[0]
    homogeneous_corners = np.hstack((corners, np.ones((N, 1))))  # transform to homogeneous coordinates (Nx4)
    
    # inv matrix
    extrinsic_inv = extrinsic  # 4x4 extrinsic inverse matrix
    
    # back to euclidean coordinates
    transformed_corners = (extrinsic_inv @ homogeneous_corners.T).T[:, :3]  # transform to Nx3
    
    return transformed_corners.tolist()

def transform_obb_to_world(obb, extrinsic):
    """
    Transform an (N,3) set of points (e.g., the 8x3 OBB corners) to the world coordinate frame using the provided extrinsic matrix.
    
    Parameters:
      points : NumPy array of shape (N,3), representing the points to be transformed
      extrinsic : 4x4 extrinsic matrix (typically the camera extrinsics), whose inverse will be used for the transformation
      
    returns:
      Transformed NumPy array of shape (N,3)
    """
    # ensure points is a NumPy array with dtype float64
    points = np.array(obb, dtype=np.float64)
    
    # convert to homogeneous coordinates, resulting in shape (N, 4)
    N = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((N, 1))))
    
    # compute the inverse of the extrinsic matrix
    extrinsic_inv = extrinsic
    
    # apply the transformation and convert back to Euclidean coordinates (N, 3)
    transformed_points = (extrinsic_inv @ homogeneous_points.T).T[:, :3]
    
    return transformed_points

def group_files_by_id(folder):
    pattern = re.compile(r'^(rgb|depth|extrinsic)_(\d{8}_\d{6})')  # match timestamp format like 20231001_123456
    groups = defaultdict(dict)
    
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            category, timestamp = match.groups()
            groups[timestamp][category] = os.path.join(folder, filename)
    
    return groups

def group_real_files_by_id(folder):
    # match depth_frame_123.npy、rgb_frame_123.png、extrinsic_frame_123.npy 
    pattern = re.compile(r'^(depth|rgb|extrinsic|visible_points_mask)_frame_(\d+)\.(npy|png)$')
    groups = {}
    
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        
        group = {}
        for filename in os.listdir(subfolder_path):
            match = pattern.match(filename)
            if match:
                category, _, _ = match.groups()
                group[category] = os.path.join(subfolder_path, filename)
        
        if all(key in group for key in ('depth', 'rgb', 'extrinsic', 'visible_points_mask')):
            groups[subfolder] = group

    return groups

def group_pickle_and_mesh_by_id(folder):
    # match depth_frame_123.npy、rgb_frame_123.png、extrinsic_frame_123.npy 
    pickle_name = "top_k_list.pkl"
    mask_mesh_name = "mask_mesh_refined.ply"
    groups = {}
    
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
        
        mask_mesh_path = os.path.join(subfolder_path, mask_mesh_name)
        
        if not os.path.exists(mask_mesh_path):
            continue
            
        content = {}
        content["pickle_path"] = os.path.join(subfolder_path, pickle_name)
        content["mask_mesh_path"] = mask_mesh_path

        groups[subfolder] = content
        
    return groups

def fit_2d_point_onto_3d_pointcloud(origin_2d, pcd, intrinsic_matrix, depth_np, depth_scale=1., transform_matrix=None):

    h, w = depth_np.shape  
    u, v = int(origin_2d[0]), int(origin_2d[1])

    # ensure the pixel coordinates are within the image bounds
    if not (0 <= u < w and 0 <= v < h):
        return None
    
    # get depth value at the pixel
    depth = depth_np[v, u] / depth_scale
    if depth == 0:
        return None

    # compute the 3D point in camera coordinates
    point_3d = np.array([origin_2d[0], origin_2d[1], 1]) * depth
    point_3d = np.dot(np.linalg.inv(intrinsic_matrix), point_3d)
    point_3d = np.append(point_3d, 1)

    if transform_matrix is not None:
        point_3d = np.dot(np.linalg.inv(transform_matrix), point_3d)

    point_3d = point_3d[:3] / point_3d[3]

    return point_3d


def find_nearest_mask_point(mask, origin_point):
    """
    If origin_point lies outside the mask, find the nearest point within the mask
    :param mask: Binary mask of shape (H, W), with values 0 or 1
    :param origin_point: (x, y) original point coordinates
    :return: (x, y) coordinates of the nearest mask point
    """
    y_indices, x_indices = np.where(mask == 1)  # Get coordinates of all mask points
    mask_points = np.column_stack((x_indices, y_indices))  # (N, 2) format

    if len(mask_points) == 0:
        raise ValueError("No points found in the mask.")

    # check if origin_point is already on the mask
    ox, oy = origin_point
    H, W = mask.shape
    if (0 <= ox < W and 0 <= oy < H) and mask[oy, ox] == 1:
        return origin_point 

    # use KDTree to find the nearest point in the mask
    tree = KDTree(mask_points)
    _, nearest_idx = tree.query([ox, oy])
    nearest_point = tuple(mask_points[nearest_idx])  # (x, y)

    return nearest_point

def save_img_with_unique_name(output_dir, base_filename):

    output_filename = os.path.join(output_dir, base_filename)
    counter = 2
    
    while os.path.exists(output_filename):
        name, ext = os.path.splitext(base_filename)
        output_filename = os.path.join(output_dir, f"{name}_{counter}{ext}")
        counter += 1
    
    return output_filename

def get_obb_max_plane_normal(obb_corners):
    """
    Compute the normal vector of the largest plane of the OBB
    :param obb_corners: numpy array, shape (8, 3), representing 8 corners of OBB
    :return: normal vector of the largest plane and center point
    """
    faces = [
        [0, 1, 2, 3],  # bot
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [1, 2, 6, 5],  # right
        [2, 3, 7, 6],  # back
        [3, 0, 4, 7]   # left
    ]
    
    max_area = 0
    max_face = None
    max_normal = None
    max_center = None
    
    for face in faces:
        # get 4 corners of the face
        face_points = obb_corners[face]
        
        # compute the area of the face using cross product
        v1 = face_points[1] - face_points[0]
        v2 = face_points[2] - face_points[0]
        area = np.linalg.norm(np.cross(v1, v2))
        
        # update the maximum area and corresponding face if needed
        if area > max_area:
            max_area = area
            max_face = face_points
            
            max_normal = np.cross(v1, v2)
            max_normal = max_normal / np.linalg.norm(max_normal)
            
            max_center = np.mean(face_points, axis=0)
    
    return max_normal, max_center

def rotate_pcd(pcd, center):
    angle_x = np.pi * 5.5 / 5  # 198 degrees
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.asarray([1, 0, 0]) * angle_x)
    pcd.rotate(R, center=center)
    
    return pcd

def rotate_corners(corners, center):
    """
    Rotate the eight corner coordinates of the cuboid
    :param corners: NumPy array of shape (8, 3) representing the cuboid’s corner coordinates
    :param center: Center of rotation
    :return: Corner coordinates of the cuboid after rotation
    """
    angle_x = np.pi * 5.5 / 5  # 198 degrees
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.asarray([1, 0, 0]) * angle_x)
    
    rotated_corners = []
    for corner in corners:
        offset = corner - center
        # apply rotation
        rotated_corner = np.dot(R, offset) + center
        rotated_corners.append(rotated_corner)
    
    return np.array(rotated_corners)

def rotate_point(point, center):
    angle_x = np.pi * 5.5 / 5  # 198 degrees
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * angle_x)

    translated_point = point - center

    rotated_translated_point = R @ translated_point

    rotated_point = rotated_translated_point + center

    return rotated_point

def pcd_to_mesh_poisson(pcd):
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    
    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.1)
    
    # generate a mask for vertices to remove based on density threshold
    vertices_to_remove = densities < threshold
    
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh

def create_cuboid_mesh(cuboid):

    width, height, depth = cuboid["width"], cuboid["height"], cuboid["depth"]
    color = cuboid["color"]

    # define 8 vertices of the cuboid
    vertices = np.array([
        [0, 0, 0],  # 0
        [width, 0, 0],  # 1
        [width, height, 0],  # 2
        [0, height, 0],  # 3
        [0, 0, depth],  # 4
        [width, 0, depth],  # 5
        [width, height, depth],  # 6
        [0, height, depth]  # 7
    ])

    # define the 12 triangular faces of the cuboid
    # Each face is defined by 2 triangles
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # front
        [4, 5, 6], [4, 6, 7],  # back
        [0, 1, 5], [0, 5, 4],  # bottom
        [2, 3, 7], [2, 7, 6],  # top
        [1, 2, 6], [1, 6, 5],  # right
        [0, 3, 7], [0, 7, 4]   # left
    ])

    mesh = o3d.geometry.TriangleMesh()

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (len(vertices), 1)))

    mesh.compute_vertex_normals()

    return mesh

# def corners_to_mesh(cuboid):
#     """
#     将长方体的顶点数组转换为双面渲染的 Mesh 对象
#     :param cuboid: 包含 "corners"（8个顶点坐标）和 "color" 的字典
#     :return: Open3D Mesh 对象
#     """

#     corners = np.array(cuboid["corners"])
#     color = np.array(cuboid["color"])

#     # 定义面，每个面由 2 个三角形组成
#     faces = np.array([
#         [0, 1, 2], [1, 2, 3],  # 前面
#         [4, 5, 6], [5, 6, 7],  # 后面
#         [0, 1, 4], [1, 4, 5],  # 左面
#         [2, 3, 6], [3, 6, 7],  # 右面
#         # [0, 2, 4], [2, 4, 6],  # 底面
#         [1, 3, 5], [3, 5, 7]   # 顶面
#     ])

#     # 生成翻转的面，确保双面渲染
#     flipped_faces = faces[:, ::-1]

#     # 合并正面和反面的三角形
#     all_faces = np.vstack((faces, flipped_faces))

#     # 创建 Mesh 对象
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices = o3d.utility.Vector3dVector(corners)
#     mesh.triangles = o3d.utility.Vector3iVector(all_faces)

#     # 赋予颜色
#     mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (len(corners), 1)))

#     # 计算法线
#     mesh.compute_vertex_normals()

#     return mesh

def corners_to_mesh(cuboid, pred_type, thickness_ratio=0.03):
    """
    Convert the cuboid's corner array into a mesh object with thickness.
    When pred_type=="translation", also generate an inner_box mesh,
    where the inner_box's outer layer matches the original box's inner layer, and its inner layer is offset by the thickness.
    
    :param cuboid: Dictionary containing "corners" (8 corner coordinates) and "color"
    :param pred_type: Determines type; when "translation", generate the inner_box mesh
    :param thickness_ratio: Scaling factor for thickness, based on the shortest edge of the "left" face (default 0.03)
    :return: Returns (box_mesh, inner_box_mesh) if pred_type=="translation"; otherwise returns box_mesh
    """
    corners = np.array(cuboid["corners"])
    color = np.array(cuboid["color"])
    
    cuboid_center = np.mean(corners, axis=0)
    
    faces_def = {
        "front":  [0, 1, 3, 2],
        "back":   [4, 5, 7, 6],
        # "left":   [0, 2, 6, 4],
        "right":  [1, 3, 7, 5],
        "top":    [2, 3, 7, 6],
        "bottom": [0, 1, 5, 4]
    }
    
    # compute the thickness based on the left face
    left_indices = [0, 2, 6, 4]
    left_face = corners[left_indices]
    edge_lengths = [np.linalg.norm(left_face[(i+1)%4] - left_face[i]) for i in range(4)]
    min_edge = min(edge_lengths)
    thickness = thickness_ratio * min_edge
    
    # construct box_mesh: for each face generate two sets of vertices—outer (original) and inner (offset by thickness),
    # and create triangles for each face (outer face, inner face, and side walls), then duplicate and flip each triangle to ensure double-sided rendering
    vertices_list = []
    triangles = []
    vertex_offset = 0
    
    # if pred_type=="translation", record each face’s inner layer to use as the outer layer of the inner_box
    inner_layer_faces = {}
    
    for key, indices in faces_def.items():
        outer = corners[indices] 
        face_center = np.mean(outer, axis=0)
        
        normal = np.cross(outer[1] - outer[0], outer[2] - outer[0])
        normal /= np.linalg.norm(normal)
        # ensure the normal points inside the cuboid
        if np.dot(normal, cuboid_center - face_center) < 0:
            normal = -normal
        
        inner = outer + thickness * normal
        
        inner_layer_faces[key] = (outer, inner, normal)
        
        face_vertices = np.vstack([outer, inner])  # shape = (8,3)
        vertices_list.append(face_vertices)
        n = outer.shape[0]  
        
        # 1. Outer face: split into two triangles
        triangles.append([vertex_offset, vertex_offset+1, vertex_offset+2])
        triangles.append([vertex_offset, vertex_offset+2, vertex_offset+3])
        # 2. Inner face: also split, but flip vertex order so its normal points inward
        triangles.append([vertex_offset+n, vertex_offset+n+2, vertex_offset+n+1])
        triangles.append([vertex_offset+n, vertex_offset+n+3, vertex_offset+n+2])
        # 3. Side walls: for each corresponding edge of outer and inner layers, generate two triangles
        for i in range(n):
            i_next = (i+1) % n
            v0 = vertex_offset + i         
            v1 = vertex_offset + i_next   
            v2 = vertex_offset + n + i_next 
            v3 = vertex_offset + n + i    
            triangles.append([v0, v1, v2])
            triangles.append([v0, v2, v3])
        
        vertex_offset += 2 * n

    # Combine all face vertices and triangles
    vertices = np.vstack(vertices_list)
    triangles = np.array(triangles, dtype=np.int32)
    # Double-sided rendering: duplicate and flip every triangle
    flipped_triangles = triangles[:, ::-1]
    all_triangles = np.vstack((triangles, flipped_triangles))
    
    box_mesh = o3d.geometry.TriangleMesh()
    box_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    box_mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
    box_mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (vertices.shape[0], 1)))
    box_mesh.compute_vertex_normals()
    
    inner_box_mesh = None

    # generate inner_box mesh
    if pred_type == "translation":
        inner_box_vertices_list = []
        inner_box_triangles = []
        vertex_offset_inner = 0

        # Identify which faces to remove:
        # 1. The face that corresponds to the part mesh (consistently the 'left' / pjmin face)
        # 2. The physical 'top' face (highest in world space)
        
        # Calculate world-space centers for all faces
        face_centers = {k: np.mean(corners[idxs], axis=0) for k, idxs in faces_def.items()}
        
        # Top face is the one with highest Z in world space (ScanNet++ standard)
        # If your 'Top' is usually Y, change the index from [2] to [1]
        top_face_key = max(face_centers, key=lambda k: face_centers[k][2])
        part_face_key = "left" # This corresponds to the pjmin plane where the part resides

        for key, indices in faces_def.items():
            if key == part_face_key or key == top_face_key:
                continue
            
            outer_face, inner_face, normal = inner_layer_faces[key]
            inner_box_outer = inner_face
            
            inner_box_inner = inner_box_outer + thickness * normal
            
            face_vertices_inner_box = np.vstack([inner_box_outer, inner_box_inner])
            inner_box_vertices_list.append(face_vertices_inner_box)
            n = inner_box_outer.shape[0]
            
            
            inner_box_triangles.append([vertex_offset_inner, vertex_offset_inner+1, vertex_offset_inner+2])
            inner_box_triangles.append([vertex_offset_inner, vertex_offset_inner+2, vertex_offset_inner+3])
            
            inner_box_triangles.append([vertex_offset_inner+n, vertex_offset_inner+n+2, vertex_offset_inner+n+1])
            inner_box_triangles.append([vertex_offset_inner+n, vertex_offset_inner+n+3, vertex_offset_inner+n+2])
            
            for i in range(n):
                i_next = (i + 1) % n
                v0 = vertex_offset_inner + i
                v1 = vertex_offset_inner + i_next
                v2 = vertex_offset_inner + n + i_next
                v3 = vertex_offset_inner + n + i
                inner_box_triangles.append([v0, v1, v2])
                inner_box_triangles.append([v0, v2, v3])
            vertex_offset_inner += 2 * n
        
        inner_box_vertices = np.vstack(inner_box_vertices_list)
        inner_box_triangles = np.array(inner_box_triangles, dtype=np.int32)
        flipped_triangles_inner = inner_box_triangles[:, ::-1]
        all_triangles_inner = np.vstack((inner_box_triangles, flipped_triangles_inner))
        
        inner_box_mesh = o3d.geometry.TriangleMesh()
        inner_box_mesh.vertices = o3d.utility.Vector3dVector(inner_box_vertices)
        inner_box_mesh.triangles = o3d.utility.Vector3iVector(all_triangles_inner)
        inner_box_mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(color, (inner_box_vertices.shape[0], 1)))
        inner_box_mesh.compute_vertex_normals()
        
    # outer box mesh and inner box mesh
    return box_mesh, inner_box_mesh


def move_mesh_to_origin(mesh):
    vertices = np.asarray(mesh.vertices)
    
    centroid = np.mean(vertices, axis=0)
    
    vertices -= centroid
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    return mesh

def move_mesh_to_target(mesh, target_position):
    
    vertices = np.asarray(mesh.vertices)
    
    centroid = np.mean(vertices, axis=0)
    
    translation = target_position - centroid
    
    vertices += translation
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    return mesh

def transform_mesh_coord_open3d_to_urdf(mesh):

    vertices = np.asarray(mesh.vertices)
    
    vertices_urdf = np.stack([
        vertices[:, 2],  # Z → X 
        vertices[:, 0],  # X → Y 
        vertices[:, 1]   # Y → Z 
    ], axis=-1)
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices_urdf)
    
    return mesh

def transform_pcd_coord_open3d_to_urdf(pcd):
    # Extract points
    pts = np.asarray(pcd.points)
    # Rearrange
    pts_urdf = np.stack([
        pts[:, 2],  # Z → X
        pts[:, 0],  # X → Y
        pts[:, 1],  # Y → Z
    ], axis=-1)
    # Assign back
    pcd.points = o3d.utility.Vector3dVector(pts_urdf)

    # If it has normals, transform them similarly
    if pcd.has_normals():
        nrm = np.asarray(pcd.normals)
        nrm_urdf = np.stack([
            nrm[:, 2],  # Z → X
            nrm[:, 0],  # X → Y
            nrm[:, 1],  # Y → Z
        ], axis=-1)
        pcd.normals = o3d.utility.Vector3dVector(nrm_urdf)

    return pcd

def transform_coordinates_open3d_to_urdf(point):

    x, y, z = point
    return np.array([z, x, y])

# def save_mesh_with_mtl(obj_filename, mesh, color):
#     """
#     保存带颜色的 OBJ 文件，并生成对应的 MTL 材质文件（支持 RViz）
#     :param obj_filename: .obj 文件路径
#     :param mesh: Open3D 生成的 mesh 对象
#     :param color: (R, G, B) 颜色，范围 0~1
#     """
#     mtl_filename = obj_filename.replace('.obj', '.mtl')

#     # 保存 mesh 为 OBJ
#     o3d.io.write_triangle_mesh(obj_filename, mesh)

#     # 生成 MTL 文件
#     mat_name = "material_0"
#     with open(mtl_filename, 'w') as f:
#         f.write(f"""# MTL file generated by Open3D
# newmtl {mat_name}
# Kd {color[0]} {color[1]} {color[2]}  # 漫反射颜色
# """)

#     # 修改 OBJ 以引用 MTL
#     with open(obj_filename, 'r') as f:
#         lines = f.readlines()

#     with open(obj_filename, 'w') as f:
#         f.write(f"mtllib {mtl_filename.split('/')[-1]}\n")  # 添加材质引用
#         f.write(f"usemtl {mat_name}\n")
#         for line in lines:
#             f.write(line)

#     print(f"Saved {obj_filename} with material {mtl_filename}")

def save_mesh_with_mtl(obj_filename, mesh, color=None, texture_file=None):
    # if color is None and texture_file is None:
    #     raise ValueError("Must provide color or texture map path!")

    # mtl_filename = obj_filename.replace('.obj', '.mtl')
    # mat_name = "material_0"

    # Save mesh as OBJ
    o3d.io.write_triangle_mesh(obj_filename, mesh)

    # Generate MTL file
    # with open(mtl_filename, 'w') as f:
    #     f.write(f"# MTL file generated by Open3D\n")
    #     f.write(f"newmtl {mat_name}\n")
    #     if color is not None:
    #         f.write(f"Kd {color[0]} {color[1]} {color[2]}  # Diffuse color\n")
    #     if texture_file is not None:
    #         # Use relative path to reference texture map
    #         texture_filename = texture_file.split('/')[-1]  # Extract filename
    #         f.write(f"map_Kd {texture_filename}\n")

    # Modify OBJ to reference MTL
    # with open(obj_filename, 'r') as f:
    #     lines = f.readlines()

    # with open(obj_filename, 'w') as f:
    #     if texture_file is None:
    #         f.write(f"mtllib {mtl_filename.split('/')[-1]}\n")  # Add material reference
    #         f.write(f"usemtl {mat_name}\n")
    #     for line in lines:
    #         f.write(line)

    print(f"Saved {obj_filename}")

def vertex_colors_to_texture(mesh, texture_size=1024):
    if not mesh.has_vertex_colors():
        raise ValueError("Mesh has no vertex color information!")

    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors) 

    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    uv_coords = (vertices - min_vals) / (max_vals - min_vals) 
    uv_coords = uv_coords[:, :2]  

    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)

    for i in range(len(uv_coords)):
        u, v = uv_coords[i]
        x = int(u * (texture_size - 1))
        y = int(v * (texture_size - 1))
        texture[y, x] = (colors[i] * 255).astype(np.uint8)

    return texture

def generate_texture_from_mesh(mesh, texture_size=1024, obj_filename="output.obj"):

    if not mesh.has_vertex_colors():
        raise ValueError("Mesh has no vertex color information!")

    mesh.compute_uvatlas() 
    uv_coords = np.asarray(mesh.triangle_uvs)  

    colors = np.asarray(mesh.vertex_colors) 

    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)

    for i in range(len(uv_coords)):
        u, v = uv_coords[i]
        x = int(u * (texture_size - 1))
        y = int((1 - v) * (texture_size - 1)) 
        if x < texture_size and y < texture_size:  
            texture[y, x] = (colors[i // 3] * 255).astype(np.uint8)  

    texture_filename = obj_filename.replace('.obj', '.png')
    Image.fromarray(texture).save(texture_filename)
    print(f"Texture map saved to: {texture_filename}")

    mtl_filename = obj_filename.replace('.obj', '.mtl')
    mat_name = "material_0"
    with open(mtl_filename, 'w') as f:
        f.write(f"# MTL file generated by Open3D\n")
        f.write(f"newmtl {mat_name}\n")
        f.write(f"Kd 1.0 1.0 1.0\n") 
        f.write(f"map_Kd {texture_filename}\n") 

    with open(obj_filename, 'w') as f:
        f.write(f"# OBJ file generated by Open3D\n")
        # f.write(f"mtllib {mtl_filename}\n")  # Reference .mtl file
        # f.write(f"usemtl {mat_name}\n")

        for vertex in mesh.vertices:
            f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        for uv in uv_coords:
            f.write(f"vt {uv[0]} {1 - uv[1]}\n") 

        triangles = np.asarray(mesh.triangles)
        for i, triangle in enumerate(triangles):
            v0 = triangle[0] + 1  
            v1 = triangle[1] + 1
            v2 = triangle[2] + 1
            uv0 = i * 3 + 1 
            uv1 = i * 3 + 2
            uv2 = i * 3 + 3
            f.write(f"f {v0}/{uv0} {v1}/{uv1} {v2}/{uv2}\n")

    print(f"OBJ file saved to: {obj_filename}")
    print(f"MTL file saved to: {mtl_filename}")

def copy_all(src, dest):
    """
    Copies all files and folders from the src directory to the dest directory.
    
    Parameters:
    - src: Source directory path.
    - dest: Destination directory path.
    """
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest):
        os.makedirs(dest)
    
    # Iterate over all items in the source directory
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dest_item = os.path.join(dest, item)
        if os.path.isdir(src_item):
            # Copy the entire directory. 'dirs_exist_ok=True' allows copying into an existing directory (Python 3.8+)
            shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
        else:
            # Copy the file, preserving metadata
            shutil.copy2(src_item, dest_item)

def compute_intersection_volume(bbx1, bbx2):
    """
    compute the intersection volume of two 3D bounding boxes defined by their corner points.
    """
    hull1 = ConvexHull(bbx1)
    hull2 = ConvexHull(bbx2)
    
    hs1 = hull1.equations  # shape (n1, 4)
    hs2 = hull2.equations  # shape (n2, 4)
    
    hs = np.vstack((hs1, hs2))
    
    center1 = np.mean(bbx1, axis=0)
    center2 = np.mean(bbx2, axis=0)
    interior = (center1 + center2) / 2.0
    
    def is_inside(point, hs, tol=1e-6):
        return np.all(np.dot(hs[:,:-1], point) + hs[:,-1] <= tol)
    
    if not is_inside(interior, hs):
        if is_inside(center1, hs):
            interior = center1
        elif is_inside(center2, hs):
            interior = center2
        else:
            return 0.0

    try:
        hs_int = HalfspaceIntersection(hs, interior)
        if hs_int.intersections.shape[0] == 0:
            return 0.0
        
        hull_int = ConvexHull(hs_int.intersections)
        return hull_int.volume
    except Exception as e:
        return 0.0

def box3d_overlap_rate(corners1, corners2, vol1, vol2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (kent): add more description on corner points' orders.
    '''
    inter_vol = compute_intersection_volume(corners1, corners2)
    
    if min(vol1, vol2) < 1e-6:
        return 0.0
    return inter_vol / min(vol1, vol2)

def compute_obb_volume(obb):
    vec_x = obb[1] - obb[0]
    vec_y = obb[3] - obb[0]
    vec_z = obb[4] - obb[0]
    return abs(np.dot(vec_x, np.cross(vec_y, vec_z)))

def process_obb(bounding_box_list, curr_output_dir, new_obb):
    new_dict = {
        "path": curr_output_dir,
        "bbox": new_obb,
        "valid": True
    }
    
    to_remove_index = None
    new_vol = compute_obb_volume(new_obb)
    # print("------------------------------Overlap rate----------------------------------")
    for i, bbox_dict in enumerate(bounding_box_list):
        if bbox_dict["valid"] is False:
            continue

        bbox = bbox_dict["bbox"]

        exist_vol = compute_obb_volume(bbox)
        overlap_rate = box3d_overlap_rate(bbox, new_obb, exist_vol, new_vol)
        # print(overlap_rate)
        if overlap_rate < 0.5:
            continue
        
        if new_vol > exist_vol:
            to_remove_index = i
        else:
            new_dict["valid"] = False

        break

    # remove smaller bounding boxes
    # for obb in to_remove:
    #     if obb in obb_list:
    #         obb_list.remove(obb)

    if to_remove_index is not None:
        print("------------------------------Remove one bounding box----------------------------------")
        print(overlap_rate)
        bounding_box_list[to_remove_index]["valid"] = False
    
    bounding_box_list.append(new_dict)

    return bounding_box_list

def coverage_ratio(visible_mask, other_mask):
    """
    calculate the coverage ratio of two masks.
    """
    if visible_mask.shape != other_mask.shape:
        raise ValueError(f"the shapes of two masks are not equal: {visible_mask.shape} vs {other_mask.shape}")

    # # based on visible_mask
    # total_vis = np.count_nonzero(visible_mask)
    # if total_vis == 0:
    #     return 0.0

    # inter = np.logical_and(visible_mask, other_mask)
    # inter_count = np.count_nonzero(inter)

    # return inter_count / total_vis

    # Calculate intersection and union
    inter = np.logical_and(visible_mask, other_mask)
    union = np.logical_or(visible_mask, other_mask)

    inter_count = np.count_nonzero(inter)
    union_count = np.count_nonzero(union)

    # If the union is empty, define IoU as 0
    if union_count == 0:
        return 0.0

    # Modified: return intersection / union, instead of previous coverage fraction
    return inter_count / union_count

# def fill_open_holes(mask, kernel_size=26):
#     """
#     使用形态学闭运算来填充二值掩码中的孔洞和缺口。

#     参数:
#     mask (np.ndarray): 输入的二值掩码 (期望值为 0 和 255)。
#     kernel_size (int): 用于形态学操作的核的大小。
#                        这个值需要足够大以闭合目标缺口。

#     返回:
#     np.ndarray: 经过填补操作后的掩码。
#     """
#     # 确保掩码是 uint8 类型
#     if mask.dtype != np.uint8:
#         mask = (mask > 0).astype(np.uint8)

#     # 如果掩码值是 0 和 1, 将其转换为 0 和 255，这是OpenCV的标准格式
#     if mask.max() == 1:
#         mask = mask * 255

#     # 定义一个矩形核（Structuring Element）
#     # 核的尺寸决定了能闭合多大的孔洞
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

#     # 应用形态学闭运算
#     # cv2.MORPH_CLOSE = Dilation followed by Erosion
#     closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#     return (closed_mask > 0).astype(np.uint8)

def fill_open_holes(mask):
    """
    Find the most compact (irregular) quadrilateral bounding box through contour approximation algorithms.

    Parameters:
    mask (np.ndarray): Input binary mask (expected values are 0 and 255).

    Returns:
    np.ndarray: New mask containing the final quadrilateral, or None if not found.
    """
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)

    # Iteratively try different precision (epsilon) to find a 4-vertex polygon
    # The value of epsilon is a percentage of the contour perimeter
    step = 0.001
    max_epsilon = 0.1
    best_approx = None

    for epsilon_perc in np.arange(step, max_epsilon, step):
        perimeter = cv2.arcLength(main_contour, True)
        epsilon = epsilon_perc * perimeter
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # If a quadrilateral is found, save it and stop
        if len(approx) == 4:
            best_approx = approx
            break
            
    if best_approx is None:
        print("Warning: Could not find a 4-point approximation. Using convex hull instead.")
        # If no quadrilateral is found, return the convex hull as an alternative
        hull = cv2.convexHull(main_contour)
        # Try to approximate on the convex hull again
        perimeter = cv2.arcLength(hull, True)
        epsilon = 0.02 * perimeter
        best_approx = cv2.approxPolyDP(hull, epsilon, True)


    # Draw this quadrilateral on a new black background
    quad_mask = np.zeros_like(mask)
    cv2.drawContours(quad_mask, [best_approx], 0, (255), thickness=cv2.FILLED)
    
    return (quad_mask > 0).astype(np.uint8)

def crop_and_scale_o3d(o3d_img, crop_box):
    """
    Crop an open3d.geometry.Image and scale it back to original size.

    :param o3d_img: open3d.geometry.Image object
    :param crop_box: (left, top, right, bottom) pixel coordinates
    :return: Processed open3d.geometry.Image object
    """
    np_img = np.asarray(o3d_img)

    # crop
    left, top, right, bottom = crop_box
    cropped = np_img[top:bottom, left:right, ...]  # [y1:y2, x1:x2]

    # scale back to original size
    orig_h, orig_w = np_img.shape[:2]
    # cv2.resize
    scaled = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

    return o3d.geometry.Image(scaled)

def linear_masked_crops_and_scale(img_path, mask, k):
    """
    Read an image from file, generate k crops of different sizes for a given binary mask (maintaining original w/h ratio, ensuring complete mask inclusion),
    gradually magnifying from the smallest crop containing the mask to the full image, and scaling each crop result back to original resolution.

    :param img_path: str, input image file path
    :param mask: numpy.ndarray, dtype=np.uint8, binary mask (0 or 1) of same size as image, indicating region of interest
    :param k: int, number of crops to generate
    :return: list of dict, each dict contains:
             - 'image': open3d.geometry.Image, cropped and scaled back image
             - 'crop_box': (left, top, right, bottom), crop box coordinates in original image
    """
    # 1) Read image
    o3d_img = o3d.io.read_image(img_path)
    np_img = np.asarray(o3d_img)
    H, W = np_img.shape[:2]
    aspect = W / H

    # 2) Calculate mask minimum bounding box
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    mh, mw = (y1 - y0 + 1), (x1 - x0 + 1)

    # 3) Expand to same aspect ratio as original image
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if mw / mh < aspect:
        target_h = mh
        target_w = mh * aspect
    else:
        target_w = mw
        target_h = mw / aspect
    min_w, min_h = min(target_w, W), min(target_h, H)

    def make_box(cx, cy, w, h):
        x1 = max(0, cx - w/2); x2 = min(W, cx + w/2)
        y1 = max(0, cy - h/2); y2 = min(H, cy + h/2)
        # Adjust if dimensions are insufficient after boundary truncation
        w2, h2 = x2 - x1, y2 - y1
        if w2 < w:
            x1 = max(0, min(x1, W - w)); x2 = x1 + w
        if h2 < h:
            y1 = max(0, min(y1, H - h)); y2 = y1 + h
        return int(x1), int(y1), int(x2), int(y2)

    # 4) Linear interpolation generates k crops and resamples
    results = []
    for i in range(1, k + 1):
        t = i / k
        w_i = min_w + t * (W - min_w)
        h_i = min_h + t * (H - min_h)
        bx = make_box(cx, cy, w_i, h_i)
        x1, y1, x2, y2 = bx

        crop = np_img[y1:y2, x1:x2]
        resized = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LANCZOS4)
        out_o3d = o3d.geometry.Image(resized)

        results.append({
            'image': out_o3d,
            'crop_box': bx
        })

    return results

def recursive_path_replace(obj, old_path, new_path):
    """
    Recursively walk through obj (which may be a dict, list, tuple, or scalar),
    and in any string value replace old_path with new_path.
    
    Args:
        obj:         A dict, list, tuple, or primitive.
        old_path:    The substring to look for.
        new_path:    The substring to replace it with.
    
    Returns:
        A new object of the same structure, with replacements applied.
    """
    # If it’s a string, do the replace:
    if isinstance(obj, str):
        return obj.replace(old_path, new_path)
    
    # If it’s a dict, recurse into its values:
    if isinstance(obj, dict):
        return {
            key: recursive_path_replace(val, old_path, new_path)
            for key, val in obj.items()
        }
    
    # If it’s a list, recurse into each element:
    if isinstance(obj, list):
        return [recursive_path_replace(elem, old_path, new_path) for elem in obj]
    
    # If it’s a tuple, rebuild it:
    if isinstance(obj, tuple):
        return tuple(recursive_path_replace(elem, old_path, new_path) for elem in obj)
    
    # Otherwise (int, float, None, etc.), leave as‑is:
    return obj

def is_flat_obb_from_axes(obb_corners: np.ndarray,
                          flatness_threshold: float = 0.2) -> bool:
    """
    examine if the given OBB corners represent a flat OBB based on the lengths of its axes.
    """
    if obb_corners.shape != (8, 3):
        raise ValueError("obb_corners must be an array of shape (8, 3)")

    # Get three principal axes
    a = obb_corners[1] - obb_corners[0]
    b = obb_corners[3] - obb_corners[0]
    c = obb_corners[4] - obb_corners[0]

    L_a = np.linalg.norm(a)
    L_b = np.linalg.norm(b)
    L_c = np.linalg.norm(c)

    if min(L_a, L_b, L_c) < 1e-6:
        # Degenerate case, edge too short
        return False

    # Normalized directions
    ua = a / L_a
    ub = b / L_b
    uc = c / L_c

    lengths = np.array([L_a, L_b, L_c])
    sorted_len = np.sort(lengths)  # Ascending order: shortest, medium, longest

    # Check if the shortest dimension is significantly smaller than second shortest: flat
    if sorted_len[0] < flatness_threshold * sorted_len[1]:
        return True
    return False

def save_image_bbx(rgb_image, mask, output_dir):

    rgb_image = cv2.imread(rgb_image)

    ys, xs = np.where(mask > 0)

    boxed_img = rgb_image.copy()
    if len(xs) > 0 and len(ys) > 0:
        # Most compact box
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        # ====== Add margin (in pixels) ======
        margin = 2
        H, W = mask.shape
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(W - 1, x_max + margin)
        y_max = min(H - 1, y_max + margin)

        # Draw box on original image
        cv2.rectangle(boxed_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
    else:
        print("mask is empty, no foreground.")

    cv2.imwrite(f"{output_dir}/2d_output_bbx.png", boxed_img)

def save_time_breakdown(input_dir, output_dir, module_name, start_time, end_time, duration):
    input_csv_path = os.path.join(input_dir, "time_breakdown.csv")
    output_csv_path = os.path.join(output_dir, "time_breakdown.csv")
    
    all_rows = []

    # try to read existing CSV
    if os.path.exists(input_csv_path):
        with open(input_csv_path, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            all_rows = list(reader)
    else:
        all_rows.append(['Stage', 'Start', 'End', 'Duration_Sec'])

    # append new row
    all_rows.append([
        module_name,
        time.strftime("%H:%M:%S", time.localtime(start_time)),
        time.strftime("%H:%M:%S", time.localtime(end_time)),
        f"{duration:.2f}"
    ])

    # write back to output CSV
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    print(f"[{module_name}] complete. Time breakdown saved to {output_csv_path}")

def load_scaled_intrinsics(json_path: str, scale: float = 3.75) -> list[float]:
    """
    Load intrinsics from JSON, divide by scale factor, and return Column-major format list.
    Matching format: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get the intrinsics of the first frame (assume key is frame_000000)
        first_frame_id = next(iter(data))
        intrinsic_matrix = np.array(data[first_frame_id]["intrinsic"])
        
        # 1. Scale intrinsics
        # Note: Usually only scale fx, fy, cx, cy, dividing matrix directly is equivalent here
        scaled_matrix = intrinsic_matrix / scale
        # Keep the last element as 1.0 (if matrix scaling turns 1.0 into 1/3.75)
        scaled_matrix[2, 2] = 1.0
        
        # 2. Convert to Column-major list (Order 'F')
        # Result order: [m00, m10, m20, m01, m11, m21, m02, m12, m22]
        # Which is: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
        intrinsic_list = scaled_matrix.flatten(order='F').tolist()
        
        print(f"-> Loaded scaled intrinsics for {first_frame_id}: fx={intrinsic_list[0]:.2f}, fy={intrinsic_list[4]:.2f}")
        return intrinsic_list
    except Exception as e:
        print(f"-> [Warning] Could not load intrinsics from {json_path}: {e}")
        return None
    