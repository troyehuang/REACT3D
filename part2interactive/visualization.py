import os
from copy import deepcopy

import imageio
import open3d as o3d
import numpy as np
from PIL import Image, ImageChops
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from compas.geometry import oriented_bounding_box_numpy
import itertools

from utilities import (save_img_with_unique_name, 
                       get_obb_max_plane_normal, 
                       rotate_pcd, corners_to_mesh, 
                       rotate_corners, rotate_point, 
                       transform_origin_and_vector_to_world, 
                       transform_object_pcd_to_world, 
                       transform_cuboid_corners_to_world,
                       is_flat_obb_from_axes)

POINT_COLOR = [1, 0, 0]  # red for demonstration
ARROW_COLOR = [0, 1, 0]  # green
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

ARROW_COLOR_ROTATION = [0, 1, 0]  # green
ARROW_COLOR_TRANSLATION = [0, 0, 1]  # blue


def generate_rotation_visualization(
    pcd: o3d.geometry.PointCloud,
    axis_arrow: o3d.geometry.TriangleMesh,
    mask: np.ndarray,
    axis_vector: np.ndarray,
    origin: np.ndarray,
    range_min: float,
    range_max: float,
    num_samples: int,
    output_dir: str,
) -> None:
    """
    Generate visualization files for a rotation motion of a part.

    :param pcd: point cloud object representing 2D image input (RGBD) as a point cloud
    :param axis_arrow: mesh object representing axis arrow of rotation to be rendered in visualization
    :param mask: mask np.array of dimensions (height, width) representing the part to be rotated in the image
    :param axis_vector: np.array of dimensions (3, ) representing the vector of the axis of rotation
    :param origin: np.array of dimensions (3, ) representing the origin point of the axis of rotation
    :param range_min: float representing the minimum range of motion in radians
    :param range_max: float representing the maximum range of motion in radians
    :param num_samples: number of sample states to visualize in between range_min and range_max of motion
    :param output_dir: string path to directory in which to save visualization output
    """
    angle_in_radians = np.linspace(range_min, range_max, num_samples)
    angles_in_degrees = angle_in_radians * 180 / np.pi

    for idx, angle_in_degrees in enumerate(angles_in_degrees):
        # Make a copy of your original point cloud and arrow for each rotation
        rotated_pcd = deepcopy(pcd)
        rotated_arrow = deepcopy(axis_arrow)

        angle_rad = np.radians(angle_in_degrees)
        rotated_pcd = rotate_part(rotated_pcd, mask, axis_vector, origin, angle_rad)

        # Create a Visualizer object for each rotation
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)

        # Add the rotated geometries
        vis.add_geometry(rotated_pcd)
        vis.add_geometry(rotated_arrow)

        # Apply the additional rotation around x-axis if desired
        angle_x = np.pi * 5.5 / 5  # 198 degrees
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(np.asarray([1, 0, 0]) * angle_x)
        rotated_pcd.rotate(rotation_matrix, center=rotated_pcd.get_center())
        rotated_arrow.rotate(rotation_matrix, center=rotated_pcd.get_center())

        # Capture and save the image
        output_filename = f"{output_dir}/{idx}.png"
        vis.capture_screen_image(output_filename, do_render=True)
        vis.destroy_window()


def generate_translation_visualization(
    pcd: o3d.geometry.PointCloud,
    axis_arrow: o3d.geometry.TriangleMesh,
    mask: np.ndarray,
    end: np.ndarray,
    range_min: float,
    range_max: float,
    num_samples: int,
    output_dir: str,
) -> None:
    """
    Generate visualization files for a translation motion of a part.

    :param pcd: point cloud object representing 2D image input (RGBD) as a point cloud
    :param axis_arrow: mesh object representing axis arrow of translation to be rendered in visualization
    :param mask: mask np.array of dimensions (height, width) representing the part to be translated in the image
    :param axis_vector: np.array of dimensions (3, ) representing the vector of the axis of translation
    :param origin: np.array of dimensions (3, ) representing the origin point of the axis of translation
    :param range_min: float representing the minimum range of motion
    :param range_max: float representing the maximum range of motion
    :param num_samples: number of sample states to visualize in between range_min and range_max of motion
    :param output_dir: string path to directory in which to save visualization output
    """
    translate_distances = np.linspace(range_min, range_max, num_samples)
    for idx, translate_distance in enumerate(translate_distances):
        translated_pcd = deepcopy(pcd)
        translated_arrow = deepcopy(axis_arrow)

        translated_pcd = translate_part(translated_pcd, mask, end, translate_distance.item())

        # Create a Visualizer object for each rotation
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)

        # Add the translated geometries
        vis.add_geometry(translated_pcd)
        vis.add_geometry(translated_arrow)

        # Apply the additional rotation around x-axis if desired
        # TODO: not sure why we need this rotation for the translation, and when it would be desired
        angle_x = np.pi * 5.5 / 5  # 198 degrees
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.asarray([1, 0, 0]) * angle_x)
        translated_pcd.rotate(R, center=translated_pcd.get_center())
        translated_arrow.rotate(R, center=translated_pcd.get_center())

        # Capture and save the image
        output_filename = f"{output_dir}/{idx}.png"
        vis.capture_screen_image(output_filename, do_render=True)
        vis.destroy_window()


def get_rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Find the rotation matrix that aligns vec1 to vec2

    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix

def pixel_to_camera_coords(pixel, intrinsic):
    """transform pixel coordinates to camera coordinates"""
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = (pixel[0] - cx) / fx
    y = (pixel[1] - cy) / fy
    return np.array([x, y, 1.0])  # Z=1

def camera_to_pixel_coords(camera_coords, intrinsic):
    """transform camera coordinates to pixel coordinates"""
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    x = fx * camera_coords[0] / camera_coords[2] + cx
    y = fy * camera_coords[1] / camera_coords[2] + cy
    return np.array([x, y])

def draw_line(start_point: np.ndarray, end_point: np.ndarray, depth_scale) -> o3d.geometry.TriangleMesh:
    """
    Generate 3D mesh representing axis from start_point to end_point.

    :param start_point: np.ndarray of dimensions (3, ) representing the start point of the axis
    :param end_point: np.ndarray of dimensions (3, ) representing the end point of the axis
    :return: mesh object representing axis from start to end
    """

    # Adjust depth scale
    # scale_ratio = 1000. / depth_scale
    # start_point = start_point * scale_ratio
    # end_point = end_point * scale_ratio

    # start_point = pixel_to_camera_coords(start_point, src_matrix)
    # end_point = pixel_to_camera_coords(end_point, src_matrix)

    # start_point = camera_to_pixel_coords(start_point, dst_matrix)
    # end_point = camera_to_pixel_coords(end_point, dst_matrix)

    # Compute direction vector and normalize it
    direction_vector = end_point - start_point
    normalized_vector = direction_vector / np.linalg.norm(direction_vector)

    # Compute the rotation matrix to align the Z-axis with the desired direction
    target_vector = np.array([0, 0, 1])
    rot_mat = get_rotation_matrix_from_vectors(target_vector, normalized_vector)

    # Create the cylinder (shaft of the arrow)
    cylinder_length = 0.9  # 90% of the total arrow length, you can adjust as needed
    cylinder_radius = 0.01  # Adjust the thickness of the arrow shaft
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=cylinder_length)

    # Move base of cylinder to origin, rotate, then translate to start_point
    cylinder.translate([0, 0, 0])
    cylinder.rotate(rot_mat, center=[0, 0, 0])
    cylinder.translate(start_point)

    # Create the cone (head of the arrow)
    cone_height = 0.1  # 10% of the total arrow length, adjust as needed
    cone_radius = 0.03  # Adjust the size of the arrowhead
    cone = o3d.geometry.TriangleMesh.create_cone(radius=cone_radius, height=cone_height)

    # Move base of cone to origin, rotate, then translate to end of cylinder
    cone.translate([-0, 0, 0])
    cone.rotate(rot_mat, center=[0, 0, 0])
    cone.translate(start_point + normalized_vector * 0.4)

    arrow = cylinder + cone
    return arrow


def create_arrow_from_vector(origin: np.ndarray,
                             vector: np.ndarray,
                             color: tuple = (1.0, 0.0, 0.0),
                             cylinder_radius: float = 0.01,
                             cone_radius: float = 0.02,
                             cylinder_height_ratio: float = 0.8,
                             cone_height_ratio: float = 0.2) -> o3d.geometry.TriangleMesh:
    """
    Create an arrow from origin and vector.

    Parameters
    ----------
    origin : np.ndarray, shape (3,)
        Arrow starting coordinates.
    vector : np.ndarray, shape (3,)
        Arrow vector (from origin to end).
    cylinder_radius : float
        Radius of the arrow shaft.
    cone_radius : float
        Radius of the arrow cone base.
    cylinder_height_ratio : float
        Ratio of shaft length to total length (0 < ratio < 1).
    cone_height_ratio : float
        Ratio of cone length to total length (0 < ratio < 1), sum of two should be 1.

    Returns
    -------
    arrow : o3d.geometry.TriangleMesh
        Constructed arrow mesh, can be directly added to scene.
    """
    # 1. Calculate length and direction
    length = np.linalg.norm(vector)
    if length == 0:
        raise ValueError("Vector length is zero, cannot create arrow.")
    direction = vector / length

    # 2. Proportionally divide shaft and cone lengths
    cylinder_height = length * cylinder_height_ratio
    cone_height = length * cone_height_ratio

    # 3. Create default arrow along +Z axis
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=cylinder_radius,
        cone_radius=cone_radius,
        cylinder_height=cylinder_height,
        cone_height=cone_height
    )
    arrow.compute_vertex_normals()

    # 4. Compute rotation matrix to align +Z axis to direction
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        # +Z or -Z case
        if np.dot(z_axis, direction) < 0:
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([1, 0, 0] * np.pi)
        else:
            R = np.eye(3)
    else:
        axis = axis / axis_len
        angle = np.arccos(np.dot(z_axis, direction))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    # 5. Rotate and translate to origin
    arrow.rotate(R, center=(0, 0, 0))
    arrow.translate(origin)

    arrow.paint_uniform_color(color)

    return arrow


def rotate_part(
    pcd: o3d.geometry.PointCloud, mask: np.ndarray, axis_vector: np.ndarray, origin: np.ndarray, angle_rad: float
) -> o3d.geometry.PointCloud:
    """
    Generate rotated point cloud of mask based on provided angle around axis.

    :param pcd: point cloud object representing points of image
    :param mask: mask np.array of dimensions (height, width) representing the part to be rotated in the image
    :param axis_vector: np.array of dimensions (3, ) representing the vector of the axis of rotation
    :param origin: np.array of dimensions (3, ) representing the origin point of the axis of rotation
    :param angle_rad: angle in radians to rotate mask part
    :return: point cloud object after rotation of masked part
    """
    # Get the coordinates of the point cloud as a numpy array
    points_np = np.asarray(pcd.points)

    # Convert point cloud colors to numpy array for easier manipulation
    colors_np = np.asarray(pcd.colors)

    # Create skew-symmetric matrix from end
    K = np.array(
        [
            [0, -axis_vector[2], axis_vector[1]],
            [axis_vector[2], 0, -axis_vector[0]],
            [-axis_vector[1], axis_vector[0], 0],
        ]
    )

    # Compute rotation matrix using Rodrigues' formula
    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * np.dot(K, K)

    # Iterate over the mask and rotate the points corresponding to the object pixels
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] > 0:  # This condition checks if the pixel belongs to the object
                point_index = i * mask.shape[1] + j

                # Translate the point such that the rotation origin is at the world origin
                translated_point = points_np[point_index] - origin

                # Rotate the translated point
                rotated_point = np.dot(R, translated_point)

                # Translate the point back
                points_np[point_index] = rotated_point + origin

                colors_np[point_index] = POINT_COLOR

    # Update the point cloud's coordinates
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Update point cloud colors
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    return pcd


def translate_part(pcd, mask, axis_vector, distance):
    """
    Generate translated point cloud of mask based on provided angle around axis.

    :param pcd: point cloud object representing points of image
    :param mask: mask np.array of dimensions (height, width) representing the part to be translated in the image
    :param axis_vector: np.array of dimensions (3, ) representing the vector of the axis of translation
    :param distance: distance within coordinate system to translate mask part
    :return: point cloud object after translation of masked part
    """
    normalized_vector = axis_vector / np.linalg.norm(axis_vector)
    translation_vector = normalized_vector * distance

    # Convert point cloud colors to numpy array for easier manipulation
    colors_np = np.asarray(pcd.colors)

    # Get the coordinates of the point cloud as a numpy array
    points_np = np.asarray(pcd.points)

    # Iterate over the mask and assign the color to the points corresponding to the object pixels
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] > 0:  # This condition checks if the pixel belongs to the object
                point_index = i * mask.shape[1] + j
                colors_np[point_index] = POINT_COLOR
                points_np[point_index] += translation_vector

    # Update point cloud colors
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    # Update the point cloud's coordinates
    pcd.points = o3d.utility.Vector3dVector(points_np)

    return pcd


def batch_trim(images_path: str, save_path: str, identical: bool = False) -> None:
    """
    Trim white spaces from all images in the given path and save new images to folder.

    :param images_path: local path to folder containing all images. Images must have the extension ".png", ".jpg", or
    ".jpeg".
    :param save_path: local path to folder in which to save trimmed images
    :param identical: if True, will apply same crop to all images, else each image will have its whitespace trimmed
    independently. Note that in the latter case, each image may have a slightly different size.
    """

    def get_trim(im):
        """Trim whitespace from an image and return the cropped image."""
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        return bbox

    if identical:  #
        images = []
        optimal_box = None

        # load all images
        for image_file in sorted(os.listdir(images_path)):
            if image_file.endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(images_path, image_file)
                images.append(Image.open(image_path))

        # find optimal box size
        for im in images:
            bbox = get_trim(im)
            if bbox is None:
                bbox = (0, 0, im.size[0], im.size[1])  # bound entire image

            if optimal_box is None:
                optimal_box = bbox
            else:
                optimal_box = (
                    min(optimal_box[0], bbox[0]),
                    min(optimal_box[1], bbox[1]),
                    max(optimal_box[2], bbox[2]),
                    max(optimal_box[3], bbox[3]),
                )

        # apply cropping, if optimal box was found
        for idx, im in enumerate(images):
            im.crop(optimal_box)
            im.save(os.path.join(save_path, f"{idx}.png"))
            im.close()

    else:  # trim each image separately
        for image_file in os.listdir(images_path):
            if image_file.endswith(IMAGE_EXTENSIONS):
                image_path = os.path.join(images_path, image_file)
                with Image.open(image_path) as im:
                    bbox = get_trim(im)
                    trimmed = im.crop(bbox) if bbox else im
                    trimmed.save(os.path.join(save_path, image_file))


def create_gif(image_folder_path: str, num_samples: int, gif_filename: str = "output.gif") -> None:
    """
    Create gif out of folder of images and save to file.

    :param image_folder_path: path to folder containing images (non-recursive). Assumes images are named as {i}.png for
    each of i from 0 to num_samples.
    :param num_samples: number of sampled images to compile into gif.
    :param gif_filename: filename for gif, defaults to "output.gif"
    """
    # Generate a list of image filenames (assuming the images are saved as 0.png, 1.png, etc.)
    image_files = [f"{image_folder_path}/{i}.png" for i in range(num_samples)]

    # Read the images using imageio
    images = [imageio.imread(image_file) for image_file in image_files]
    assert all(
        images[0].shape == im.shape for im in images
    ), f"Found some images with a different shape: {[im.shape for im in images]}"

    # Save images as a gif
    gif_output_path = f"{image_folder_path}/{gif_filename}"
    imageio.mimsave(gif_output_path, images, duration=0.1)

    return

def generate_articulation_visualization(
    pcd: o3d.geometry.PointCloud,
    axis_arrow: o3d.geometry.TriangleMesh,
    mask: np.ndarray,
    axis_vector: np.ndarray,
    origin: np.ndarray,
    output_dir: str,
    intrinsic_matrix,
    depth_np,
    origin_3d,
    extrinsic,
    pred_type,
    scene_pcd,
    local_mask_mesh,
    best_k,
    best_pcd,
) -> None:
    """
    Generate visualization files for a translation motion of a part.

    :param pcd: point cloud object representing 2D image input (RGBD) as a point cloud
    :param axis_arrow: mesh object representing axis arrow of translation to be rendered in visualization
    :param mask: mask np.array of dimensions (height, width) representing the part to be translated in the image
    :param axis_vector: np.array of dimensions (3, ) representing the vector of the axis of translation
    :param origin: np.array of dimensions (3, ) representing the origin point of the axis of translation
    :param range_min: float representing the minimum range of motion
    :param range_max: float representing the maximum range of motion
    :param num_samples: number of sample states to visualize in between range_min and range_max of motion
    :param output_dir: string path to directory in which to save visualization output
    """
    translated_local_mask_mesh = deepcopy(local_mask_mesh)
    translated_pcd = deepcopy(pcd)
    translated_arrow = deepcopy(axis_arrow)
    translated_best_pcd = deepcopy(best_pcd)

    original_color_array = np.asarray(translated_pcd.colors).copy()

    # translated_pcd = translate_part(translated_pcd, mask, end, translate_distance.item())
    # translated_pcd = mask_part(translated_pcd, mask, axis_vector, origin)
    translated_pcd, masked_indices = fit_2d_mask_onto_3d_pointcloud(mask, translated_pcd, intrinsic_matrix, depth_np, depth_scale=1.0)
    
    # uncomment this line to show the original color (without red mask)
    translated_pcd.colors = o3d.utility.Vector3dVector(original_color_array)

    translated_pcd, obb, mask_pcd, box_pcd, cuboid = complete_point_cloud(translated_pcd, masked_indices, scene_pcd, extrinsic, pred_type)
    # translated_pcd, obb, mask_pcd, box_pcd, cuboid = complete_point_cloud_depth_img(translated_pcd, masked_indices, pred_type, depth_np, mask)

    original_mask_pcd = deepcopy(mask_pcd)

    if pred_type == "rotation":
        translated_pcd = rotate_mask_around_axis(translated_pcd, masked_indices, origin_3d, axis_vector, 45)
    elif pred_type == "translation":
        translated_pcd = translate_mask_through_axis(translated_pcd, masked_indices, origin_3d, axis_vector, 0.5)
    else:
        raise ValueError(f"Unknown pred_type: {pred_type}")

    # Create a Visualizer object for each rotation
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)

    # Add the translated geometries
    vis.add_geometry(translated_pcd)
    vis.add_geometry(translated_arrow)
    vis.add_geometry(local_mask_mesh)

    original_combined_pcd = deepcopy(translated_pcd)

    # Apply the additional rotation around x-axis if desired
    # TODO: not sure why we need this rotation for the translation, and when it would be desired
    center = translated_pcd.get_center()
    angle_x = np.pi * 5.5 / 5  # 198 degrees
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.asarray([1, 0, 0]) * angle_x)
    translated_pcd.rotate(R, center=center)
    translated_arrow.rotate(R, center=center)
    mask_pcd.rotate(R, center=center)
    local_mask_mesh.rotate(R, center=center)
    # original_mask_pcd.rotate(R, center=center)

    # visualize bbox and normal arrow
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # side face
    ]
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(obb)  
    lineset.lines = o3d.utility.Vector2iVector(lines)  
    lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
    lineset.rotate(R, center=translated_pcd.get_center())

    max_normal, max_center = get_obb_max_plane_normal(obb)
    arrow = create_arrow(max_center, max_normal, length=0.5, color=[0, 1, 0])

    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0, 0, 0])
    # opt.point_size = 1.

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    vis.add_geometry(lineset)
    # vis.add_geometry(arrow)
    # vis.add_geometry(mask_pcd)
    # vis.add_geometry(original_mask_pcd)

    # if origin_3d is not None:
    #     origin_3d = np.asarray(origin_3d)  # Convert to NumPy array
    #     if origin_3d.shape == (3,):  # If it is a 1D vector
    #         origin_3d = origin_3d.reshape(1, 3)  # Change to (1,3)
    #     elif origin_3d.shape == (4,):  # If homogeneous coordinates
    #         origin_3d = origin_3d[:3].reshape(1, 3)  # Remove the last dimension
    #     elif len(origin_3d.shape) == 1 or origin_3d.shape[1] != 3:  
    #         print("Error: origin_3d must have shape (N, 3)")
    #         return

    #     # Create red point cloud
    #     pp = o3d.geometry.PointCloud()
    #     pp.points = o3d.utility.Vector3dVector(origin_3d)
    #     pp.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (origin_3d.shape[0], 1)))  # Color matches number of points

    #     vis.add_geometry(pp)

    # Capture and save the image
    # output_filename = f"{output_dir}/3d_output.png"
    output_filename = save_img_with_unique_name(output_dir, "3d_output.png")

    mask_indices = np.array(masked_indices)
    mask_points = np.asarray(translated_pcd.points)[mask_indices]
    mask_colors = np.asarray(translated_pcd.colors)[mask_indices]

    mask_pcd = o3d.geometry.PointCloud()
    mask_pcd.points = o3d.utility.Vector3dVector(mask_points)
    mask_pcd.colors = o3d.utility.Vector3dVector(mask_colors)

    original_box_pcd = deepcopy(box_pcd)
    box_pcd  = rotate_pcd(box_pcd, center)
    # vis.add_geometry(box_pcd)
    
    # visualization
    # vis.capture_screen_image(output_filename, do_render=True)
    
    vis.run()
    vis.destroy_window()

    # rotate end and origin points, and then form vector
    end_3d = origin_3d + axis_vector
    translated_end_3d = rotate_point(end_3d, center)
    translated_origin_3d = rotate_point(origin_3d, center)


    cuboid_world = deepcopy(cuboid)
    cuboid_world["corners"] = transform_cuboid_corners_to_world(cuboid["corners"], extrinsic)

    cuboid["corners"] = rotate_corners(cuboid["corners"], center)
    
    return original_combined_pcd, original_box_pcd, original_mask_pcd, cuboid_world, translated_arrow, translated_end_3d - translated_origin_3d, translated_origin_3d, obb


def generate_mesh_articulation_visualization(
    pcd: o3d.geometry.PointCloud,
    axis_arrow: o3d.geometry.TriangleMesh,
    mask: np.ndarray,
    axis_vector: np.ndarray,
    origin: np.ndarray,
    output_dir: str,
    intrinsic_matrix,
    depth_np,
    origin_3d,
    extrinsic,
    pred_type,
    scene_pcd,
    local_mask_mesh,
    best_k,
    best_pcd,
    if_vis,
    range_max,
    local_mask_mesh_mesh
) -> None:
    """
    Generate visualization files for a translation motion of a part.

    :param pcd: point cloud object representing 2D image input (RGBD) as a point cloud
    :param axis_arrow: mesh object representing axis arrow of translation to be rendered in visualization
    :param mask: mask np.array of dimensions (height, width) representing the part to be translated in the image
    :param axis_vector: np.array of dimensions (3, ) representing the vector of the axis of translation
    :param origin: np.array of dimensions (3, ) representing the origin point of the axis of translation
    :param range_min: float representing the minimum range of motion
    :param range_max: float representing the maximum range of motion
    :param num_samples: number of sample states to visualize in between range_min and range_max of motion
    :param output_dir: string path to directory in which to save visualization output
    """
    translated_local_mask_mesh = deepcopy(local_mask_mesh)
    translated_pcd = deepcopy(pcd)
    translated_arrow = deepcopy(axis_arrow)
    translated_best_pcd = deepcopy(best_pcd)

    # change color of translated_arrow
    translated_arrow.paint_uniform_color([1, 0, 0])

    original_color_array = np.asarray(translated_pcd.colors).copy()

    # uncomment this line to show the original color (without red mask)
    translated_pcd.colors = o3d.utility.Vector3dVector(original_color_array)

    translated_pcd, obb, box_pcd, cuboid, obb_center, obb_normal = complete_point_cloud_true_mesh(translated_best_pcd, translated_local_mask_mesh, scene_pcd, extrinsic, pred_type, if_vis)
    if box_pcd is None:
        return None, None, None, None, None, None, None, None, None

    _, inner_box_mesh = corners_to_mesh(cuboid, pred_type)

    original_box_pcd = deepcopy(box_pcd)

    # refine origin and vector
    refined_origin_3d, refined_vector = refine_origin_and_vector(origin_3d, axis_vector, obb, pred_type)

    # ablation study: directly use the predicted origin and vector without refinement
    # refined_origin_3d, refined_vector = origin_3d, axis_vector

    original_refined_origin_3d = deepcopy(refined_origin_3d)
    original_refined_vector = deepcopy(refined_vector)
    obb_normal_arrow = create_arrow_from_vector(obb_center, obb_normal)

    # refined_axis_arrow = draw_line(refined_origin_3d, refined_vector + origin_3d, 1.)
    refined_axis_arrow = create_arrow_from_vector(refined_origin_3d, refined_vector, color=(0, 1, 0), cylinder_height_ratio=0.6)

    if pred_type == "rotation":
        # translated_pcd = rotate_mask_around_axis(translated_pcd, masked_indices, origin_3d, axis_vector, 45)
        range_max_degree = float(range_max * 180 / np.pi)
        print(f"range_max_degree: {range_max_degree}")
        translated_local_mask_mesh = rotate_mask_mesh_around_axis(translated_local_mask_mesh, refined_origin_3d, refined_vector, range_max_degree)
        local_mask_mesh_mesh = rotate_triangle_mesh_around_axis(local_mask_mesh_mesh, refined_origin_3d, refined_vector, 45)
        # local_mask_mesh_mesh = rotate_triangle_mesh_around_axis(local_mask_mesh_mesh, origin_3d, axis_vector, 45)
    elif pred_type == "translation":
        # translated_pcd = translate_mask_through_axis(translated_pcd, masked_indices, origin_3d, axis_vector, 0.5)
        range_max_distance = float(range_max)
        translated_local_mask_mesh = translate_mask_mesh_through_axis(translated_local_mask_mesh, refined_origin_3d, refined_vector, 0.5)
        local_mask_mesh_mesh = translate_triangle_mesh_through_axis(local_mask_mesh_mesh, refined_origin_3d, refined_vector, 0.5, inner_box_mesh=inner_box_mesh)
    else:
        raise ValueError(f"Unknown pred_type: {pred_type}")

    # Create a Visualizer object for each rotation
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    # Add the translated geometries
    vis.add_geometry(translated_pcd)
    # vis.add_geometry(translated_arrow)
    vis.add_geometry(refined_axis_arrow)
    # vis.add_geometry(translated_local_mask_mesh)
    vis.add_geometry(local_mask_mesh_mesh)
    vis.add_geometry(box_pcd)
    if inner_box_mesh:
        vis.add_geometry(inner_box_mesh)
    # vis.add_geometry(obb_normal_arrow)

    original_combined_pcd = deepcopy(translated_pcd)

    # Apply the additional rotation around x-axis if desired
    # TODO: not sure why we need this rotation for the translation, and when it would be desired
    center = translated_pcd.get_center()
    angle_x = np.pi * 5.5 / 5  # 198 degrees
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.asarray([1, 0, 0]) * angle_x)
    translated_pcd.rotate(R, center=center)
    translated_arrow.rotate(R, center=center)
    refined_axis_arrow.rotate(R, center=center)
    translated_local_mask_mesh.rotate(R, center=center)
    local_mask_mesh_mesh.rotate(R, center=center)
    box_pcd = rotate_pcd(box_pcd, center)
    obb_normal_arrow.rotate(R, center=center)
    if inner_box_mesh:
        inner_box_mesh.rotate(R, center=center)
    # original_mask_pcd.rotate(R, center=center)

    # visualize bbox and normal arrow
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # side face
    ]
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(obb)  
    lineset.lines = o3d.utility.Vector2iVector(lines)  
    lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
    lineset.rotate(R, center=translated_pcd.get_center())

    max_normal, max_center = get_obb_max_plane_normal(obb)
    arrow = create_arrow(max_center, max_normal, length=0.5, color=[0, 1, 0])

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    # vis.add_geometry(coordinate_frame)
    vis.add_geometry(lineset)
    # vis.add_geometry(arrow)
    # vis.add_geometry(mask_pcd)
    # vis.add_geometry(original_mask_pcd)

    # if origin_3d is not None:
    #     origin_3d = np.asarray(origin_3d)  # Convert to NumPy array
    #     if origin_3d.shape == (3,):  # If it is a 1D vector
    #         origin_3d = origin_3d.reshape(1, 3)  # Change to (1,3)
    #     elif origin_3d.shape == (4,):  # If homogeneous coordinates
    #         origin_3d = origin_3d[:3].reshape(1, 3)  # Remove the last dimension
    #     elif len(origin_3d.shape) == 1 or origin_3d.shape[1] != 3:  
    #         print("Error: origin_3d must have shape (N, 3)")
    #         return

    #     # Create red point cloud
    #     pp = o3d.geometry.PointCloud()
    #     pp.points = o3d.utility.Vector3dVector(origin_3d)
    #     pp.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (origin_3d.shape[0], 1)))  # Color matches number of points

    #     vis.add_geometry(pp)

    # Capture and save the image
    # output_filename = f"{output_dir}/3d_output.png"
    output_filename = save_img_with_unique_name(output_dir, "3d_output.png")
    
    # visualization
    # vis.capture_screen_image(output_filename, do_render=True)
    
    if if_vis:
        vis.run()
        vis.destroy_window()

    # rotate end and origin points, and then form vector
    end_3d = origin_3d + axis_vector
    translated_end_3d = rotate_point(end_3d, center)
    translated_origin_3d = rotate_point(origin_3d, center)


    cuboid_world = deepcopy(cuboid)
    cuboid_world["corners"] = transform_cuboid_corners_to_world(cuboid["corners"], extrinsic)

    cuboid["corners"] = rotate_corners(cuboid["corners"], center)
    
    return original_combined_pcd, original_box_pcd, cuboid_world, translated_arrow, translated_end_3d - translated_origin_3d, translated_origin_3d, obb, original_refined_origin_3d, original_refined_vector

def mask_part(
    pcd: o3d.geometry.PointCloud, mask: np.ndarray, axis_vector: np.ndarray, origin: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    Generate rotated point cloud of mask based on provided angle around axis.

    :param pcd: point cloud object representing points of image
    :param mask: mask np.array of dimensions (height, width) representing the part to be rotated in the image
    :param axis_vector: np.array of dimensions (3, ) representing the vector of the axis of rotation
    :param origin: np.array of dimensions (3, ) representing the origin point of the axis of rotation
    :param angle_rad: angle in radians to rotate mask part
    :return: point cloud object after rotation of masked part
    """
    # Get the coordinates of the point cloud as a numpy array
    points_np = np.asarray(pcd.points)

    # Convert point cloud colors to numpy array for easier manipulation
    colors_np = np.asarray(pcd.colors)

    # Iterate over the mask and rotate the points corresponding to the object pixels
    # max_index = (mask.shape[0] - 1) * mask.shape[1] + mask.shape[1] - 1
    # print(max_index)
    # print(len(points_np))
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] > 0:  # This condition checks if the pixel belongs to the object
                point_index = i * mask.shape[1] + j

                # Translate the point such that the rotation origin is at the world origin
                # translated_point = points_np[point_index] - origin

                # Rotate the translated point
                # rotated_point = np.dot(R, translated_point)

                # Translate the point back
                # points_np[point_index] = rotated_point + origin

                colors_np[point_index] = POINT_COLOR

    # Update the point cloud's coordinates
    pcd.points = o3d.utility.Vector3dVector(points_np)

    # Update point cloud colors
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    return pcd

def project_3d_to_2d(point, K):
    
    point_h = np.append(point, 1)  # transform to homogeneous coordinates
    pixel_coords = K @ point_h[:3]  # project to pixel coordinates
    pixel_coords /= pixel_coords[2] 
    return int(pixel_coords[0]), int(pixel_coords[1])

def visualize_articulation_in_scene(scene_pcd, origin_world: np.ndarray, axis_vector_world: np.ndarray, pred_type):
    axis_arrow_world = draw_line(origin_world, origin_world + axis_vector_world, depth_scale=1.0)

    if pred_type == "rotation":
        axis_arrow_world.paint_uniform_color(ARROW_COLOR_ROTATION)
    elif pred_type == "translation":
        axis_arrow_world.paint_uniform_color(ARROW_COLOR_TRANSLATION)

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(scene_pcd)
    vis.add_geometry(axis_arrow_world)
    vis.add_geometry(coordinate_frame)
    vis.run()

    return axis_arrow_world

def fit_2d_mask_onto_3d_pointcloud(mask, pcd, intrinsic_matrix, depth_np, depth_scale=1.):
    h, w = depth_np.shape
    mask_indices = np.argwhere(mask > 0)  # get all indices of mask
    
    points_3d = []
    masked_indices = []
    
    for v, u in mask_indices:
        if not (0 <= u < w and 0 <= v < h):
            continue
        
        depth = depth_np[v, u] / depth_scale
        if depth == 0:
            continue
        
        # calculate 3D point
        pixel_coords = np.array([u, v, 1]) * depth
        point_3d = np.dot(np.linalg.inv(intrinsic_matrix), pixel_coords)
        points_3d.append(point_3d)
    
    points_3d = np.array(points_3d)
    if points_3d.shape[0] == 0:
        return pcd, None
    
    # get kd-tree
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    color_array = np.asarray(pcd.colors)
    
    for p in points_3d:
        _, idx, _ = kdtree.search_knn_vector_3d(p, 1)  # search for the nearest point
        color_array[idx[0]] = [1, 0, 0]  # red mask
        masked_indices.append(idx[0])
    
    # Create a point cloud from the masked points
    masked_pcd = pcd.select_by_index(masked_indices)
    
    # Perform DBSCAN clustering to separate the main cluster from outliers
    eps = 0.1  # DBSCAN parameter: neighborhood radius
    min_points = 100  # DBSCAN parameter: minimum points to form a cluster
    labels = np.array(masked_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    # If no clusters found, return the original point cloud and no indices
    if labels.max() >= 0:
        # return pcd, None
    
        # Find the largest cluster (main body)
        largest_cluster_index = np.argmax(np.bincount(labels[labels >= 0]))
        
        # Get the indices of points in the largest cluster
        largest_cluster_indices = np.where(labels == largest_cluster_index)[0]
        inlier_indices = [masked_indices[i] for i in largest_cluster_indices]
    else:
        print("No clusters found.")
        inlier_indices = masked_indices
    
    # Update the colors of the inlier points
    color_array = np.asarray(pcd.colors)
    for idx in inlier_indices:
        color_array[idx] = [1, 0, 0]  # red mask
    
    pcd.colors = o3d.utility.Vector3dVector(color_array)
    
    return pcd, inlier_indices

def rotate_mask_around_axis(pcd, masked_indices, origin, axis, angle):
    """
    **for revolute joint motion**
    rotate the points in pcd that are masked by masked_indices around the specified axis by a given angle.
    """
    if masked_indices is []:
        return pcd
    
    axis = axis / np.linalg.norm(axis)
    rotation_matrix = Rotation.from_rotvec(np.radians(angle) * axis).as_matrix()
    
    pcd_points = np.asarray(pcd.points)
    masked_points = pcd_points[masked_indices]
    
    transformed_points = (masked_points - origin) @ rotation_matrix.T + origin
    
    # update transformed points positions
    pcd_points[masked_indices] = transformed_points

    return pcd

def rotate_mask_mesh_around_axis(mask_mesh, origin, axis, angle):
    """
    **for revolute joint motion**
    rotate the points in pcd that are masked by masked_indices around the specified axis by a given angle.
    """
    
    axis = axis / np.linalg.norm(axis)
    rotation_matrix = Rotation.from_rotvec(np.radians(angle) * axis).as_matrix()
    
    masked_points = np.asarray(mask_mesh.points)
    
    transformed_points = (masked_points - origin) @ rotation_matrix.T + origin
    
    # update transformed points positions
    mask_mesh.points = o3d.utility.Vector3dVector(transformed_points)

    return mask_mesh

def rotate_triangle_mesh_around_axis(mesh: o3d.geometry.TriangleMesh,
                                     origin: np.ndarray,
                                     axis: np.ndarray,
                                     angle_deg: float,
                                     inplace: bool = True) -> o3d.geometry.TriangleMesh:
    """
    Rotate TriangleMesh around a line passing through `origin` with direction `axis` by angle_deg (in degrees).
    Uses 4x4 homogeneous matrix: T = T(origin) @ R @ T(-origin), calling mesh.transform(T).
    - Rotates vertices and existing normals simultaneously.
    - `inplace=False` returns a rotated copy without modifying original mesh.
    """
    # Normalize axis
    axis = axis / np.linalg.norm(axis)

    # Axis-angle -> 3x3 rotation matrix
    R = Rotation.from_rotvec(np.deg2rad(angle_deg) * axis).as_matrix()

    # Assemble 4x4 homogeneous transformation
    T = np.eye(4)
    T[:3, :3] = R

    # T_total = T(origin) @ R @ T(-origin)
    T_to_origin = np.eye(4)
    T_to_origin[:3, 3] = -origin
    T_back = np.eye(4)
    T_back[:3, 3] = origin
    T_total = T_back @ T @ T_to_origin

    # Inplace or copy
    target = mesh if inplace else copy.deepcopy(mesh)

    # Apply transformation (automatically rotates normals)
    target.transform(T_total)

    # If no normals, can optionally recompute here
    if not target.has_vertex_normals():
        try:
            target.compute_vertex_normals()
        except Exception:
            pass

    return target

def translate_mask_through_axis(pcd, masked_indices, origin, axis, distance):
    """
    **for prismatic joint motion**
    translate the points in pcd that are masked by masked_indices along the specified axis by a given distance.
    """
    if masked_indices == []:
        return pcd

    axis = axis / np.linalg.norm(axis)
    translation_vector = distance * axis

    pcd_points = np.asarray(pcd.points)
    pcd_points[masked_indices] = pcd_points[masked_indices] + translation_vector

    return pcd

def translate_mask_mesh_through_axis(mask_mesh, origin, axis, distance):
    """
    **for prismatic joint motion**
    translate the points in pcd that are masked by masked_indices along the specified axis by a given distance.
    """

    axis = axis / np.linalg.norm(axis)
    translation_vector = distance * axis

    masked_points = np.asarray(mask_mesh.points)
    masked_points += translation_vector

    mask_mesh.points = o3d.utility.Vector3dVector(masked_points)

    return mask_mesh

def translate_triangle_mesh_through_axis(mesh: o3d.geometry.TriangleMesh,
                                         origin: np.ndarray,
                                         axis: np.ndarray,
                                         distance: float,
                                         inplace: bool = True,
                                         inner_box_mesh = None) -> o3d.geometry.TriangleMesh:
    axis = axis / np.linalg.norm(axis)
    t = distance * axis

    T = np.eye(4)
    T[:3, 3] = t

    target = mesh if inplace else deepcopy(mesh)
    if inner_box_mesh is not None:
        inner_box_mesh.transform(T)
    target.transform(T)  
    return target

# def get_deepest_in_direction(mask_points, longest_edge_length, obb_normal, scene_pcd, extrinsic, pj_max):

#     # 1. Compute mask center point
#     mask_center = np.mean(mask_points, axis=0)
#     dist_center_along_normal = mask_center.dot(obb_normal)

#     # mask_center_world, obb_normal_world = transform_object_pcd_to_world(mask_center, obb_normal, np.linalg.inv(extrinsic))
#     scene_pcd_camera = transform_object_pcd_to_world(scene_pcd, extrinsic)
    
#     # 2. Get longest edge length and compute cylinder radius
#     cylinder_radius = longest_edge_length * 0.1

#     # 3. Convert scene_pcd to array and compute vectors pointing from mask_center to each point
#     scene_pts = np.asarray(scene_pcd.points)
#     vecs = scene_pts - mask_center

#     # 4. Compute projection distance of these vectors on obb_normal
#     dists_along = vecs.dot(obb_normal)
#     # Only consider points in positive direction (along obb_normal direction)
#     print("pj_min - dist_center_along_normal:", pj_max - dist_center_along_normal)
#     forward_mask = dists_along > pj_max - dist_center_along_normal

#     # 5. Compute radial distance to the axis
#     perp_vecs = vecs - np.outer(dists_along, obb_normal)
#     perp_dists = np.linalg.norm(perp_vecs, axis=1)

#     # 6. Filter indices of points within cylinder
#     cyl_inds = np.where((forward_mask) & (perp_dists <= cylinder_radius))[0]

#     # 7. If there are intersection points, take minimum distance; otherwise keep original proj_max
#     if cyl_inds.size > 0:
#         first_hit_relative = np.min(dists_along[cyl_inds])

#         dist_center_along_normal = mask_center.dot(obb_normal)
#         first_hit = dist_center_along_normal + first_hit_relative

#         # Filter points whose distance exactly equals first_hit
#         hit_inds = cyl_inds[np.isclose(dists_along[cyl_inds], first_hit_relative)]

#         # Prepare a copy of scene point cloud for coloring
#         scene_highlight = o3d.geometry.PointCloud(scene_pcd_camera)  # Deep copy
#         pts = np.asarray(scene_highlight.points)
#         # Default gray
#         colors = np.tile(np.array([0.5, 0.5, 0.5]), (pts.shape[0], 1))
#         # Highlight hit points in red
#         colors[hit_inds] = np.array([1, 0, 0])
#         scene_highlight.colors = o3d.utility.Vector3dVector(colors)

#         # Visualization: original scene + highlighted points + mask_center and normal vector
#         geoms = [scene_highlight]

#         # Red sphere for mask_center
#         sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cylinder_radius * 0.1)
#         sphere.translate(mask_center)
#         sphere.paint_uniform_color([1.0, 0.0, 0.0])
#         geoms.append(sphere)

#         # Green arrow for obb_normal
#         arrow_length = longest_edge_length * 2
#         start = mask_center
#         end = mask_center + obb_normal * arrow_length
#         arrow = o3d.geometry.LineSet(
#             points=o3d.utility.Vector3dVector([start, end]),
#             lines=o3d.utility.Vector2iVector([[0, 1]])
#         )
#         arrow.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]])
#         geoms.append(arrow)

#         o3d.visualization.draw_geometries(
#             geoms,
#             window_name="First Hit Highlighted",
#             width=800, height=600
#         )
#     else:
#         first_hit = None
    
#     return first_hit

def get_deepest_in_direction(mask_points,
                             longest_edge_length,
                             obb_normal,
                             scene_pcd,
                             extrinsic,
                             pj_dist,
                             if_vis,
                             depth_window=0.05,
                             plane_distance_thresh=100,
                             min_inlier_ratio=0.5):
    """
    Based on the original first-hit logic:
    - Extract a layer of point cloud within a depth_window near first_hit
    - Use RANSAC to fit a plane, if inlier ratio >= min_inlier_ratio, consider plane found
    - Visualization: If plane found, render these plane inliers in green; otherwise keep red highlight for first hit
    """

    # 1. Compute mask center and its projection on obb_normal
    mask_center = np.mean(mask_points, axis=0)
    dist_center = mask_center.dot(obb_normal)
    mask_points_number = mask_points.shape[0]

    # 2. Cylinder radius
    cylinder_radius = longest_edge_length * 0.1

    # 3. Transform scene pcd to camera frame and extract ndarray
    scene_cam = transform_object_pcd_to_world(deepcopy(scene_pcd), extrinsic)
    pts = np.asarray(scene_cam.points)
    vecs = pts - mask_center

    # 4. Projection distance and radial distance
    dists_along = vecs.dot(obb_normal)
    perp_dists = np.linalg.norm(vecs - np.outer(dists_along, obb_normal), axis=1)

    # 5. ROI: forward & within cylinder
    forward_mask = dists_along > (2. * pj_dist)
    forward_inds = np.where(forward_mask)[0]
    cyl_inds = np.where(forward_mask & (perp_dists <= cylinder_radius))[0]
    if cyl_inds.size == 0:
        if forward_inds.size != 0:
            deepest_rel = np.max(dists_along[forward_mask])
            deepest = dist_center + deepest_rel
            return deepest
        else:
            return None

    # 6. Find first hit distance (relative to mask_center)
    first_hit_rel = np.min(dists_along[cyl_inds])
    first_hit = dist_center + first_hit_rel

    # 7. Take a depth slice near first_hit_rel
    depth_mask = np.abs(dists_along - first_hit_rel) <= depth_window
    layer_inds = cyl_inds[depth_mask[cyl_inds]]
    
    # 8. Try RANSAC plane fitting
    plane_found = False
    plane_inliers = []
    if layer_inds.size >= 30:  # At least 30 points
        pcd_layer = o3d.geometry.PointCloud()
        pcd_layer.points = o3d.utility.Vector3dVector(pts[layer_inds])

        plane_model, inliers = pcd_layer.segment_plane(
            distance_threshold=plane_distance_thresh,
            ransac_n=3,
            num_iterations=500
        )
        inlier_ratio = len(inliers) / layer_inds.size

        if inlier_ratio >= min_inlier_ratio:
            plane_found = True
            # Map inlier indices back to original scene_cam point cloud
            plane_inliers = layer_inds[np.array(inliers)]

    # 9. Visualization
    scene_highlight = o3d.geometry.PointCloud(scene_cam)  # Deep copy
    colors = np.tile([0.5, 0.5, 0.5], (pts.shape[0], 1))  # Default gray

    if plane_found:
        # Render plane inliers as green
        colors[layer_inds] = [0.0, 0.0, 1.0]
        colors[plane_inliers] = [0.0, 1.0, 0.0]
    else:
        # Otherwise highlight first hit points as red
        hit_inds = cyl_inds[np.isclose(dists_along[cyl_inds], first_hit_rel)]
        colors[hit_inds] = [1.0, 0.0, 0.0]

    scene_highlight.colors = o3d.utility.Vector3dVector(colors)

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    mask_pcd = o3d.geometry.PointCloud()
    mask_pcd.points = o3d.utility.Vector3dVector(mask_points)
    mask_pcd.paint_uniform_color([1.0, 1.0, 0.0])  # Yellow

    geoms = [scene_highlight, world_frame, mask_pcd]
    # Blue sphere representing mask_center
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=cylinder_radius * 0.1)
    sphere.translate(mask_center)
    sphere.paint_uniform_color([0.0, 0.0, 1.0])
    geoms.append(sphere)
    # Green arrow representing obb_normal direction
    arrow_length = longest_edge_length * 2
    start = mask_center
    end = mask_center + obb_normal * arrow_length
    arrow = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([start, end]),
        lines=o3d.utility.Vector2iVector([[0, 1]])
    )
    arrow.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0]])
    geoms.append(arrow)

    if if_vis:
        o3d.visualization.draw_geometries(
            geoms,
            window_name="First Hit / Plane Highlighted",
            width=800, height=600
        )

    deepest_rel = np.max(dists_along[forward_inds])  # Farthest point (max projection)
    deepest = dist_center + deepest_rel
    first_hit = min(first_hit, deepest) if first_hit else deepest

    return first_hit

def complete_point_cloud(input_pcd, mask_indices, scene_pcd, extrinsic, pred_type, num_samples=20000):
    """
    complete the point cloud
    """

    mask_indices = np.array(mask_indices)
    mask_points = np.asarray(input_pcd.points)[mask_indices]
    mask_colors = np.asarray(input_pcd.colors)[mask_indices]
    # set the color of the mask points to purple
    # mask_colors = np.array([1, 0, 1])  # purple color

    # compute the OBB using compas’s oriented_bounding_box_numpy
    # assume oriented_bounding_box_numpy returns 8 corner points
    obb_corners = np.array(oriented_bounding_box_numpy(mask_points))  # convert to numpy array

    # compute the OBB center
    obb_center = np.mean(obb_corners, axis=0)

    # compute the OBB normal and axes
    # compute three edges of the OBB
    a = obb_corners[1] - obb_corners[0]
    b = obb_corners[3] - obb_corners[0]
    c = obb_corners[4] - obb_corners[0]

    # compute the lengths of the edges
    L_a = np.linalg.norm(a)
    L_b = np.linalg.norm(b)
    L_c = np.linalg.norm(c)

    # choose the two longest edges to define the OBB axes
    edges = [(a, L_a), (b, L_b), (c, L_c)]
    edges_sorted = sorted(edges, key=lambda x: x[1], reverse=True)
    v1 = edges_sorted[0][0]  # longest edge
    v2 = edges_sorted[1][0]  # second longest edge

    # compute the face normal as z-axis 
    face_normal = np.cross(v1, v2)
    
    # ensure the face normal points to origin
    origin_dir = -obb_center
    dot1 = np.dot(face_normal, origin_dir)
    dot2 = np.dot(-face_normal, origin_dir)
    obb_normal = face_normal if dot1 < dot2 else -face_normal

    # noramlize the vectors
    obb_normal = obb_normal / np.linalg.norm(obb_normal)
    obb_x_direction = v1 / np.linalg.norm(v1)
    # compute the y direction as cross product of normal and x direction
    obb_y_direction = np.cross(obb_normal, obb_x_direction)
    obb_y_direction = obb_y_direction / np.linalg.norm(obb_y_direction)

    # compute projection of points onto the OBB normal
    projections = np.asarray(input_pcd.points).dot(obb_normal)
    projections_mask = mask_points.dot(obb_normal)

    proj_min, proj_max = np.min(projections), np.max(projections)
    pjmin, pjmax = np.min(projections_mask), np.max(projections_mask)

    first_hit = get_deepest_in_direction(mask_points, edges_sorted[0][1], obb_normal, scene_pcd, extrinsic, pjmax)
    if first_hit:
        proj_max = min(proj_max, first_hit)
        print(f"First hit found at {first_hit}, setting proj_max to {proj_max}")

    # get the average color of the mask points
    avg_color = np.mean(mask_colors, axis=0)

    # compute the min and max in the OBB local coordinates
    x_min, x_max = np.min(mask_points.dot(obb_x_direction)), np.max(mask_points.dot(obb_x_direction))
    y_min, y_max = np.min(mask_points.dot(obb_y_direction)), np.max(mask_points.dot(obb_y_direction))

    # create cuboid point cloud
    x_samples = np.random.uniform(x_min, x_max, num_samples)
    y_samples = np.random.uniform(y_min, y_max, num_samples)
    z_samples = np.random.uniform(pjmin, proj_max, num_samples)

    box_points = np.vstack([
        np.column_stack((x_samples, y_samples, np.full(num_samples, proj_max))),  # back
        np.column_stack((x_samples, np.full(num_samples, y_min), z_samples)),      # top
        np.column_stack((x_samples, np.full(num_samples, y_max), z_samples)),      # bottom
        np.column_stack((np.full(num_samples, x_min), y_samples, z_samples)),      # side
        np.column_stack((np.full(num_samples, x_max), y_samples, z_samples)),      # side
    ])

    # transform the box points from OBB local coordinates to world coordinates
    transformed_points = []
    for point in box_points:
        transformed_point = (point[0] * obb_x_direction +
                             point[1] * obb_y_direction +
                             point[2] * obb_normal)
        transformed_points.append(transformed_point)
    transformed_points = np.array(transformed_points)

    completed_pcd = o3d.geometry.PointCloud()
    completed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    completed_pcd.colors = o3d.utility.Vector3dVector(np.tile(avg_color, (transformed_points.shape[0], 1)))

    # combine point clouds
    combined_pcd = input_pcd + completed_pcd

    # build mask point cloud
    mask_pcd = o3d.geometry.PointCloud()
    mask_pcd.points = o3d.utility.Vector3dVector(mask_points)
    mask_pcd.colors = o3d.utility.Vector3dVector(mask_colors)

    # build cuboid corners
    corners = np.array([
        [x_min, y_min, pjmin],  # 1
        [x_min, y_min, proj_max],  # 2
        [x_min, y_max, pjmin],  # 3
        [x_min, y_max, proj_max],  # 4
        [x_max, y_min, pjmin],  # 5
        [x_max, y_min, proj_max],  # 6
        [x_max, y_max, pjmin],  # 7
        [x_max, y_max, proj_max],  # 8
    ])

    transformed_corners = []
    for point in corners:
        transformed_point = (point[0] * obb_x_direction +
                             point[1] * obb_y_direction +
                             point[2] * obb_normal)
        transformed_corners.append(transformed_point)

    cuboid = {"corners": transformed_corners, "color": avg_color}

    return combined_pcd, obb_corners, mask_pcd, completed_pcd, cuboid

def complete_point_cloud_true_mesh(translated_best_pcd, translated_local_mask_mesh, scene_pcd, extrinsic, pred_type, if_vis, num_samples=20000):
    """
    complete the point cloud
    """

    mask_points = np.asarray(translated_local_mask_mesh.points)
    mask_colors = np.asarray(translated_local_mask_mesh.colors)


    # set the color of the mask points to purple
    # mask_colors = np.array([1, 0, 1])  # purple color

    # compute the OBB using compas’s oriented_bounding_box_numpy
    # assume oriented_bounding_box_numpy returns 8 corner points
    obb_corners = np.array(oriented_bounding_box_numpy(mask_points))  # convert to numpy array

    if_flat = is_flat_obb_from_axes(obb_corners, 0.5)
    if not if_flat:
        print("The OBB is not flat, cannot complete the point cloud.")
        return None, None, None, None, None, None

    # compute the OBB center
    obb_center = np.mean(obb_corners, axis=0)

    # compute the OBB normal and axes
    # compute three edges of the OBB
    a = obb_corners[1] - obb_corners[0]
    b = obb_corners[3] - obb_corners[0]
    c = obb_corners[4] - obb_corners[0]

    # compute the lengths of the edges
    L_a = np.linalg.norm(a)
    L_b = np.linalg.norm(b)
    L_c = np.linalg.norm(c)

    # choose the two longest edges to define the OBB axes
    edges = [(a, L_a), (b, L_b), (c, L_c)]
    edges_sorted = sorted(edges, key=lambda x: x[1], reverse=True)
    v1 = edges_sorted[0][0]  # longest edge
    v2 = edges_sorted[1][0]  # second longest edge

    # compute the face normal as z-axis 
    face_normal = np.cross(v1, v2)
    
    # ensure the face normal points to origin
    origin_dir = -obb_center
    dot1 = np.dot(face_normal, origin_dir)
    dot2 = np.dot(-face_normal, origin_dir)
    obb_normal = face_normal if dot1 < dot2 else -face_normal

    # noramlize the vectors
    obb_normal = obb_normal / np.linalg.norm(obb_normal)
    obb_x_direction = v1 / np.linalg.norm(v1)
    # compute the y direction as cross product of normal and x direction
    obb_y_direction = np.cross(obb_normal, obb_x_direction)
    obb_y_direction = obb_y_direction / np.linalg.norm(obb_y_direction)

    # compute projection of points onto the OBB normal
    projections = np.asarray(translated_best_pcd.points).dot(obb_normal)
    projections_mask = mask_points.dot(obb_normal)

    proj_min, proj_max = np.min(projections), np.max(projections)
    pjmin, pjmax = np.min(projections_mask), np.max(projections_mask)

    first_hit = get_deepest_in_direction(mask_points, edges_sorted[0][1], obb_normal, scene_pcd, extrinsic, pjmax - pjmin, if_vis)
    if first_hit:
        proj_max = min(proj_max, first_hit)
        print(f"First hit found at {first_hit}, setting proj_max to {proj_max}")

    # get the average color of the mask points
    avg_color = np.mean(mask_colors, axis=0)

    # compute the min and max in the OBB local coordinates
    x_min, x_max = np.min(mask_points.dot(obb_x_direction)), np.max(mask_points.dot(obb_x_direction))
    y_min, y_max = np.min(mask_points.dot(obb_y_direction)), np.max(mask_points.dot(obb_y_direction))

    # create cuboid point cloud
    x_samples = np.random.uniform(x_min, x_max, num_samples)
    y_samples = np.random.uniform(y_min, y_max, num_samples)
    z_samples = np.random.uniform(pjmin, proj_max, num_samples)

    box_points = np.vstack([
        np.column_stack((x_samples, y_samples, np.full(num_samples, proj_max))),  # back
        np.column_stack((x_samples, np.full(num_samples, y_min), z_samples)),      # top
        np.column_stack((x_samples, np.full(num_samples, y_max), z_samples)),      # bottom
        np.column_stack((np.full(num_samples, x_min), y_samples, z_samples)),      # side
        np.column_stack((np.full(num_samples, x_max), y_samples, z_samples)),      # side
    ])

    # transform the box points from OBB local coordinates to world coordinates
    transformed_points = []
    for point in box_points:
        transformed_point = (point[0] * obb_x_direction +
                             point[1] * obb_y_direction +
                             point[2] * obb_normal)
        transformed_points.append(transformed_point)
    transformed_points = np.array(transformed_points)

    completed_pcd = o3d.geometry.PointCloud()
    completed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
    completed_pcd.colors = o3d.utility.Vector3dVector(np.tile(avg_color, (transformed_points.shape[0], 1)))

    # combine point clouds
    combined_pcd = translated_best_pcd + completed_pcd

    # build cuboid corners
    corners = np.array([
        [x_min, y_min, pjmin],  # 1
        [x_min, y_min, proj_max],  # 2
        [x_min, y_max, pjmin],  # 3
        [x_min, y_max, proj_max],  # 4
        [x_max, y_min, pjmin],  # 5
        [x_max, y_min, proj_max],  # 6
        [x_max, y_max, pjmin],  # 7
        [x_max, y_max, proj_max],  # 8
    ])

    transformed_corners = []
    for point in corners:
        transformed_point = (point[0] * obb_x_direction +
                             point[1] * obb_y_direction +
                             point[2] * obb_normal)
        transformed_corners.append(transformed_point)

    cuboid = {"corners": transformed_corners, "color": avg_color}

    return combined_pcd, obb_corners, completed_pcd, cuboid, obb_center, obb_normal

# def complete_point_cloud_depth_img(input_pcd, mask_indices, pred_type, depth, mask, num_samples=20000):
#     """
#     complete the point cloud
#     """

#     mask_indices = np.array(mask_indices)
#     mask_points = np.asarray(input_pcd.points)[mask_indices]
#     mask_colors = np.asarray(input_pcd.colors)[mask_indices]

#     # compute the OBB using compas’s oriented_bounding_box_numpy
#     # assume oriented_bounding_box_numpy returns 8 corner points
#     obb_corners = np.array(oriented_bounding_box_numpy(mask_points))  # convert to numpy array

#     # compute the OBB center
#     obb_center = np.mean(obb_corners, axis=0)

#     # compute the OBB normal and axes
#     # compute three edges of the OBB
#     a = obb_corners[1] - obb_corners[0]
#     b = obb_corners[3] - obb_corners[0]
#     c = obb_corners[4] - obb_corners[0]

#     # compute the lengths of the edges
#     L_a = np.linalg.norm(a)
#     L_b = np.linalg.norm(b)
#     L_c = np.linalg.norm(c)

#     # choose the two longest edges to define the OBB axes
#     edges = [(a, L_a), (b, L_b), (c, L_c)]
#     edges_sorted = sorted(edges, key=lambda x: x[1], reverse=True)
#     v1 = edges_sorted[0][0]  # longest edge
#     v2 = edges_sorted[1][0]  # second longest edge

#     # compute the face normal as z-axis 
#     face_normal = np.cross(v1, v2)
    
#     # ensure the face normal points to origin
#     origin_dir = -obb_center
#     dot1 = np.dot(face_normal, origin_dir)
#     dot2 = np.dot(-face_normal, origin_dir)
#     obb_normal = face_normal if dot1 < dot2 else -face_normal

#     # noramlize the vectors
#     obb_normal = obb_normal / np.linalg.norm(obb_normal)
#     obb_x_direction = v1 / np.linalg.norm(v1)
#     # compute the y direction as cross product of normal and x direction
#     obb_y_direction = np.cross(obb_normal, obb_x_direction)
#     obb_y_direction = obb_y_direction / np.linalg.norm(obb_y_direction)

#     # compute projection of points onto the OBB normal
#     projections = np.asarray(input_pcd.points).dot(obb_normal)
#     projections_mask = mask_points.dot(obb_normal)

#     proj_min, proj_max = np.min(projections), np.max(projections)
#     pjmin, pjmax = np.min(projections_mask), np.max(projections_mask)

#     # get the average color of the mask points
#     avg_color = np.mean(mask_colors, axis=0)

#     # compute the min and max in the OBB local coordinates
#     x_min, x_max = np.min(mask_points.dot(obb_x_direction)), np.max(mask_points.dot(obb_x_direction))
#     y_min, y_max = np.min(mask_points.dot(obb_y_direction)), np.max(mask_points.dot(obb_y_direction))

#     # create cuboid point cloud
#     x_samples = np.random.uniform(x_min, x_max, num_samples)
#     y_samples = np.random.uniform(y_min, y_max, num_samples)
#     z_samples = np.random.uniform(pjmin, proj_max, num_samples)

#     box_points = np.vstack([
#         np.column_stack((x_samples, y_samples, np.full(num_samples, proj_max))),  # back
#         np.column_stack((x_samples, np.full(num_samples, y_min), z_samples)),      # top
#         np.column_stack((x_samples, np.full(num_samples, y_max), z_samples)),      # bottom
#         np.column_stack((np.full(num_samples, x_min), y_samples, z_samples)),      # side
#         np.column_stack((np.full(num_samples, x_max), y_samples, z_samples)),      # side
#     ])

#     # transform the box points from OBB local coordinates to world coordinates
#     transformed_points = []
#     for point in box_points:
#         transformed_point = (point[0] * obb_x_direction +
#                              point[1] * obb_y_direction +
#                              point[2] * obb_normal)
#         transformed_points.append(transformed_point)
#     transformed_points = np.array(transformed_points)

#     completed_pcd = o3d.geometry.PointCloud()
#     completed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
#     completed_pcd.colors = o3d.utility.Vector3dVector(np.tile(avg_color, (transformed_points.shape[0], 1)))

#     # combine point clouds
#     combined_pcd = input_pcd + completed_pcd

#     # build mask point cloud
#     mask_pcd = o3d.geometry.PointCloud()
#     mask_pcd.points = o3d.utility.Vector3dVector(mask_points)
#     mask_pcd.colors = o3d.utility.Vector3dVector(mask_colors)

#     # build cuboid corners
#     corners = np.array([
#         [x_min, y_min, pjmin],  # 1
#         [x_min, y_min, proj_max],  # 2
#         [x_min, y_max, pjmin],  # 3
#         [x_min, y_max, proj_max],  # 4
#         [x_max, y_min, pjmin],  # 5
#         [x_max, y_min, proj_max],  # 6
#         [x_max, y_max, pjmin],  # 7
#         [x_max, y_max, proj_max],  # 8
#     ])

#     transformed_corners = []
#     for point in corners:
#         transformed_point = (point[0] * obb_x_direction +
#                              point[1] * obb_y_direction +
#                              point[2] * obb_normal)
#         transformed_corners.append(transformed_point)

#     cuboid = {"corners": transformed_corners, "color": avg_color}

#     return combined_pcd, obb_corners, mask_pcd, completed_pcd, cuboid

# def refine_origin_and_vector(origin, vector, pcd, pred_type):
#     if pred_type != "rotation":
#         return origin, vector

#     origin = np.asarray(origin, float)
#     vector = np.asarray(vector, float)
#     v_unit = vector / np.linalg.norm(vector)
#     pcd_points = np.asarray(pcd.points)

#     # Get OBB corners and compute center
#     obb_corners = np.array(oriented_bounding_box_numpy(pcd_points))  # (8,3)
#     center = obb_corners.mean(axis=0)

#     # PCA on corners to get principal axes
#     cov = np.cov((obb_corners - center).T)
#     eigvals, eigvecs = np.linalg.eigh(cov)
#     # sort by descending variance
#     order = np.argsort(eigvals)[::-1]
#     axes = [eigvecs[:, i] for i in order]

#     # Extents along each axis: max - min projection
#     extents = []
#     for i_vec in axes:
#         proj = (obb_corners - center) @ i_vec
#         extents.append(proj.max() - proj.min())
#     extents = np.array(extents)
#     half = extents * 0.5

#     # Face pairs (axis indices after sorting)
#     face_pairs = [(0, 1), (0, 2), (1, 2)]
#     # Compute face areas
#     areas = {pair: extents[pair[0]] * extents[pair[1]] for pair in face_pairs}
#     i, j = max(areas, key=areas.get)
#     # Normal axis index is the remaining one
#     norm_idx = list({0, 1, 2} - {i, j})[0]
#     N_A = axes[norm_idx]

#     # Distance helper
#     def dist_to_line(pt, line_orig, line_dir):
#         return np.linalg.norm(np.cross(pt - line_orig, line_dir))

#     # Pick nearer parallel plane for face A
#     plane_centers_A = [center + N_A * half[norm_idx], center - N_A * half[norm_idx]]
#     dA = [dist_to_line(c, origin, v_unit) for c in plane_centers_A]
#     center_A = plane_centers_A[int(np.argmin(dA))]

#     # On face A, choose edge direction aligning with vector
#     U_i, U_j = axes[i], axes[j]
#     pi, pj = abs(np.dot(v_unit, U_i)), abs(np.dot(v_unit, U_j))
#     if pi >= pj:
#         edge_axis = i
#         edge_dir = U_i
#     else:
#         edge_axis = j
#         edge_dir = U_j

#     result_vector = edge_dir * extents[edge_axis]
#     if np.dot(result_vector, vector) < 0:
#         result_vector = -result_vector

#     # Face B shares the chosen edge: normal is remaining axis
#     other_norm = list({0, 1, 2} - {edge_axis, norm_idx})[0]
#     N_B = axes[other_norm]
#     plane_centers_B = [center + N_B * half[other_norm], center - N_B * half[other_norm]]
#     dB = [dist_to_line(c, origin, v_unit) for c in plane_centers_B]
#     result_origin = plane_centers_B[int(np.argmin(dB))]

#     return result_origin, result_vector

def refine_origin_and_vector(origin, vector, obb_corners, pred_type=None):

    # Calculate three principal axes of OBB (starting from corner0)
    a = obb_corners[1] - obb_corners[0]
    b = obb_corners[3] - obb_corners[0]
    c = obb_corners[4] - obb_corners[0]
    
    # Calculate lengths of each axis and sort
    len_a = np.linalg.norm(a)
    len_b = np.linalg.norm(b)
    len_c = np.linalg.norm(c)
    edges = [(a, len_a, 0), (b, len_b, 1), (c, len_c, 2)]
    edges_sorted = sorted(edges, key=lambda x: x[1], reverse=True)
    
    # Select the two longest axes
    v1, len_v1, idx1 = edges_sorted[0]
    v2, len_v2, idx2 = edges_sorted[1]
    
    # Determine which axis the input vector is closer to
    vector_norm = np.linalg.norm(vector)

    if pred_type != "rotation":
        face_normal = np.cross(v1, v2)
        norm_face_normal = np.linalg.norm(face_normal)
        if norm_face_normal < 1e-8:  # Degenerate, try using third principal axis
            v3 = edges_sorted[2][0]
            face_normal = np.cross(v1, v3)
            norm_face_normal = np.linalg.norm(face_normal)
            if norm_face_normal < 1e-8:
                face_normal = v1
                norm_face_normal = np.linalg.norm(face_normal) if np.linalg.norm(face_normal) > 1e-8 else 1.0
        dir_vec = face_normal / norm_face_normal
        if vector_norm >= 1e-8:
            if np.dot(dir_vec, vector) < 0:
                dir_vec = -dir_vec
            result_vector = dir_vec * vector_norm
        else:
            result_vector = dir_vec

        return origin, result_vector
    
    if vector_norm < 1e-8:  # Process zero vector
        edge_selected = v1
        edge_unselected = v2
        selected_idx = idx1
        unselected_idx = idx2
    else:
        cos_v1 = abs(np.dot(vector, v1)) / (vector_norm * len_v1)
        cos_v2 = abs(np.dot(vector, v2)) / (vector_norm * len_v2)
        if cos_v1 >= cos_v2:
            edge_selected = v1
            edge_unselected = v2
            selected_idx = idx1
            unselected_idx = idx2
        else:
            edge_selected = v2
            edge_unselected = v1
            selected_idx = idx2
            unselected_idx = idx1
    
    # Get endpoints of unselected axis
    # Determine endpoint indices of unselected axis based on index
    unselected_end_indices = {
        0: [0, 1],  # a axis connects points 0 and 1
        1: [0, 3],  # b axis connects points 0 and 3
        2: [0, 4]   # c axis connects points 0 and 4
    }[unselected_idx]

    selected_end_indices = {
        0: [0, 1],  # a axis connects points 0 and 1
        1: [0, 3],  # b axis connects points 0 and 3
        2: [0, 4]   # c axis connects points 0 and 4
    }[selected_idx]
    
    # Find all edges parallel to the selected axis
    parallel_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            edge_vec = obb_corners[j] - obb_corners[i]
            # Check if parallel (consider opposite direction)
            if (not (i == selected_end_indices[0] and j == selected_end_indices[1])) and (not (i == selected_end_indices[1] and j == selected_end_indices[0])) and np.linalg.norm(np.cross(edge_vec, edge_selected)) < 1e-6:
                parallel_edges.append((obb_corners[i], obb_corners[j]))
    
    # Select parallel edge connected to unselected axis endpoint
    edge_selected_parallel = None
    for p1, p2 in parallel_edges:
        # Check if connected to either endpoint of unselected axis
        if (np.array_equal(p1, obb_corners[unselected_end_indices[0]]) or 
            np.array_equal(p1, obb_corners[unselected_end_indices[1]]) or
            np.array_equal(p2, obb_corners[unselected_end_indices[0]]) or 
            np.array_equal(p2, obb_corners[unselected_end_indices[1]])):
            edge_selected_parallel = (p1, p2)
            idx1 = np.where((obb_corners == p1).all(axis=1))[0][0]
            idx2 = np.where((obb_corners == p2).all(axis=1))[0][0]
            edge_selected_parallel_end_indices = [idx1, idx2]
            break
    
    # If not found, use default edge (this should be avoided)
    # if edge_selected_parallel is None:
    #     edge_selected_parallel = (obb_corners[0], obb_corners[0] + edge_selected)
    
    # print(f"edge_selected_parallel_end_indices: {edge_selected_parallel_end_indices}")
    # Compute distance from origin to two lines
    def point_to_line_distance(point, line_point1, line_point2):
        line_dir = line_point2 - line_point1
        vec = point - line_point1
        cross = np.cross(vec, line_dir)
        return np.linalg.norm(cross) / np.linalg.norm(line_dir)
    
    # Compute distance to selected axis (using principal axis)
    dist_selected = point_to_line_distance(origin, obb_corners[0], obb_corners[0] + edge_selected)
    
    # Compute distance to parallel edge
    dist_parallel = point_to_line_distance(origin, edge_selected_parallel[0], edge_selected_parallel[1])
    
    # Select the closer line
    if dist_selected <= dist_parallel:
        A = obb_corners[0]
        B = obb_corners[0] + edge_selected
        p1 = selected_end_indices[0]
        p2 = selected_end_indices[1]
    else:
        A, B = edge_selected_parallel
        p1, p2 = edge_selected_parallel_end_indices
    
    # Determine direction of final edge (aligned with input vector)
    AB = B - A

    AB_len = np.linalg.norm(AB)
    mid_AB = (A + B) / 2.0
    result_origin = mid_AB  # fallback

    if AB_len > 1e-8:
        AB_dir = AB / AB_len
        min_dist = float("inf")
        closest_mid = None
        for c1, c2 in parallel_edges:
            # Skip self
            if (np.allclose(c1, A) and np.allclose(c2, B)) or (np.allclose(c1, B) and np.allclose(c2, A)):
                continue
            edge_vec = c2 - c1
            if np.linalg.norm(np.cross(edge_vec, AB)) >= 1e-6:
                continue  # Skip if not parallel
            mid = (c1 + c2) / 2.0
            diff_mid = mid - mid_AB
            perp = diff_mid - np.dot(diff_mid, AB_dir) * AB_dir
            dist = np.linalg.norm(perp)
            if dist < min_dist:
                min_dist = dist
                closest_mid = mid
        if closest_mid is not None:
            result_origin = (mid_AB + closest_mid) / 2.0

    if np.dot(AB, vector) >= 0:
        result_vector = AB
    else:
        result_vector = -AB
    
    # Compute midpoint as new origin
    # result_origin = (A + B) / 2.0

    # print("cor:", obb_corners[p1] - obb_corners[p2])
    # print("result_vec:", AB)
    
    return result_origin, result_vector
    

def align_obb_to_ground(input_pcd):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(input_pcd.points))

    obb = pcd.get_oriented_bounding_box()

    obb_normal = obb.R[:, 2]

    ground_normal = np.array([0, 0, 1])

    cos_angle = np.dot(obb_normal, ground_normal)
    angle = np.arccos(cos_angle)

    rotation_axis = np.cross(obb_normal, ground_normal)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

    obb.rotate(R, center=obb.get_center())

    return obb

def visualize_pointcloud_with_obb(points, input_pcd):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    obb = pcd.get_oriented_bounding_box()
    obb.color = (1, 0, 0)

    o3d.visualization.draw_geometries([obb, pcd])

    return obb

def create_arrow(max_center, max_normal, length=1.0, color=[0, 1, 0]):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02, cone_radius=0.04, cylinder_height=0.8 * length, cone_height=0.2 * length
    )
    arrow.compute_vertex_normals()
    
    # rotate the arrow to align with the max_normal
    z_axis = np.array([0, 0, 1])
    rotation_axis = np.cross(z_axis, max_normal)
    rotation_angle = np.arccos(np.dot(z_axis, max_normal))
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
        arrow.rotate(rotation_matrix, center=[0, 0, 0])
    
    arrow.translate(max_center)
    
    arrow.paint_uniform_color(color)
    
    return arrow

def vis_mesh_base_part(base, part, axis_arrow, inner_box):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="mesh")

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    vis.add_geometry(base)
    vis.add_geometry(part)
    vis.add_geometry(axis_arrow)
    if inner_box:
        vis.add_geometry(inner_box)

    render_option = vis.get_render_option()

    render_option.mesh_show_back_face = True

    vis.run()
    vis.destroy_window()
