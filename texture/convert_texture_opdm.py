import cv2
import numpy as np
from pathlib import Path
import trimesh
import xatlas
from PIL import Image, ImageFilter
from IPython.display import display
import argparse
import os
import sys
from tqdm import tqdm
import time, csv

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
        all_rows.append(['Module', 'Start_Time', 'End_Time', 'Duration_Seconds'])

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

# Barycentric interpolation
def barycentric_interpolate(v0, v1, v2, c0, c1, c2, p):
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = p - v0
    d00 = np.dot(v0v1, v0v1)
    d01 = np.dot(v0v1, v0v2)
    d11 = np.dot(v0v2, v0v2)
    d20 = np.dot(v0p, v0v1)
    d21 = np.dot(v0p, v0v2)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-8:
        return (c0 + c1 + c2) / 3
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = np.clip(w, 0, 1)
    interpolate_color = u * c0 + v * c1 + w * c2
    return np.clip(interpolate_color, 0, 255)

# Point-in-Triangle test
def is_point_in_triangle(p, v0, v1, v2):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = sign(p, v0, v1)
    d2 = sign(p, v1, v2)
    d3 = sign(p, v2, v0)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)

def convert_glb_to_obj(glb_path: Path, out_dir: Path) -> Path:
    glb_path = Path(glb_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose output .obj name based on the GLB name
    obj_path = out_dir / f"{glb_path.stem}.obj"

    # Load (scene preferred; falls back to mesh if single)
    scene_or_mesh = trimesh.load(str(glb_path), force='scene')

    # Tell trimesh to write to disk; this will create .obj, .mtl, and textures
    # (Export infers type from the .obj extension.)
    scene_or_mesh.export(str(obj_path))

    if not obj_path.exists():
        raise RuntimeError("Export did not create the OBJ file as expected.")

    return obj_path

def scene_to_single_mesh(scene_or_mesh: trimesh.Scene) -> trimesh.Trimesh:
    """
    无论 GLB 读出来是 Scene 还是 Trimesh，都统一成一个 Trimesh。
    多几何就合并；保留原始拓扑与 UV。
    """
    if isinstance(scene_or_mesh, trimesh.Trimesh):
        return scene_or_mesh

    # Scene → 合并所有 geometry
    if not isinstance(scene_or_mesh, trimesh.Scene):
        raise TypeError(f"Unsupported type: {type(scene_or_mesh)}")

    if not scene_or_mesh.geometry:
        raise ValueError("Scene has no geometry.")

    geoms = list(scene_or_mesh.geometry.values())
    if len(geoms) == 1:
        return geoms[0]

    # 将多几何合并（会处理索引偏移；保留 per-vertex UV/颜色等）
    merged = trimesh.util.concatenate(geoms)
    return merged

def convert_glb_to_obj_with_normals(glb_path: Path, out_obj_path: Path) -> None:

    glb_path = Path(glb_path).expanduser().resolve()
    out_obj_path = Path(out_obj_path).expanduser().resolve()
    out_obj_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载 GLB；尽量读成 mesh，若是多几何则合并
    # 这里不加 process=True，避免读入时重算法线等副作用
    scene_or_mesh = trimesh.load(str(glb_path), force='scene', process=False)
    mesh = scene_to_single_mesh(scene_or_mesh)

    # 注意：这里不调用 mesh.fix_normals()，目的是**保留文件内自带的 normal**
    # 如果你想在“无 normal”时改为计算并导出，可以在外层判断里调用 mesh.fix_normals()

    # 强制导出带法线与纹理
    mesh.export(
        str(out_obj_path),
        file_type='obj',
        include_normals=True,
        include_texture=True
    )


def process_obj_file(input_mesh_path, texture_size=256):

    # input_mesh_path = "/home/troye/ssd/Zhao_SP/opdm_code/scene_output/40aec5fffa/microwave_0/0/part.obj"
    # output_file_name = input_mesh_path.split("/")[-1].split(".")[0]
    # output_texture_path = f"./texture/{output_file_name}.png"
    base, _ = os.path.splitext(input_mesh_path)
    # output_mesh_path = input_mesh_path.replace(".obj", ".glb")
    output_mesh_path = base + ".glb"
    output_obj_dir = input_mesh_path.replace(".obj", "_obj_textured")
    print(f"output_mesh_path: {output_mesh_path}")

    mesh = trimesh.load_mesh(input_mesh_path)
    # mesh.show()

    # UV Part
    vertex_colors = mesh.visual.vertex_colors
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

    vertices = mesh.vertices[vmapping]
    vertex_colors = vertex_colors[vmapping]

    mesh.vertices = vertices
    mesh.faces = indices

    # Texture Part
    # texture_size = 256

    upscale_factor = 1
    buffer_size = texture_size * upscale_factor

    texture_buffer = np.zeros((buffer_size, buffer_size, 4), dtype=np.uint8)

    # Texture-filling loop
    for face in mesh.faces:
        uv0, uv1, uv2 = uvs[face]
        c0, c1, c2 = vertex_colors[face]

        uv0 = (uv0 * (buffer_size - 1)).astype(int)
        uv1 = (uv1 * (buffer_size - 1)).astype(int)
        uv2 = (uv2 * (buffer_size - 1)).astype(int)

        min_x = max(int(np.floor(min(uv0[0], uv1[0], uv2[0]))), 0)
        max_x = min(int(np.ceil(max(uv0[0], uv1[0], uv2[0]))), buffer_size - 1)
        min_y = max(int(np.floor(min(uv0[1], uv1[1], uv2[1]))), 0)
        max_y = min(int(np.ceil(max(uv0[1], uv1[1], uv2[1]))), buffer_size - 1)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x + 0.5, y + 0.5])
                if is_point_in_triangle(p, uv0, uv1, uv2):
                    color = barycentric_interpolate(uv0, uv1, uv2, c0, c1, c2, p)
                    texture_buffer[y, x] = np.clip(color, 0, 255).astype(
                        np.uint8
                    )

    image_texture = Image.fromarray(texture_buffer)
    # display(image_texture)

    # Inpainting
    image_bgra = texture_buffer.copy()
    mask = (image_bgra[:, :, 3] == 0).astype(np.uint8) * 255
    image_bgr = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2BGR)
    inpainted_bgr = cv2.inpaint(
        image_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )
    inpainted_bgra = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2BGRA)
    texture_buffer = inpainted_bgra[::-1]
    image_texture = Image.fromarray(texture_buffer)

    # Median filter
    image_texture = image_texture.filter(ImageFilter.MedianFilter(size=3))

    # Gaussian blur
    image_texture = image_texture.filter(ImageFilter.GaussianBlur(radius=1))

    # Downsample
    image_texture = image_texture.resize((texture_size, texture_size), Image.LANCZOS)

    # Display the final texture
    # display(image_texture)

    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        baseColorTexture=image_texture,
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )

    visuals = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    mesh.visual = visuals
    # mesh.show()

    # export glb file
    mesh.fix_normals()
    mesh.export(output_mesh_path)

    # convert_glb_to_obj(output_mesh_path, output_obj_dir)
    # convert_glb_to_obj_with_normals(output_mesh_path, output_obj_dir)

def main(root_dir, texture_size=256):
    for subfolder in tqdm(os.listdir(root_dir)):
        subfolder_path = os.path.join(root_dir, subfolder)
        # if not os.path.isdir(subfolder_path):
        #     continue
        # unopenable mesh
        if subfolder == "remain_scene.ply":
            # continue
            process_obj_file(subfolder_path, 512)

        # openable mesh
        # for inner in os.listdir(subfolder_path):
        #     inner_path = os.path.join(subfolder_path, inner)
        #     if not os.path.isdir(inner_path):
        #         print(f"Skipping {inner_path}, not a directory")
        #         continue
        #     # if inner.endswith("not_valid"):
        #     #     continue

        obj_path = os.path.join(subfolder_path, "part.obj")
        if os.path.exists(obj_path):
            try:
                process_obj_file(obj_path, texture_size)
            except Exception as e:
                print(f"Error when processing {obj_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Traverse the specified root directory to process all matching part.obj files, perform UV parameterization, texture generation, and export OBJ/MTL files."
    )
    parser.add_argument("--folder", help="opdm scene result folder")
    parser.add_argument("--texture-size", help="resolution of texture", type=int, default=256)
    
    args = parser.parse_args()
    
    root_folder = args.folder
    if not os.path.exists(root_folder):
        print(f"{root_folder} doesn't exist")
        exit(1)
    
    # start timer
    start_time = time.time()

    main(root_folder, args.texture_size)

    # end timer
    end_time = time.time()
    duration = end_time - start_time

    # --- TIME BREAKDOWN LOGGING ---
    save_time_breakdown(
        input_dir=root_folder,
        output_dir=root_folder,
        module_name="Module 6: Texture Generation",
        start_time=start_time,
        end_time=end_time,
        duration=duration
    )
