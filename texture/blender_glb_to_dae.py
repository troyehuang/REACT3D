import bpy
import os
import math
import argparse
import sys
# from tqdm import tqdm

# key coordinate system correction workflow
def fix_coordinate_system():
    # switch to Object Mode and select all objects
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='SELECT')

    # compensation for converting URDF (Z-up) to Blender (Y-up)
    # rotate –90° around the X axis (convert Z-up to Y-up so the model displays with the correct orientation in Blender)
    bpy.ops.transform.rotate(
        value=-math.pi/2,          # -90 degrees in radians
        orient_axis='X',           # rotate around the X axis
        orient_type='GLOBAL',       # use global coordinate system
        constraint_axis=(True, False, False)
    )

    # apply the rotation transform (bake the rotation into the mesh data)
    bpy.ops.object.transform_apply(
        location=False,
        rotation=True,
        scale=False
    )

    # clear selection
    bpy.ops.object.select_all(action='DESELECT')

def convert_to_dae_and_texture_image(glb_path):

    # clear existing data
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # import the GLB file
    bpy.ops.import_scene.gltf(filepath=glb_path)

    # perform the coordinate system correction
    fix_coordinate_system()

    # export texture and DAE
    file_name = glb_path.split("/")[-1].split(".")[0]
    dae_path = glb_path.replace(".glb", ".dae")
    for i, img in enumerate(bpy.data.images):
        if img.name.startswith('Image'):  # only process images that are likely textures
            # generate a new filename (e.g., MyModel_Diffuse_0.png)
            img.name = f"{file_name}.png"  # rename the texture inside Blender
    bpy.ops.wm.collada_export(
        filepath=dae_path,
        check_existing=True,
    )

def main(root_dir):
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            if subfolder == "remain_scene.glb":
                convert_to_dae_and_texture_image(subfolder_path)
            else:
                continue
        # if subfolder == "unopenable":
        #     for entry in os.listdir(subfolder_path):
        #         inner_path = os.path.join(subfolder_path, entry)
        #         if os.path.isfile(inner_path) and entry.lower().endswith('.glb'):
        #             try:
        #                 convert_to_dae_and_texture_image(inner_path)
        #             except Exception as e:
        #                 print(f"Error when processing {inner_path}: {e}")

        # for inner in os.listdir(subfolder_path):
        #     inner_path = os.path.join(subfolder_path, inner)
        #     if not os.path.isdir(inner_path):
        #         continue
        #     if inner.endswith("not_valid"):
        #         continue

        glb_path = os.path.join(subfolder_path, "part.glb")
        if os.path.exists(glb_path):
            try:
                convert_to_dae_and_texture_image(glb_path)
            except Exception as e:
                print(f"Error when processing {glb_path}: {e}")

if __name__ == '__main__':
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Traverse the specified root directory to process all matching glb files, convert them to DAE format, and export textures."
    )
    parser.add_argument("--folder", help="opdm scene result folder")
    
    args = parser.parse_args(argv)
    
    root_folder = args.folder
    if not os.path.exists(root_folder):
        print(f"{root_folder} doesn't exist")
        exit(1)
        
    main(root_folder)