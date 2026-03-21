import argparse
import os
from glob import glob
from PIL import Image
from tqdm import tqdm

def resize_and_convert(src_dir, dst_dir, scale):
    os.makedirs(dst_dir, exist_ok=True)
    jpg_files = glob(os.path.join(src_dir, '*.jpg'))

    for img_path in tqdm(jpg_files):
        with Image.open(img_path) as img:
            w, h = img.size
            new_size = (int(w * scale), int(h * scale))
            #img_resized = img.resize(new_size, Image.BILINEAR)
            img_resized = img.reduce(2)
            basename = os.path.basename(img_path)
            name, _ = os.path.splitext(basename)
            dst_path = os.path.join(dst_dir, f'{name}.png')
            img_resized.save(dst_path, format='PNG')
            # print(f'Done: {img_path} -> {dst_path}, size={new_size}')

def convert_jpg_to_png(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    jpg_files = glob(os.path.join(src_dir, '*.jpg'))

    for img_path in tqdm(jpg_files):
        with Image.open(img_path) as img:
            basename = os.path.basename(img_path)
            name, _ = os.path.splitext(basename)
            dst_path = os.path.join(dst_dir, f'{name}.png')
            img.save(dst_path, format='PNG')

def main():
    parser = argparse.ArgumentParser(
        description="Resize JPG images by a scale factor and convert to PNG."
    )
    # parser.add_argument(
    #     "src_dir",
    #     help="Source directory containing .jpg images."
    # )
    # parser.add_argument(
    #     "dst_dir",
    #     help="Destination directory to save resized .png images."
    # )
    parser.add_argument(
        "data_dir",
        help="Source directory containing .jpg images."
    )
    parser.add_argument(
        "--scale", "-s",
        type=float,
        default=0.5,
        help="Scale factor for resizing (default: 0.5)."
    )

    args = parser.parse_args()

    #scene_ids = ["95d525fbfd", "2970e95b65", "27dd4da69e", "9f79564dbf", "09c1414f1b", "6b40d1a939", "3db0a1c8f3", "5f99900f09"]

    #scene_ids = ["0a184cf634", "0d2ee665be", "1d003b07bd", "2a496183e1", "3f15a9266d", "4ba22fa7e4", "5ee7c22ba0", "5fb5d2dbf2", "6cc2231b9c"]
    
    #scene_ids = ["6f1848d1e3", "7bc286c1b6", "7e09430da7", "7f4d173c9c", "8a35ef3cfe", "8be0cd3817", "09c1414f1b", "39e6ee46df"]

    #scene_ids = ["104acbf7d2", "324d07a5b3", "4318f8bb3c", "6115eddb86", "8890d0a267", "9460c8889d", "5656608266"]
    #scene_ids = ["3c95c89d61_drawer_vis_original"]
    #scene_ids = ["6464461276", "a29cccc784", "ab11145646", "c0f5742640", "d7abfc4b17"]
    #scene_ids = ["d755b3d9d8", "ef69d58016", "f3d64c30f8", "faec2f0468"]
    scene_ids = ["0a76e06478", "1366d5ae89", "2e74812d00", "419cbe7c11", "4422722c49", "55b2bf8036", "61adeff7d5", "88627b561e", "8a20d62ac0", "8f82c394d6", "bcd2436daf", "e91722b5a3"]
    src_dirs = []
    for scene_id in scene_ids:
        src_dirs.append(os.path.join(args.data_dir, scene_id, 'rgb'))

    for src_dir in src_dirs:
        dst_dir = src_dir.replace('rgb', 'images_2')
        resize_and_convert(src_dir, dst_dir, args.scale)
        #convert_jpg_to_png(src_dir, dst_dir)

if __name__ == "__main__":
    main()

