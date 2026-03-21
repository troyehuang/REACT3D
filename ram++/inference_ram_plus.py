'''
 * The Recognize Anything Plus Model (RAM++)
 * Written by Xinyu Huang
'''
import argparse
import numpy as np
import random
import os
import json
from tqdm import tqdm

import torch

from PIL import Image
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform


parser = argparse.ArgumentParser(
    description='Tag2Text inferece for tagging and captioning')
parser.add_argument('--image_dir',
                    metavar='DIR',
                    help='path to dataset directory',
                    default='images/demo/')
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_plus_swin_large_14m.pth')
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')


if __name__ == "__main__":

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = get_transform(image_size=args.image_size)

    #######load model
    model = ram_plus(pretrained=args.pretrained,
                             image_size=args.image_size,
                             vit='swin_l')
    model.eval()

    model = model.to(device)

    dir_path = args.image_dir

    tags = set()
    for image_name in tqdm(sorted(os.listdir(dir_path))):
        if not image_name.endswith(('.jpg', '.png')):
            continue
        # print(f'Processing image: {image_name}')
        image_path = os.path.join(dir_path, image_name)
        image = transform(Image.open(image_path)).unsqueeze(0).to(device)

        res = inference(image, model)[0]
        # print("Image Tags: ", res)

        tag_list = [tag.strip() for tag in res.split('|')]
        tags.update(tag_list)
    
    print("Unique Tags: ", tags)

    # Save the unique tags to a json file
    output_dir = os.path.dirname(os.path.normpath(dir_path))
    output_file = os.path.join(output_dir, 'scene_tags.json')
    with open(output_file, 'w') as f:
        json.dump(list(tags), f, indent=4)
    print(f"Unique tags saved to {output_file}")

    
