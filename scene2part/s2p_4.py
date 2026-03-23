import os
import numpy as np
import pickle
from PIL import Image
import cv2
from tqdm import tqdm
import torch
import subprocess
from scipy.ndimage import label, binary_fill_holes, binary_dilation
import argparse

def largest_connected_true_region(array):
    # Step 1: Label all connected regions of `True` values
    labeled_array, num_features = label(array)
    
    # Step 2: Calculate the size of each region and find the largest one
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0  # Ignore the background (label 0)
    largest_label = sizes.argmax()
    
    # Step 3: Create a mask for only the largest connected region
    largest_region_mask = (labeled_array == largest_label)
    
    # Step 4: Fill holes inside the largest region
    filled_region = binary_fill_holes(largest_region_mask)
    
    # Step 5: Modify the original array to only keep the largest filled region
    result_array = np.zeros_like(array, dtype=bool)
    result_array[filled_region] = True
    
    return result_array

def check_rectangle_overlap(rectangles):
    """
    Check if any rectangles in the list overlap with each other.
    Each rectangle is defined as (x_min, y_min, x_max, y_max).
    Returns True if there's an overlap, False otherwise.
    """
    for i in range(len(rectangles)):
        for j in range(i + 1, len(rectangles)):
            rect1 = rectangles[i]
            rect2 = rectangles[j]
            
            # Check if rectangles overlap
            if (rect1[0] < rect2[2] and rect1[2] > rect2[0] and 
                rect1[1] < rect2[3] and rect1[3] > rect2[1]):
                return True
    
    return False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    
    back_match_dir = os.path.join(args.data_dir, 'perception', "vis_groups_back_match")
    save_dir = os.path.join(args.data_dir, 'perception', "vis_groups_cleaned")
    save_mask_dir = os.path.join(args.data_dir, 'perception', "vis_groups_cleaned_masks")

    existing_stem_masks = []

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_mask_dir, exist_ok=True)

    subprocess.run(f"rm {save_dir}/*", shell=True)
    subprocess.run(f"rm {save_mask_dir}/*", shell=True)

    image_names = sorted(os.listdir(back_match_dir))
    stem_list = [name[:-len("_cropped_original.png")] for name in image_names if name.endswith("_cropped_original.png")]

    for stem in tqdm(stem_list):
        original_image_path = os.path.join(back_match_dir, f"{stem}_original.png")
        cropped_original_image_path = os.path.join(back_match_dir, f"{stem}_cropped_original.png")
        mask_image_path = os.path.join(back_match_dir, f"{stem}_mask.png")
        image_masked_image_path = os.path.join(back_match_dir, f"{stem}_image_masked.png")

        save_image_path = os.path.join(save_dir, f"{stem}.png")
        save_mask_path = os.path.join(save_mask_dir, f"{stem}.png")

        original_image = np.array(cv2.imread(original_image_path)).astype(np.float32) / 255.
        cropped_original_image = cv2.imread(cropped_original_image_path)
        mask_image = np.array(cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)).astype(np.float32) / 255. > 0
        
        w_mask = np.any(mask_image, axis=0).reshape(-1) # (w)
        h_mask = np.any(mask_image, axis=1).reshape(-1) # (h)

        w_mask_valid_idxs = np.nonzero(w_mask)
        h_mask_valid_idxs = np.nonzero(h_mask)

        w_min = np.min(w_mask_valid_idxs)
        w_max = np.max(w_mask_valid_idxs)

        h_min = np.min(h_mask_valid_idxs)
        h_max = np.max(h_mask_valid_idxs)

        cropped_mask_image = mask_image[h_min:h_max, w_min:w_max]  > 0
        cropped_mask_image = largest_connected_true_region(cropped_mask_image)

        mask_image = largest_connected_true_region(mask_image)
        image_masked_image = original_image.copy()
        image_masked_image[mask_image] = image_masked_image[mask_image] * 0.7 + np.array([0., 1., 0.]) * 0.3
        image_masked_image = np.clip(image_masked_image*255., 0, 255).astype(np.uint8)
            
        # MODIFIED: dynamically split stem into prefix and numeric suffix
        prefix, suffix = stem.rsplit('_', 1)
        if not suffix.isdigit():
            raise ValueError(f"Unexpected stem format, cannot parse frame index: {stem}")
        mask_note = (prefix, int(suffix))  # MODIFIED

        if mask_note not in existing_stem_masks:
            cv2.imwrite(save_image_path, image_masked_image)
            cv2.imwrite(save_mask_path, np.clip((mask_image > 0).astype(np.float32)*255., 0, 255).astype(np.uint8))
            existing_stem_masks.append(mask_note)
