import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def build_opengl_proj_from_intrinsics(K, W, H, near=0.1, far=3.0):
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # Map pixel coordinate system to NDC (-1...+1)
    # x_ndc = (u/cx - 1) etc, after derivation:
    A =  2.0 * fx / W
    B =  2.0 * fy / H
    C =  2.0 * (cx / W) - 1.0
    D = -2.0 * (cy / H) + 1.0

    # Depth mapping
    E = -(far + near) / (far - near)
    F = -2.0 * far * near / (far - near)

    M = torch.zeros((4,4), dtype=torch.float32, device=K.device)
    M[0,0] = A
    M[0,2] = C
    M[1,1] = B
    M[1,2] = D
    M[2,2] = E
    M[2,3] = F
    M[3,2] = -1.0
    return M

def compute_projection(c2w, intrinsic, W, H):
    T_world2cam = torch.inverse(c2w)

    flip_z = torch.diag(torch.tensor([1,1,-1,1], dtype=T_world2cam.dtype, device=T_world2cam.device))
    T_eyeGL  = flip_z @ T_world2cam
    proj4 = build_opengl_proj_from_intrinsics(intrinsic, W, H, near=0.1, far=10.0)
    mvp = proj4 @ T_eyeGL
    
    return mvp.float().cuda()


def visualize_masks(valid: np.ndarray,
                    binary_mask: np.ndarray,
                    figsize = (12, 4),
                    cmap = 'gray',
                    titles = ('Valid', 'Binary Mask', 'Overlay')):
    """
    Visualize valid, binary_mask, and their overlay side by side.

    Args:
        valid (np.ndarray): Boolean mask obtained by mesh projection, shape (H, W).
        binary_mask (np.ndarray): SAM binary mask after resize, shape (H, W).
        figsize (tuple): Size of the entire figure.
        cmap (str): colormap used for single-channel images.
        titles (tuple): Titles for the three subplots.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. valid
    axes[0].imshow(valid, cmap=cmap)
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    # 2. binary_mask
    axes[1].imshow(binary_mask, cmap=cmap)
    axes[1].set_title(titles[1])
    axes[1].axis('off')

    # 3. Overlay: valid red, binary_mask blue
    overlay = np.zeros((*valid.shape, 3), dtype=np.float32)
    overlay[..., 0] = valid.astype(float)       # R channel
    overlay[..., 2] = binary_mask.astype(float) # B channel
    axes[2].imshow(overlay)
    axes[2].set_title(titles[2])
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

def fill_mask_holes_bool(binary_mask: np.ndarray, kernel_size: int = 20) -> np.ndarray:
    """
    Fill small gaps and holes in a binary mask provided as a boolean array.

    Args:
        binary_mask (np.ndarray): 2D boolean array (True=foreground, False=background).
        kernel_size (int): Size of the square structuring element used for morphological closing.

    Returns:
        np.ndarray: 2D boolean array with holes filled.
    """
    # convert to uint8 0/255
    bin_u8 = (binary_mask.astype(np.uint8)) * 255

    # morphological closing to bridge small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(bin_u8, cv2.MORPH_CLOSE, kernel)

    # flood-fill from the background corner to find external region
    h, w = closed.shape
    ff_mask = np.zeros((h+2, w+2), np.uint8)
    flood_filled = closed.copy()
    cv2.floodFill(flood_filled, ff_mask, (0, 0), 255)

    # invert flood-filled to get holes, then OR with closed result
    holes = cv2.bitwise_not(flood_filled)
    filled_u8 = closed | holes

    # convert back to boolean
    return filled_u8.astype(bool)
