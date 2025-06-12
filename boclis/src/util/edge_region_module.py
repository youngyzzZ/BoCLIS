import torch
import numpy as np
import cv2
from typing import List, Dict

def safe_crop(center_y, center_x, half_height, half_width, shape):
    h, w = shape
    top = max(center_y - half_height, 0)
    bottom = min(center_y + half_height, h)
    left = max(center_x - half_width, 0)
    right = min(center_x + half_width, w)
    return top, bottom, left, right

def extract_foreground_patch_random(edge_arr, patch_ratio=0.25):
    points = np.column_stack(np.where(edge_arr == 255))
    if len(points) == 0:
        return np.zeros_like(edge_arr)
    h_range = points[:, 0].max() - points[:, 0].min()
    w_range = points[:, 1].max() - points[:, 1].min()
    half_h = max(int(h_range * patch_ratio), 1)
    half_w = max(int(w_range * patch_ratio), 1)
    np.random.seed(0)
    idx = np.random.choice(len(points))
    y, x = points[idx]
    top, bottom, left, right = safe_crop(y, x, half_h, half_w, edge_arr.shape)
    mask = np.zeros_like(edge_arr)
    mask[top:bottom, left:right] = 1
    return edge_arr * mask

def dilate_and_mask(patch_arr, kernel, iterations, region_mask):
    return cv2.dilate(patch_arr, kernel, iterations=iterations) * region_mask

def crop_and_mask_by_edge(gray_image, mask):
    y_indices, x_indices = np.where(mask > 0)
    if len(y_indices) == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    cropped_image = gray_image[y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
    return np.where(cropped_mask > 0, cropped_image, 0).astype(np.uint8)

def edge_regions_from_tensor(image_tensor: torch.Tensor, label_tensor: torch.Tensor) -> List[Dict[int, np.ndarray]]:
    """
    Extract local edge regions from model prediction output and crop grayscale image accordingly.
    """
    if image_tensor.ndim == 4 and image_tensor.shape[1] == 1:
        image_tensor = image_tensor.squeeze(1)

    image_tensor = image_tensor.cpu().numpy()
    label_tensor = label_tensor.cpu().numpy()

    B, C, H, W = label_tensor.shape
    kernel = np.ones((2, 2), np.uint8)
    kernel_cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    results = []

    for b in range(B):
        gray_image = (image_tensor[b] * 255).astype(np.uint8) if image_tensor[b].max() <= 1 else image_tensor[b].astype(np.uint8)
        seg = np.argmax(label_tensor[b], axis=0).astype(np.uint8)
        edges = cv2.Canny(seg, 32, 128)
        dilated_edges = cv2.dilate((edges > 0).astype(np.uint8) * 255, kernel, iterations=1)

        sample_result = {}
        for c in range(C):
            region_mask = (seg == c).astype(np.uint8)
            dilated_edge_arr = dilated_edges * region_mask
            part_edge_arr = extract_foreground_patch_random(dilated_edge_arr)
            iterations = 3 if c == 1 else 6
            dilated_part_edge_arr = dilate_and_mask(part_edge_arr, kernel_cross, iterations, region_mask)
            cropped = crop_and_mask_by_edge(gray_image, dilated_part_edge_arr)
            sample_result[c] = cropped
        results.append(sample_result)

    return results