import os
import cv2
import numpy as np
import torch
from typing import List, Dict

def crop_region(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop the image using the bounding box of the foreground in the mask.
    Only pixels inside the mask are kept; background is set to black.
    """
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped_img = image[y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
    return np.where(cropped_mask > 0, cropped_img, 0).astype(np.uint8)

def whole_regions_from_tensor(
    image_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    save_path: str = None
) -> List[Dict[int, np.ndarray]]:
    """
    Crop grayscale image regions based on one-hot label tensor masks.

    Args:
        image_tensor (Tensor): (B, 1, H, W) or (B, H, W), grayscale input.
        label_tensor (Tensor): (B, class_num, H, W), one-hot encoded prediction mask.
        save_path (str, optional): If provided, saves cropped patches.

    Returns:
        List[Dict[int, np.ndarray]]: List of per-class cropped grayscale patches per image.
    """
    if save_path:
        os.makedirs(save_path, exist_ok=True)

    if image_tensor.ndim == 4 and image_tensor.shape[1] == 1:
        image_tensor = image_tensor.squeeze(1)  # (B, H, W)

    image_tensor = image_tensor.cpu().numpy()
    label_tensor = label_tensor.cpu().numpy()

    B, C, H, W = label_tensor.shape
    results = []

    for b in range(B):
        gray_image = image_tensor[b]
        if gray_image.max() <= 1:
            gray_image = (gray_image * 255).astype(np.uint8)
        else:
            gray_image = gray_image.astype(np.uint8)

        sample_result = {}

        for c in range(C):
            mask = label_tensor[b, c] > 0.5
            mask_uint8 = mask.astype(np.uint8)

            cropped = crop_region(gray_image, mask_uint8)
            sample_result[c] = cropped

            if save_path and cropped.shape[0] > 1 and cropped.shape[1] > 1:
                cv2.imwrite(os.path.join(save_path, f"sample{b}_class{c}.png"), cropped)

        results.append(sample_result)

    return results