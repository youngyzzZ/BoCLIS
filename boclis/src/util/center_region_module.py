import numpy as np
import torch
import cv2

def crop_region(image, mask):
    """
    Crop the input image using the tight bounding box of the foreground in the mask.
    Only the area within the mask is preserved; others are set to black.
    """
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    cropped_img = image[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
    return np.where(cropped_mask > 0, cropped_img, 0).astype(np.uint8)

def center_regions_from_tensor(image_tensor: torch.Tensor, label_tensor: torch.Tensor, erosion_iters: dict = None):
    """
    Extract cropped central regions from model predictions using connected components and erosion.

    Args:
        image_tensor (Tensor): Input grayscale images of shape (B, 1, H, W) or (B, H, W).
        label_tensor (Tensor): Model output in one-hot format of shape (B, C, H, W).
        erosion_iters (dict, optional): Number of erosion iterations for each class index.
                                        Example: {0: 4, 1: 2, 2: 5}

    Returns:
        List[Dict[int, np.ndarray]]: A list of dictionaries for each sample.
                                     Each dictionary maps class index to its cropped grayscale patch.
    """
    if image_tensor.ndim == 4 and image_tensor.shape[1] == 1:
        image_tensor = image_tensor.squeeze(1)  # Convert to (B, H, W)
    image_tensor = image_tensor.cpu().numpy()
    label_tensor = label_tensor.cpu().numpy()

    B, class_num, H, W = label_tensor.shape
    results = []

    for b in range(B):
        image = image_tensor[b]
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        sample_result = {}
        for c in range(class_num):
            mask = label_tensor[b, c] > 0.5  # Binarize mask
            mask_255 = mask.astype(np.uint8) * 255

            # Determine erosion iterations for current class
            iters = erosion_iters[c] if erosion_iters and c in erosion_iters else 4
            kernel = np.ones((3, 3), np.uint8)
            eroded = cv2.erode(mask_255, kernel, iterations=iters)

            # Keep only the largest connected component
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
            selected = np.zeros_like(eroded)
            if num_labels > 1:
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                selected[labels == largest_label] = 255
            else:
                selected = eroded

            cropped = crop_region(image, selected)
            sample_result[c] = cropped
        results.append(sample_result)

    return results