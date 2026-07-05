import cv2
import numpy as np


def postprocess_mask(
    mask,
    open_kernel_size=3,
    close_kernel_size=5,
    min_area=100,
    keep_largest=True,
):
    if mask is None:
        return None

    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_bin = (mask > 0).astype(np.uint8) * 255

    if open_kernel_size is not None and open_kernel_size > 1:
        kernel_open = np.ones((open_kernel_size, open_kernel_size), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_open)

    if close_kernel_size is not None and close_kernel_size > 1:
        kernel_close = np.ones((close_kernel_size, close_kernel_size), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin,
        connectivity=8,
    )

    if num_labels <= 1:
        return mask_bin

    cleaned = np.zeros_like(mask_bin)

    if keep_largest:
        areas = stats[1:, cv2.CC_STAT_AREA]

        if len(areas) == 0:
            return mask_bin

        largest_label = 1 + int(np.argmax(areas))
        largest_area = stats[largest_label, cv2.CC_STAT_AREA]

        if largest_area >= min_area:
            cleaned[labels == largest_label] = 255

        return cleaned

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label_id] = 255

    return cleaned
