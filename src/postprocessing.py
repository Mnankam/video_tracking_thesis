import cv2
import numpy as np


def postprocess_mask(
    mask,
    open_kernel_size=3,
    close_kernel_size=5,
    min_area=100,
    keep_largest=True,
    min_width=1,
    min_height=1,
    min_fill_ratio=0.0,
    return_features=False,
):

    if mask is None:
        if return_features:
            return None, []
        return None

    # -------------------------------------------------------------------------
    # 1. Convert to grayscale and binary representation
    # -------------------------------------------------------------------------
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_bin = (mask > 0).astype(np.uint8) * 255

    # -------------------------------------------------------------------------
    # 2. Morphological opening
    #    Removes small foreground artefacts.
    # -------------------------------------------------------------------------
    if open_kernel_size is not None and open_kernel_size > 1:
        kernel_open = np.ones(
            (open_kernel_size, open_kernel_size),
            dtype=np.uint8,
        )

        mask_bin = cv2.morphologyEx(
            mask_bin,
            cv2.MORPH_OPEN,
            kernel_open,
        )

    # -------------------------------------------------------------------------
    # 3. Morphological closing
    #    Fills small gaps and regularises foreground regions.
    # -------------------------------------------------------------------------
    if close_kernel_size is not None and close_kernel_size > 1:
        kernel_close = np.ones(
            (close_kernel_size, close_kernel_size),
            dtype=np.uint8,
        )

        mask_bin = cv2.morphologyEx(
            mask_bin,
            cv2.MORPH_CLOSE,
            kernel_close,
        )

    # -------------------------------------------------------------------------
    # 4. Connected-component analysis
    # -------------------------------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bin,
        connectivity=8,
    )

    component_mask = np.zeros_like(mask_bin)

    # No foreground component was detected.
    if num_labels <= 1:
        if return_features:
            return component_mask, []
        return component_mask

    valid_labels = []

    # -------------------------------------------------------------------------
    # 5. Area-based connected-component filtering
    # -------------------------------------------------------------------------
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area >= min_area:
            valid_labels.append(label_id)

    if not valid_labels:
        if return_features:
            return component_mask, []
        return component_mask

    if keep_largest:
        largest_label = max(
            valid_labels,
            key=lambda idx: stats[idx, cv2.CC_STAT_AREA],
        )

        valid_labels = [largest_label]

    for label_id in valid_labels:
        component_mask[labels == label_id] = 255

    # -------------------------------------------------------------------------
    # 6. Contour extraction
    # -------------------------------------------------------------------------
    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    refined_mask = np.zeros_like(mask_bin)
    features = []

    # -------------------------------------------------------------------------
    # 7–9. Centroid computation, bounding boxes and geometric filtering
    # -------------------------------------------------------------------------
    for contour in contours:

        contour_area = cv2.contourArea(contour)

        if contour_area < min_area:
            continue

        # Bounding-box estimation
        x, y, width, height = cv2.boundingRect(contour)

        if width < min_width or height < min_height:
            continue

        bbox_area = width * height

        if bbox_area <= 0:
            continue

        fill_ratio = contour_area / float(bbox_area)

        if fill_ratio < min_fill_ratio:
            continue

        # Centroid computation from contour moments
        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            centroid_x = moments["m10"] / moments["m00"]
            centroid_y = moments["m01"] / moments["m00"]
        else:
            # Fallback for degenerate contour
            centroid_x = x + width / 2.0
            centroid_y = y + height / 2.0

        centroid = (
            float(centroid_x),
            float(centroid_y),
        )

        bounding_box = (
            int(x),
            int(y),
            int(width),
            int(height),
        )

        # Preserve valid contour in the final mask
        cv2.drawContours(
            refined_mask,
            [contour],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )

        features.append(
            {
                "contour": contour,
                "area": float(contour_area),
                "centroid": centroid,
                "bounding_box": bounding_box,
                "fill_ratio": float(fill_ratio),
            }
        )

    # If requested, keep only the largest geometrically valid contour.
    if keep_largest and len(features) > 1:

        largest_index = int(
            np.argmax([feature["area"] for feature in features])
        )

        largest_feature = features[largest_index]

        refined_mask[:] = 0

        cv2.drawContours(
            refined_mask,
            [largest_feature["contour"]],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED,
        )

        features = [largest_feature]

    # -------------------------------------------------------------------------
    # 10. Export result
    # -------------------------------------------------------------------------
    if return_features:
        return refined_mask, features

    return refined_mask