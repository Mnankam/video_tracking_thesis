#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# Postprocessing
# =============================================================================

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
    """
    Refine a binary segmentation mask and optionally extract geometric
    descriptors.

    Processing steps:
        1. Convert mask to binary representation.
        2. Apply morphological opening.
        3. Apply morphological closing.
        4. Perform connected-component analysis.
        5. Apply area-based component filtering.
        6. Extract contours.
        7. Estimate bounding boxes.
        8. Compute centroids.
        9. Apply geometric filtering.
       10. Return refined mask and optional descriptors.
    """

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

    if num_labels <= 1:
        if return_features:
            return component_mask, []
        return component_mask

    valid_labels = []

    # -------------------------------------------------------------------------
    # 5. Area-based filtering
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
    # 7-9. Bounding boxes, centroids and geometric filtering
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

        # Centroid from contour moments
        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            centroid_x = moments["m10"] / moments["m00"]
            centroid_y = moments["m01"] / moments["m00"]
        else:
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

        # Preserve valid region in final mask
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

    # -------------------------------------------------------------------------
    # Keep largest geometrically valid contour if requested
    # -------------------------------------------------------------------------
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
    # 10. Return result
    # -------------------------------------------------------------------------
    if return_features:
        return refined_mask, features

    return refined_mask


# =============================================================================
# Visualization
# =============================================================================

def create_feature_visualization(mask, features):
    """
    Draw contour, bounding box and centroid on the postprocessed mask.

    A grayscale visualization is used so that the generated thesis figure
    remains suitable for monochrome presentation.
    """

    visualization = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    for feature in features:
        contour = feature["contour"]
        x, y, width, height = feature["bounding_box"]
        cx, cy = feature["centroid"]

        # Contour
        cv2.drawContours(
            visualization,
            [contour],
            contourIdx=-1,
            color=(150, 150, 150),
            thickness=2,
        )

        # Bounding box
        cv2.rectangle(
            visualization,
            (x, y),
            (x + width, y + height),
            color=(255, 255, 255),
            thickness=2,
        )

        # Centroid
        cv2.drawMarker(
            visualization,
            (int(round(cx)), int(round(cy))),
            color=(255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )

    return visualization


# =============================================================================
# Single-mask processing
# =============================================================================

def process_single_mask(
    input_path,
    input_root,
    output_root,
    open_kernel_size,
    close_kernel_size,
    min_area,
    keep_largest,
    min_width,
    min_height,
    min_fill_ratio,
):
    """
    Process one mask and save all generated outputs.
    """

    mask = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        print(f"[WARNING] Could not read: {input_path}")
        return []

    refined_mask, features = postprocess_mask(
        mask,
        open_kernel_size=open_kernel_size,
        close_kernel_size=close_kernel_size,
        min_area=min_area,
        keep_largest=keep_largest,
        min_width=min_width,
        min_height=min_height,
        min_fill_ratio=min_fill_ratio,
        return_features=True,
    )

    # Preserve input subdirectory structure.
    relative_path = input_path.relative_to(input_root)
    relative_parent = relative_path.parent

    mask_output_dir = output_root / "masks" / relative_parent
    feature_output_dir = output_root / "geometric_features" / relative_parent

    mask_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = input_path.stem

    refined_path = (
        mask_output_dir /
        f"{stem}_postprocessed.png"
    )

    visualization_path = (
        feature_output_dir /
        f"{stem}_features.png"
    )

    # Save refined binary mask.
    cv2.imwrite(
        str(refined_path),
        refined_mask,
    )

    # Save geometric-feature visualization.
    visualization = create_feature_visualization(
        refined_mask,
        features,
    )

    cv2.imwrite(
        str(visualization_path),
        visualization,
    )

    rows = []

    if not features:
        rows.append(
            {
                "source_file": str(relative_path),
                "region_id": "",
                "area_px": "",
                "centroid_x_px": "",
                "centroid_y_px": "",
                "bbox_x_px": "",
                "bbox_y_px": "",
                "bbox_width_px": "",
                "bbox_height_px": "",
                "fill_ratio": "",
            }
        )

    for region_id, feature in enumerate(features, start=1):
        x, y, width, height = feature["bounding_box"]
        cx, cy = feature["centroid"]

        rows.append(
            {
                "source_file": str(relative_path),
                "region_id": region_id,
                "area_px": feature["area"],
                "centroid_x_px": cx,
                "centroid_y_px": cy,
                "bbox_x_px": x,
                "bbox_y_px": y,
                "bbox_width_px": width,
                "bbox_height_px": height,
                "fill_ratio": feature["fill_ratio"],
            }
        )

    return rows


# =============================================================================
# Batch processing
# =============================================================================

def find_mask_files(input_root):
    """
    Recursively find supported image files.
    """

    extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
    }

    files = [
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]

    return sorted(files)


def write_feature_csv(rows, output_path):
    """
    Write extracted geometric descriptors to CSV.
    """

    fieldnames = [
        "source_file",
        "region_id",
        "area_px",
        "centroid_x_px",
        "centroid_y_px",
        "bbox_x_px",
        "bbox_y_px",
        "bbox_width_px",
        "bbox_height_px",
        "fill_ratio",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# Command-line interface
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess segmentation masks and extract contours, "
            "centroids and bounding boxes."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
            "video_tracking_thesis/outputs/"
            "Internal_Validation_GX010129/masks"
        ),
        help="Root directory containing segmentation masks.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/ceph-hdd/projects/mthesis_s_kouomnankam/"
            "video_tracking_thesis/outputs/"
            "Internal_Validation_GX010129/results_postprocessing"
        ),
        help="Directory used for postprocessing results.",
    )

    parser.add_argument(
        "--open-kernel",
        type=int,
        default=3,
        help="Morphological opening kernel size.",
    )

    parser.add_argument(
        "--close-kernel",
        type=int,
        default=5,
        help="Morphological closing kernel size.",
    )

    parser.add_argument(
        "--min-area",
        type=float,
        default=100,
        help="Minimum region area in pixels.",
    )

    parser.add_argument(
        "--min-width",
        type=int,
        default=1,
        help="Minimum bounding-box width.",
    )

    parser.add_argument(
        "--min-height",
        type=int,
        default=1,
        help="Minimum bounding-box height.",
    )

    parser.add_argument(
        "--min-fill-ratio",
        type=float,
        default=0.0,
        help="Minimum contour-to-bounding-box area ratio.",
    )

    parser.add_argument(
        "--keep-all",
        action="store_true",
        help=(
            "Keep all valid connected regions instead of only "
            "the largest region."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_arguments()

    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n{input_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_files = find_mask_files(input_root)

    if not mask_files:
        raise RuntimeError(
            f"No mask images found in:\n{input_root}"
        )

    print("=" * 79)
    print("POSTPROCESSING")
    print("=" * 79)
    print(f"Input directory : {input_root}")
    print(f"Output directory: {output_root}")
    print(f"Images found    : {len(mask_files)}")
    print(f"Opening kernel  : {args.open_kernel}")
    print(f"Closing kernel  : {args.close_kernel}")
    print(f"Minimum area    : {args.min_area}")
    print(f"Keep largest    : {not args.keep_all}")
    print("=" * 79)

    all_rows = []
    processed_count = 0

    for index, input_path in enumerate(mask_files, start=1):
        relative_path = input_path.relative_to(input_root)

        print(
            f"[{index:5d}/{len(mask_files):5d}] "
            f"{relative_path}"
        )

        rows = process_single_mask(
            input_path=input_path,
            input_root=input_root,
            output_root=output_root,
            open_kernel_size=args.open_kernel,
            close_kernel_size=args.close_kernel,
            min_area=args.min_area,
            keep_largest=not args.keep_all,
            min_width=args.min_width,
            min_height=args.min_height,
            min_fill_ratio=args.min_fill_ratio,
        )

        all_rows.extend(rows)
        processed_count += 1

    csv_path = output_root / "geometric_features.csv"

    write_feature_csv(
        all_rows,
        csv_path,
    )

    print()
    print("=" * 79)
    print("POSTPROCESSING COMPLETED")
    print("=" * 79)
    print(f"Processed images : {processed_count}")
    print(f"Feature records  : {len(all_rows)}")
    print(f"Feature CSV      : {csv_path}")
    print(f"Results directory: {output_root}")
    print("=" * 79)


if __name__ == "__main__":
    main()