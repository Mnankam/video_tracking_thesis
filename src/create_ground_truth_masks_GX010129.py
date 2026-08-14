#!/usr/bin/env python3
"""
create_ground_truth_masks_GX010129.py
=====================================

Creates binary ground-truth masks from LabelMe JSON annotations for GX010129.

Expected input structure
------------------------

data_gt/ground_truth/GX010129/
├── frames/
│   ├── frame_000000.png
│   ├── frame_000000.json
│   ├── frame_001500.png
│   ├── frame_001500.json
│   └── ...
├── predictions/
│   ├── inner_pipe/
│   └── particle_bed/
└── selected_frames.csv

Output structure
----------------

data_gt/ground_truth/GX010129/
└── ground_truth/
    ├── inner_pipe/
    │   ├── frame_000000_inner_pipe_gt.png
    │   └── ...
    └── particle_bed/
        ├── frame_000000_particle_bed_gt.png
        └── ...

Mask convention
---------------

0   = background
255 = annotated object

The script expects exactly the two semantic classes:

    *_inner_pipe
    *_particle_bed

Author
------
Serge Kouomnankam
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(
    "data_gt/ground_truth/GX010129"
)

FRAMES_DIR = ROOT / "frames"

GROUND_TRUTH_ROOT = ROOT / "ground_truth"

INNER_PIPE_GT_DIR = (
    GROUND_TRUTH_ROOT / "inner_pipe"
)

PARTICLE_BED_GT_DIR = (
    GROUND_TRUTH_ROOT / "particle_bed"
)


def create_directories() -> None:
    """Create output directories."""

    INNER_PIPE_GT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PARTICLE_BED_GT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_json(path: Path) -> dict:
    """Load one LabelMe JSON file."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def find_image_for_json(
    json_path: Path,
) -> Path:
    """
    Locate the source image corresponding to one JSON annotation.
    """

    image_path = (
        json_path.parent
        / f"{json_path.stem}.png"
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            "Corresponding image not found:\n"
            f"{image_path}"
        )

    return image_path


def polygon_to_int_array(
    points,
) -> np.ndarray:
    """
    Convert LabelMe polygon coordinates to an OpenCV-compatible array.
    """

    array = np.asarray(
        points,
        dtype=np.float64,
    )

    if (
        array.ndim != 2
        or array.shape[1] != 2
    ):
        raise ValueError(
            f"Invalid polygon shape: {array.shape}"
        )

    if len(array) < 3:
        raise ValueError(
            "A polygon requires at least three points."
        )

    array = np.rint(
        array
    ).astype(
        np.int32
    )

    return array


def classify_label(
    label: str,
) -> str | None:
    """
    Map LabelMe label to one of the two ground-truth classes.
    """

    if label.endswith(
        "_inner_pipe"
    ):
        return "inner_pipe"

    if label.endswith(
        "_particle_bed"
    ):
        return "particle_bed"

    return None


def rasterize_annotation(
    json_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict,
]:
    """
    Rasterize one LabelMe JSON file into two binary masks.

    Returns
    -------
    inner_pipe_mask
    particle_bed_mask
    statistics
    """

    data = load_json(
        json_path
    )

    image_path = find_image_for_json(
        json_path
    )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            "Image could not be loaded:\n"
            f"{image_path}"
        )

    height, width = image.shape[:2]

    inner_pipe_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    particle_bed_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    num_inner_pipe_shapes = 0
    num_particle_bed_shapes = 0
    ignored_labels: list[str] = []

    shapes = data.get(
        "shapes",
        []
    )

    for shape in shapes:

        label = str(
            shape.get(
                "label",
                ""
            )
        )

        object_class = classify_label(
            label
        )

        if object_class is None:
            ignored_labels.append(
                label
            )
            continue

        points = shape.get(
            "points",
            []
        )

        if len(points) < 3:
            print(
                f"[WARNING] {json_path.name}: "
                f"shape '{label}' has fewer than 3 points."
            )
            continue

        polygon = polygon_to_int_array(
            points
        )

        # Clamp polygon coordinates to image boundaries.
        polygon[:, 0] = np.clip(
            polygon[:, 0],
            0,
            width - 1,
        )

        polygon[:, 1] = np.clip(
            polygon[:, 1],
            0,
            height - 1,
        )

        polygon_cv = polygon.reshape(
            (-1, 1, 2)
        )

        if object_class == "inner_pipe":

            cv2.fillPoly(
                inner_pipe_mask,
                [polygon_cv],
                255,
            )

            num_inner_pipe_shapes += 1

        elif object_class == "particle_bed":

            cv2.fillPoly(
                particle_bed_mask,
                [polygon_cv],
                255,
            )

            num_particle_bed_shapes += 1

    statistics = {
        "frame": json_path.stem,
        "inner_pipe_shapes": num_inner_pipe_shapes,
        "particle_bed_shapes": num_particle_bed_shapes,
        "ignored_labels": ignored_labels,
        "width": width,
        "height": height,
        "inner_pipe_pixels": int(
            np.count_nonzero(
                inner_pipe_mask
            )
        ),
        "particle_bed_pixels": int(
            np.count_nonzero(
                particle_bed_mask
            )
        ),
    }

    return (
        inner_pipe_mask,
        particle_bed_mask,
        statistics,
    )


def output_paths(
    frame_name: str,
) -> tuple[
    Path,
    Path,
]:
    """Return GT output paths for one frame."""

    inner_pipe_path = (
        INNER_PIPE_GT_DIR
        / f"{frame_name}_inner_pipe_gt.png"
    )

    particle_bed_path = (
        PARTICLE_BED_GT_DIR
        / f"{frame_name}_particle_bed_gt.png"
    )

    return (
        inner_pipe_path,
        particle_bed_path,
    )


def save_mask(
    path: Path,
    mask: np.ndarray,
) -> None:
    """Save one binary mask."""

    success = cv2.imwrite(
        str(path),
        mask,
    )

    if not success:
        raise RuntimeError(
            f"Mask could not be written:\n{path}"
        )


def process_all_annotations() -> list[dict]:
    """Process all LabelMe JSON files."""

    json_files = sorted(
        FRAMES_DIR.glob(
            "*.json"
        )
    )

    if not json_files:
        raise FileNotFoundError(
            "No LabelMe JSON files found in:\n"
            f"{FRAMES_DIR}"
        )

    print("=" * 80)
    print("GX010129 Ground-Truth Mask Generation")
    print("=" * 80)
    print(f"Input directory:   {FRAMES_DIR}")
    print(f"JSON files:        {len(json_files)}")
    print()

    all_statistics: list[dict] = []

    for number, json_path in enumerate(
        json_files,
        start=1,
    ):

        (
            inner_pipe_mask,
            particle_bed_mask,
            statistics,
        ) = rasterize_annotation(
            json_path
        )

        frame_name = (
            json_path.stem
        )

        (
            inner_pipe_path,
            particle_bed_path,
        ) = output_paths(
            frame_name
        )

        save_mask(
            inner_pipe_path,
            inner_pipe_mask,
        )

        save_mask(
            particle_bed_path,
            particle_bed_mask,
        )

        all_statistics.append(
            statistics
        )

        print(
            f"[{number:02d}/{len(json_files):02d}] "
            f"{frame_name} | "
            f"inner_pipe shapes="
            f"{statistics['inner_pipe_shapes']} | "
            f"particle_bed shapes="
            f"{statistics['particle_bed_shapes']} | "
            f"inner_pipe pixels="
            f"{statistics['inner_pipe_pixels']} | "
            f"particle_bed pixels="
            f"{statistics['particle_bed_pixels']}"
        )

        if statistics[
            "ignored_labels"
        ]:
            print(
                "    ignored labels:",
                statistics[
                    "ignored_labels"
                ],
            )

    return all_statistics


def validate_results(
    statistics: list[dict],
) -> None:
    """
    Perform basic consistency checks on the generated masks.
    """

    problems: list[str] = []

    for item in statistics:

        frame = item[
            "frame"
        ]

        if (
            item[
                "inner_pipe_shapes"
            ]
            == 0
        ):
            problems.append(
                f"{frame}: no inner-pipe annotation"
            )

        if (
            item[
                "particle_bed_shapes"
            ]
            == 0
        ):
            problems.append(
                f"{frame}: no particle-bed annotation"
            )

        if (
            item[
                "inner_pipe_pixels"
            ]
            == 0
        ):
            problems.append(
                f"{frame}: empty inner-pipe mask"
            )

        if (
            item[
                "particle_bed_pixels"
            ]
            == 0
        ):
            problems.append(
                f"{frame}: empty particle-bed mask"
            )

    print()
    print("=" * 80)
    print("Validation")
    print("=" * 80)

    if problems:
        print(
            f"Detected {len(problems)} potential problem(s):"
        )

        for problem in problems:
            print(
                f"  - {problem}"
            )
    else:
        print(
            "All annotations produced non-empty "
            "inner-pipe and particle-bed masks."
        )


def print_summary(
    statistics: list[dict],
) -> None:
    """Print output summary."""

    inner_pipe_files = list(
        INNER_PIPE_GT_DIR.glob(
            "*.png"
        )
    )

    particle_bed_files = list(
        PARTICLE_BED_GT_DIR.glob(
            "*.png"
        )
    )

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    print(
        f"Processed annotations:       {len(statistics)}"
    )

    print(
        f"Inner-pipe GT masks:         {len(inner_pipe_files)}"
    )

    print(
        f"Particle-bed GT masks:       {len(particle_bed_files)}"
    )

    print()
    print(
        "Inner-pipe masks:"
    )

    print(
        f"  {INNER_PIPE_GT_DIR}"
    )

    print()
    print(
        "Particle-bed masks:"
    )

    print(
        f"  {PARTICLE_BED_GT_DIR}"
    )

    print()
    print(
        "Mask convention:"
    )

    print(
        "  0   = background"
    )

    print(
        "  255 = object"
    )


def main() -> None:
    """Program entry point."""

    create_directories()

    statistics = (
        process_all_annotations()
    )

    validate_results(
        statistics
    )

    print_summary(
        statistics
    )


if __name__ == "__main__":
    main()