# Design and Evaluation of a Scalable GPU-Accelerated Pipeline  
## for Automated Segmentation and Position Tracking in High-Speed Flow Loop Videos

---

## Overview

This repository contains the implementation of a scalable, GPU-accelerated video processing pipeline developed as part of a Master's thesis.

The objective is the automated segmentation and position tracking of pipe structures and particle beds in high-speed flow loop experiments (~100,000 frames at 200 FPS).

Due to illumination flickering (50 Hz lab power supply) and the large data volume, manual analysis is infeasible. This project provides a fully automated and reproducible processing framework deployable on HPC infrastructure (GWDG SCC).

---

## Objectives

- Automated illumination correction (deflickering)
- Robust object segmentation (classical CV and/or deep learning)
- Multi-frame object tracking
- Scalable batch processing using Slurm job arrays
- GPU acceleration for segmentation models
- Quantitative evaluation of segmentation and tracking performance
- Performance benchmarking (CPU vs GPU)

---

## Repository Structure

```
video-tracking-thesis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── deflicker.py
│   ├── segmentation.py
│   ├── tracking.py
│   └── pipeline.py
│
├── scripts/
│   ├── run_chunk.py
│   └── submit_array.sh
│
├── configs/
│   └── config.yaml
│
├── apptainer/
│   └── detectron2.def
│
└── docs/
    └── notes.md
```

---

## Pipeline Architecture

The processing workflow consists of:

1. Frame extraction from raw video
2. Illumination normalization (deflickering)
3. Object segmentation
   - Classical OpenCV-based methods
   - Optional deep learning models (e.g., Detectron2)
4. Multi-frame tracking
5. Result export and visualization

The system is designed to process data in parallel chunks to enable scalable execution on HPC clusters.

---

## HPC Deployment (GWDG SCC)

The pipeline is executed on the SCC cluster using:

- Slurm job arrays
- Apptainer containers
- GPU acceleration (if available)

Example submission:

```bash
sbatch scripts/submit_array.sh


Local Development (WSL / Linux)

Create virtual environment:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

#########################################################
Data Handling

Raw video files are not included in this repository.

Due to large file sizes (~100k frames), processing is performed in chunk-based batches suitable for distributed HPC execution.

#########################################################

Evaluation Metrics:

 - Intersection over Union (IoU)

 - Tracking accuracy

 - Frame processing time

 - GPU vs CPU performance comparison

 Author

Serge Muriel Kouom Nankam
M.Ing. Electrical Engineering and Information Technology
HAWK University of Applied Sciences

