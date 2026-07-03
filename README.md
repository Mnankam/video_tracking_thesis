# Video Tracking Thesis

## Project Overview

This project focuses on the automated segmentation, motion estimation, and position tracking of relevant structures in high-speed experimental flow loop recordings.

The experimental setup generates large-scale video datasets, where individual recordings may contain more than 100,000 frames acquired at a frame rate of 200 FPS.

Due to the large data volume, illumination flickering caused by the 50 Hz laboratory power supply, motion blur, and visually complex backgrounds, manual frame-by-frame analysis is impractical.

The goal of this project is to provide a fully automated, reproducible, and scalable video processing framework for scientific data analysis, deployable on High Performance Computing (HPC) infrastructure provided by the GWDG Scientific Compute Cluster (SCC).

The implemented framework combines classical computer vision algorithms, optical flow based motion estimation, deep learning based segmentation, and large-scale batch processing for efficient analysis of experimental video data.

---

## Project Objectives

The system is designed to achieve the following objectives:

- Automated illumination correction (FFT-based deflickering)
- Robust object segmentation using classical computer vision methods
- Deep learning based segmentation using Detectron2
- Optical flow based motion estimation
  - Lucas-Kanade sparse optical flow
  - Farneback dense optical flow
- Temporal object tracking across multiple frames
- Particle bed edge estimation
- Scalable batch processing using Slurm job arrays
- GPU accelerated inference for deep learning segmentation models
- Quantitative segmentation evaluation using Intersection over Union (IoU)
- Performance benchmarking across CPU and GPU processing configurations
- Reproducible large-scale execution on HPC infrastructure

---

## Research Project Context

This repository contains the implementation developed as part of a Master Thesis in Electrical Engineering and Information Technology at HAWK University of Applied Sciences.

The research focuses on scalable computer vision pipelines for automated analysis of high-speed experimental flow loop recordings in scientific computing environments.

The project investigates how modern computer vision, deep learning, and HPC technologies can be combined to process large experimental video datasets efficiently and reproducibly.

---

## Repository Structure

```text
video_tracking_thesis/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .gitlab-ci.yml
│
├── src/
│   ├── deflicker.py
│   │      FFT-based deflickering for 50 Hz illumination removal
│   │
│   ├── segmentation.py
│   │      Classical OpenCV segmentation methods
│   │
│   ├── detectron2_inference.py
│   │      Deep learning based segmentation using Detectron2
│   │
│   ├── optical_flow.py
│   │      Lucas-Kanade and Farneback optical flow methods
│   │
│   ├── tracking.py
│   │      Temporal object tracking and trajectory estimation
│   │
│   ├── bed_edge.py
│   │      Particle bed edge estimation
│   │
│   ├── pipeline.py
│   │      Main processing pipeline
│   │
│   └── evaluation.py
│          Performance evaluation and statistics
│
├── configs/
│   ├── config.yaml
│   ├── test_inner_pipe_cv1.yaml
│   └── detectron2_config.yaml
│
├── scripts/
│   ├── run_chunk.py
│   ├── submit_array.sh
│   ├── run_batch.sh
│   ├── run_batch_detectron2.sh
│   └── run_batch_cuda.sh
│
├── apptainer/
│   └── detectron2.def
│
├── outputs/
│   ├── results.csv
│   ├── summary.csv
│   ├── masks/
│   ├── debug/
│   ├── overlay_video.mp4
│   └── benchmark_results/
│
├── tests/
│   └── test_pipeline.py
│
└── docs/
    └── notes.md
```

---

## Processing Pipeline Architecture

The implemented processing workflow consists of the following stages:

1. Frame extraction from raw video recordings

2. Illumination normalization and deflickering

3. Region of Interest (ROI) extraction

4. Object segmentation

   - Classical OpenCV-based segmentation  
   - Deep learning based segmentation using Detectron2

5. Motion estimation using optical flow

   - Lucas-Kanade sparse optical flow  
   - Farneback dense optical flow

6. Temporal object tracking

7. Bed edge estimation

8. Result export and visualization

9. Quantitative performance benchmarking

The framework is designed as a modular processing pipeline to support flexible experimentation and large-scale execution.

---

## HPC Deployment (GWDG SCC)

Large-scale video processing experiments are executed on the GWDG Scientific Compute Cluster (SCC).

The framework supports both CPU-based and GPU-based execution.

The deployment architecture includes:

- Slurm job arrays for large-scale batch execution
- Apptainer containers for reproducible environments
- CPU-based OpenCV processing
- GPU accelerated Detectron2 inference
- Experimental CUDA based optical flow implementations
- Parallel execution across multiple compute nodes

Example batch submission:

```bash
sbatch scripts/submit_array.sh
```

Example pipeline execution inside Apptainer:

```bash
apptainer exec \
  -B /mnt/ceph-hdd:/mnt/ceph-hdd \
  container.sif \
  python -m src.pipeline \
  --config configs/config.yaml
```

---

## Local Development (Linux / WSL)

Create Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run local pipeline:

```bash
python -m src.pipeline --config configs/config.yaml
```

---

## Data Handling

Raw experimental video files are not included in this repository.

High-speed recordings may contain more than 100,000 frames per video.

Due to large file sizes and computational complexity, processing is performed using chunk-based execution suitable for distributed HPC environments.

---

## Evaluation Metrics

The implemented benchmarking framework evaluates system performance using the following quantitative metrics:

- Intersection over Union (IoU) for segmentation quality evaluation
- Average processing time per frame
- Processing throughput (Frames Per Second)
- CPU versus GPU runtime comparison
- Scalability under increasing workload
- Optical flow computational performance comparison

---

## Technologies Used

The project uses the following software technologies:

- Python 3.11.9
- OpenCV: 4.13.0
- NumPy: 2.4.2
- Pandas: 3.0.1
- Matplotlib
- PyTorch: 2.10.0 + cu128
- Detectron2
- CUDA
- Apptainer
- Slurm Workload Manager
- Git / GitLab
- Linux / HPC Environment

---

## Author

**Serge Muriel Kouom Nankam**

Master of Engineering (M.Eng.)  
Electrical Engineering and Information Technology  

HAWK University of Applied Sciences  
Faculty of Engineering and Health  

---

