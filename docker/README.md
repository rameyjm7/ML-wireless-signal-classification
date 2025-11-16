# ML Wireless Signal Classification — Docker Environment

This directory provides the full containerization and orchestration setup for running **Wireless Signal Classification via Deep Learning** on both **local GPU workstations** and **HPC environments (Apptainer)**.
It includes the complete build system for the LSTM-RNN model used in the research paper, with GPU acceleration and Jupyter Lab for training and analysis.

## Contents

| File           | Purpose                                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| **Dockerfile** | Defines the CUDA 12.1 + TensorFlow 2.16 GPU container with all dependencies pre-installed.                   |
| **Makefile**   | Unified automation for both Docker and Apptainer (HPC). Builds, pushes, runs, and converts to `.sif` images. |

---

## Environment Overview

**Base image:** `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
**Installed stacks:**

* Python 3.10 + TensorFlow 2.16 (GPU build)
* NumPy, scikit-learn, seaborn, matplotlib, h5py
* Jupyter Lab 4.2 for interactive development

**Exposed service:** Jupyter Lab on port `8888`

---

## Usage

### 1. Build Docker Image

Build the GPU image locally with your code included:

```bash
make build
```

### 2. Run Locally (Docker)

Start Jupyter Lab on your local GPU workstation:

```bash
make run
```

Access Jupyter Lab at:

```
http://localhost:8888
```

To stop the container:

```bash
docker ps  # find container ID
docker stop <container_id>
```

### 3. Interactive Shell

For debugging or manual experiments:

```bash
make shell
```

### 4. Push Image to Docker Hub

```bash
make push
```

---

## HPC / Apptainer Workflow

### 5. Load Required Module (HPC)

Before running any Apptainer command on the cluster, load the module:

```bash
module load apptainer
```

### 6. Build or Pull `.sif` Image

If building locally from the Docker image:

```bash
make asif
```

Or pull directly from Docker Hub:

```bash
apptainer pull ml-wireless-signal-classification-hpc.sif docker://rameyjm7/ml-wireless-signal-classification:latest
```

### 7. Launch Jupyter Lab on HPC

```bash
apptainer run --nv ml-wireless-signal-classification-hpc.sif jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

### 8. Run Script Non-Interactively

To execute a Python training script directly:

```bash
apptainer exec --nv ml-wireless-signal-classification-hpc.sif python3 -m ml_wireless_classification
```

---

## Cleanup

Remove local Docker artifacts:

```bash
make clean
```

---

## Notes

* The image embeds the Git commit hash (`org.opencontainers.image.revision`) for traceability.
* Compatible with NVIDIA DGX and other CUDA 12.1+ GPU systems.
* The working directory inside the container is `/workspace`.
* The environment was verified with V100, A100, and H200 GPUs.

---

**Maintainer:** Jacob M. Ramey
**Collaborator:** Paras Goda
**DockerHub:** [rameyjm7/ml-wireless-signal-classification](https://hub.docker.com/r/rameyjm7/ml-wireless-signal-classification)
