# ML Wireless Signal Classification

This repository contains the Docker setup and code for training and evaluating machine learning models for wireless signal classification.

## Prerequisites

Ensure the following dependencies are installed on your system:
1. **NVIDIA Container Toolkit**: [Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. **Docker CE**: [Docker Installation](https://docs.docker.com/get-docker/)
3. **Make**: Install with `sudo apt-get install -y build-essential`
4. **NVIDIA Drivers**: Ensure drivers compatible with CUDA 11.8 are installed.

## Makefile Commands

| Command            | Description                                      |
|--------------------|--------------------------------------------------|
| `make build`       | Build the Docker image with your code included.  |
| `make interactive` | Open an interactive shell inside the container.  |
| `make start`       | Start the container in detached mode.            |
| `make stop`        | Stop and remove the running container.           |
| `make rebuild`     | Rebuild the Docker image and restart the container. |
| `make train`       | Run the training script inside the container.    |
| `make super-clean` | Remove all Docker containers and images.         |

## Setup and Usage

1. Build the Docker image:

   ```make build```
3. Train the top model: This will load from __main__.py

   ```make train```
