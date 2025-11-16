# --------------------------------------------------------------
# Makefile for ML Wireless Signal Classification HPC Container
# --------------------------------------------------------------

NAME = ml-wireless-signal-classification
DOCKERHUB_USER = rameyjm7
IMAGE_TAG = latest
FULL_IMAGE = $(DOCKERHUB_USER)/$(NAME):$(IMAGE_TAG)
CONTAINER_NAME = $(NAME)_container
SIF_NAME = $(NAME)-hpc.sif
GITHASH := $(shell git rev-parse --short HEAD 2>/dev/null || echo unknown)

.PHONY: build push asif shell run clean

build:
	@echo ">>> Building tarball for source code..."
	cd .. && tar -czf docker/code.tar.gz src/ tests/ setup.py RML2016.10a_dict.pkl
	@echo ">>> Building Docker image $(FULL_IMAGE)..."
	docker build --build-arg GITHASH=$(GITHASH) -t $(FULL_IMAGE) -f docker/Dockerfile docker/

push:
	@echo ">>> Pushing image to DockerHub: $(FULL_IMAGE)"
	docker push $(FULL_IMAGE)

asif:
	@echo ">>> Converting Docker image to .sif for HPC..."
	apptainer build $(SIF_NAME) docker-daemon://$(FULL_IMAGE)
	@echo "Created: $(SIF_NAME)"

shell:
	@echo ">>> Launching interactive shell in container..."
	docker run --rm -it --gpus all -v $(PWD)/..:/workspace $(FULL_IMAGE) /bin/bash

run:
	@echo ">>> Running container with JupyterLab..."
	docker run --rm -it --gpus all -p 8888:8888 -v $(PWD)/..:/workspace $(FULL_IMAGE)

clean:
	@echo ">>> Cleaning up Docker artifacts..."
	docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	docker rmi $(FULL_IMAGE) 2>/dev/null || true
	rm -f $(SIF_NAME)
