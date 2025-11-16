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
TOKEN := mlsc
PORT := 8888

.PHONY: build push asif shell run clean

build:
	@echo ">>> Building tarball with full project contents..."
	cd .. && tar -czf docker/code.tar.gz \
		src/ \
		tests/ \
		models/ \
		data/ \
		docs/ \
		notebooks/ \
		scratch/ \
		setup.py \
		LICENSE \
		README.md \
		RML2016.10a_dict.pkl
	@echo ">>> Building Docker image $(FULL_IMAGE)..."
	docker build --build-arg GITHASH=$(GITHASH) -t $(FULL_IMAGE) -f docker/Dockerfile docker/

push:
	@echo ">>> Pushing image to DockerHub: $(FULL_IMAGE)"
	docker push $(FULL_IMAGE)

asif:
	@echo ">>> Pulling image from DockerHub and converting to .sif for HPC..."
	module load apptainer || true
	apptainer build $(SIF_NAME) docker://$(FULL_IMAGE)
	@echo "Created: $(SIF_NAME) from DockerHub image $(FULL_IMAGE)"

shell:
	@echo ">>> Launching interactive shell in container..."
	docker run --rm -it --gpus all -v $(PWD)/..:/workspace $(FULL_IMAGE) /bin/bash

run:
	@echo ">>> Running container with JupyterLab..."
	docker run --rm -it --gpus all -p 8888:8888 -v $(PWD)/..:/workspace $(FULL_IMAGE)

apptainer-lab:
	@echo "Launching Jupyter Lab inside Apptainer on port $(PORT)"
	module load apptainer && \
	apptainer exec --nv $(SIF_NAME) \
	    jupyter lab --ip=0.0.0.0 --port=$(PORT) \
	    --no-browser --allow-root --NotebookApp.token=$(TOKEN)

clean:
	@echo ">>> Cleaning up Docker artifacts..."
	docker rm -f $(CONTAINER_NAME) 2>/dev/null || true
	docker rmi $(FULL_IMAGE) 2>/dev/null || true
	rm -f $(SIF_NAME) docker/code.tar.gz 2>/dev/null || true
