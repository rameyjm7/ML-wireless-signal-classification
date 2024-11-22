NAME = ml-wireless-signal-classification
CONTAINER_NAME = ${NAME}_container

.PHONY: build start stop rebuild interactive super-clean logs train

build:
	#@echo "Creating a tarball for the source code..."
	#@cd ..; mkdir -p /workspace/code; tar -czf code.tar.gz src/ tests/ setup.py RML2016.10a_dict.pkl; mv code.tar.gz docker/
	@echo "Building Docker image: ${NAME}..."
	docker build -t ${NAME} .

start:
	@echo "Starting Docker container: ${CONTAINER_NAME}..."
	docker run -d --gpus all \
		-p 5000:5000 -p 3000:3000 \
		--name ${CONTAINER_NAME} ${NAME}

stop:
	@echo "Stopping and removing Docker container: ${CONTAINER_NAME}..."
	docker stop ${CONTAINER_NAME} || true
	docker rm ${CONTAINER_NAME} || true

rebuild:
	@echo "Rebuilding Docker image and restarting container..."
	make stop
	make build
	make start

interactive:
	@echo "Starting Docker container in interactive mode with GPU support..."
	docker run --gpus all -it --name ${CONTAINER_NAME} ${NAME} /bin/bash

super-clean:
	@echo "Performing a full system cleanup of Docker images, containers, and unused data..."
	docker system prune -a -f

logs:
	@echo "Fetching logs from Docker container: ${CONTAINER_NAME}..."
	docker logs -f ${CONTAINER_NAME}

train:
	@echo "Running training command inside the Docker container..."
	docker run --gpus all --rm ${NAME} python3 -m ml_wireless_classification
