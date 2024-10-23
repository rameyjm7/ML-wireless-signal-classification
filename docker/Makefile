NAME = ml-wireless-signal-classification

.PHONY: build run stop interactive

build:
	@zip -r code.zip RNN/
	docker build -t ${NAME} .

start:
	docker run -d -p 5000:5000 -p 3000:3000 --name ${NAME}_container ${NAME}

stop:
	docker stop ${NAME}_container || true
	docker rm ${NAME}_container || true

rebuild:
	make stop
	make build
	make start

interactive:
	docker run -it -p 5000:5000 -p 3000:3000 --name ${NAME}_container ${NAME} /bin/bash

super-clean:
	docker system prune -a
