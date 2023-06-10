IMAGE_NAME := gcr.io/$(GCP_PROJECT)/$(APP)
CPU_IMAGE_NAME := $(IMAGE_NAME)-cpu
GPU_IMAGE_NAME := $(IMAGE_NAME)-gpu

.PHONY: build-dev build-pro run stop test shell logs push

build-dev: ## builds CPU-optimized docker image for local dev
	docker build -t $(CPU_IMAGE_NAME) \
	--build-arg ARCH=cpu --build-arg MODEL_PATH=$(PWD) \
	--build-arg MODEL_NAME=$(APP) --progress=plain .

build-pro: ## builds GPU-optimized docker image
	docker build -t $(GPU_IMAGE_NAME) \
	--build-arg ARCH=gpu --build-arg MODEL_PATH=$(PWD) \
	--build-arg MODEL_NAME=$(APP) --build-arg=$(N_WORKERS) .

run: ## runs docker container locally
	docker run -t -d --rm -p 7080:7080 -p 7081:7081 \
	--name $(APP) $(CPU_IMAGE_NAME)

stop: ## stops docker container
	docker stop $(APP)

delete: ## deletes container
	docker rm -f $(APP)

test: ## tests that local model endpoint is running
	curl http://localhost:7081/models/
	curl -d '{"instances": ["How to prepare a spanish omelette:"]}' \
		-H "Content-Type: application/json" \
		-X POST http://localhost:7080/predictions/$(APP)

test-prod: ## tests that vertexai model endpoint is running, needs ENDPOINT_ID
	curl \
        -X POST \
        -H "Authorization: Bearer $$(gcloud auth print-access-token)" \
        -H "Content-Type: application/json" \
        https://europe-west4-aiplatform.googleapis.com/v1/projects/$(GCP_PROJECT)/locations/europe-west4/endpoints/$(ENDPOINT_ID):predict \
        -d '{"instances":["How to prepare a spanish omelette:"]}'

shell: ## run interactive shell within local container
	docker exec --interactive --tty $(APP) /bin/bash

logs: ## display local container logs
	docker logs $(APP) --tail=100

push: ## push container to GCR registry
	docker push $(GPU_IMAGE_NAME)

infra: ## create infra in GCP
	terraform apply infra/main.tf

deploy-prod: ## Deploy to vertexai
	python deploy_to_vertexai.py $(APP) $(VERSION) $(APP)-gpu

.DEFAULT_GOAL := help
.PHONY: help
help: ## Displays makefile commands
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' Makefile | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'
