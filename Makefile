PROJECT_NAME=ai-ecosystem-intelligence-platform
NAMESPACE=ai-ecosystem

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make scaffold-check        Check repository folders"
	@echo "  make index                 Build RAG documents and vector index"
	@echo "  make test-step7            Check Docker Compose setup"
	@echo "  make compose-up            Start local Docker stack"
	@echo "  make compose-down          Stop local Docker stack"
	@echo "  make k8s-deploy            Deploy manifests after Kubernetes files are implemented"

.PHONY: scaffold-check
scaffold-check:
	@test -d docker/model-api
	@test -d docker/data-indexer
	@test -d docker/ab-router
	@test -d docker/airflow
	@test -d src/indexing
	@test -d src/inference
	@test -d src/router
	@test -d src/fine_tuning
	@test -d src/evaluation
	@test -d data/dashboard
	@echo "Repository scaffold looks correct."

.PHONY: index
index:
	python scripts/build_rag_documents.py
	python scripts/build_vector_index.py

.PHONY: test-step7
test-step7:
	python scripts/check_docker_compose_setup.py --run-tests

.PHONY: compose-up
compose-up:
	docker compose up --build

.PHONY: compose-down
compose-down:
	docker compose down

.PHONY: k8s-deploy
k8s-deploy:
	kubectl apply -f k8s/manifests/

