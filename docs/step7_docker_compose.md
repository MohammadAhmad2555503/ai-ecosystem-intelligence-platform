# Step 7: Docker Compose Containerisation

Step 7 makes the local platform runnable as containers.

## Language

Python 3.11 for the services. Docker Compose for local orchestration.

## Services

```text
data-indexer
model-api
ab-router
mlflow
```

## Added files

```text
docker-compose.yml
docker/data-indexer/Dockerfile
docker/data-indexer/requirements.txt
docker/mlflow/Dockerfile
docker/mlflow/requirements.txt
scripts/check_docker_compose_setup.py
docs/step7_docker_compose.md
```

## Run local checks

```bash
python scripts/check_docker_compose_setup.py --run-tests
```

## Start the stack

```bash
docker compose up --build
```

## Service URLs

```text
Model API: http://localhost:8000/docs
A/B Router: http://localhost:8080/docs
MLflow: http://localhost:5000
```

## Why this matters

This turns the project from a group of scripts into a runnable local platform. It also prepares the project for Kubernetes because each major component is now isolated as a service.

