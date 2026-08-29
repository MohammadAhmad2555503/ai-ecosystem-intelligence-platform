# Step 5: A/B Router and Experiment Logging

Step 5 adds a traffic router in front of the model API.

## Language

Python 3.11

## Main dependency

FastAPI is used for the router service.

## Added files

```text
src/router/config.py
src/router/core.py
src/router/schemas.py
src/router/app.py
scripts/run_ab_router.py
scripts/test_ab_router_logic.py
docker/ab-router/Dockerfile
docker/ab-router/requirements.txt
docs/step5_ab_router.md
```

## What the router does

The router accepts RAG questions, chooses model A or model B using deterministic request-ID hashing, forwards the request to the selected model API, and logs the result.

## Endpoints

```text
GET  /health
GET  /router/summary
POST /rag-answer
POST /feedback
```

## Logs

```text
data/logs/ab_requests.jsonl
data/logs/ab_feedback.jsonl
```

Each prediction log includes:

```text
timestamp_utc
request_id
user_id
selected_model
question
answer
latency_ms
evidence_count
status
```

Each feedback log includes:

```text
timestamp_utc
request_id
user_id
rating
comment
```

## Run tests

```bash
python scripts/test_ab_router_logic.py
```

## Run the router locally

```bash
pip install -r docker/ab-router/requirements.txt
python scripts/run_ab_router.py
```

## Example request

```bash
curl -X POST "http://localhost:8080/rag-answer" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Why is Microsoft influential?\",\"top_k\":5,\"user_id\":\"demo\"}"
```

## Why this matters

This step prepares the project for statistical A/B testing. The next stage can analyse request and feedback logs to decide whether model B is better than model A.

