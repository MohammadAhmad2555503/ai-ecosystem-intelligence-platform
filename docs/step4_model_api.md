# Step 4: FastAPI Model and RAG Inference API

Step 4 turns the local vector index into an application service.

## Language

Python 3.11

## Main dependency

FastAPI is used for the serving layer. The RAG logic still uses the local Step 3 index.

## Added files

```text
src/inference/schemas.py
src/inference/rag_engine.py
src/inference/app.py
scripts/run_model_api.py
scripts/test_model_api_logic.py
docker/model-api/Dockerfile
docker/model-api/requirements.txt
```

## Endpoints

```text
GET  /health
GET  /ready
GET  /dashboard/summary
POST /semantic-search
POST /rag-answer
```

## Run tests

```bash
python scripts/test_model_api_logic.py
```

## Run API locally

```bash
pip install -r docker/model-api/requirements.txt
python scripts/run_model_api.py
```

Then open:

```text
http://localhost:8000/docs
```

## Example request

```bash
curl -X POST "http://localhost:8000/rag-answer" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Why is Microsoft influential in the AI ecosystem?\",\"top_k\":5}"
```

## Why this step matters

This makes the dissertation outputs queryable through a real API. The answer is grounded in the Step 5 dashboard outputs through the Step 2 documents and Step 3 vector index.

