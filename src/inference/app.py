"""
FastAPI application for Step 4.

Endpoints:
- GET /health
- GET /ready
- GET /dashboard/summary
- POST /semantic-search
- POST /rag-answer
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.inference.rag_engine import RagEngine, serialise_search_result
from src.inference.schemas import (
    DashboardSummary,
    RagAnswer,
    RagRequest,
    SearchRequest,
    SearchResponse,
    SearchResult,
)


DEFAULT_INDEX_PATH = "data/vector_store/local_vector_index.json"

app = FastAPI(
    title="AI Ecosystem Intelligence Model API",
    description="Step 4 FastAPI service for semantic search and grounded RAG answers.",
    version="0.4.0",
)


@lru_cache(maxsize=1)
def load_engine() -> RagEngine:
    index_path = Path(os.getenv("VECTOR_INDEX_PATH", DEFAULT_INDEX_PATH))
    if not index_path.exists():
        raise FileNotFoundError(f"Vector index not found: {index_path}")
    return RagEngine.from_index_path(index_path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "model-api"}


@app.get("/ready")
def ready() -> dict[str, str]:
    try:
        engine = load_engine()
    except FileNotFoundError as missing_index:
        raise HTTPException(status_code=503, detail=str(missing_index)) from missing_index
    return {"status": "ready", "indexed_documents": str(len(engine.records))}


@app.get("/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary() -> DashboardSummary:
    engine = load_engine()
    return DashboardSummary(
        status="ready",
        model_version=engine.model_version,
        indexed_documents=len(engine.records),
        document_type_counts=engine.document_type_counts(),
    )


@app.post("/semantic-search", response_model=SearchResponse)
def semantic_search(request: SearchRequest) -> SearchResponse:
    engine = load_engine()
    raw_results = engine.search_documents(request.query, request.top_k)
    results = [SearchResult(**serialise_search_result(result)) for result in raw_results]
    return SearchResponse(
        query=request.query,
        model_version=engine.model_version,
        result_count=len(results),
        results=results,
    )


@app.post("/rag-answer", response_model=RagAnswer)
def rag_answer(request: RagRequest) -> RagAnswer:
    engine = load_engine()
    answer, evidence, latency_ms = engine.answer_question(request.question, request.top_k)
    results = [SearchResult(**serialise_search_result(result)) for result in evidence]
    return RagAnswer(
        question=request.question,
        answer=answer,
        model_version=engine.model_version,
        evidence_count=len(results),
        evidence=results,
        latency_ms=round(latency_ms, 3),
    )

