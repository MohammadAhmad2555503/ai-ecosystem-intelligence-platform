"""
Request and response models for the Step 4 inference API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    document_id: str
    document_type: str
    title: str
    score: float
    text: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    model_version: str
    result_count: int
    results: list[SearchResult]


class RagRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=10)


class RagAnswer(BaseModel):
    question: str
    answer: str
    model_version: str
    evidence_count: int
    evidence: list[SearchResult]
    latency_ms: float


class DashboardSummary(BaseModel):
    status: str
    model_version: str
    indexed_documents: int
    document_type_counts: dict[str, int]

