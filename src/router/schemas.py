"""
Request and response models for the Step 5 A/B router.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RoutedRagRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = Field(default=5, ge=1, le=10)
    request_id: str | None = None
    user_id: str = "anonymous"


class RoutedRagResponse(BaseModel):
    request_id: str
    user_id: str
    selected_model: str
    answer: str
    latency_ms: float
    evidence_count: int
    evidence: list[dict[str, Any]]


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=3)
    rating: int = Field(..., ge=1, le=5)
    user_id: str = "anonymous"
    comment: str = ""


class FeedbackResponse(BaseModel):
    status: str
    request_id: str
    rating: int


class RouterSummary(BaseModel):
    status: str
    model_a_weight: float
    model_b_weight: float
    log_path: str

