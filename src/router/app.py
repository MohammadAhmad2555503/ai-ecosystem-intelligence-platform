"""
FastAPI A/B router for Step 5.

This service sits in front of model API variants and records experiment logs.
"""

from __future__ import annotations

from fastapi import FastAPI

from src.router.config import load_router_config
from src.router.core import build_feedback_log, route_request, write_feedback_log
from src.router.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    RoutedRagRequest,
    RoutedRagResponse,
    RouterSummary,
)


app = FastAPI(
    title="AI Ecosystem Intelligence A/B Router",
    description="Step 5 router for controlled model variant traffic splitting.",
    version="0.5.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ab-router"}


@app.get("/router/summary", response_model=RouterSummary)
def router_summary() -> RouterSummary:
    config = load_router_config()
    return RouterSummary(
        status="ready",
        model_a_weight=config.model_a_weight,
        model_b_weight=config.model_b_weight,
        log_path=str(config.request_log_path),
    )


@app.post("/rag-answer", response_model=RoutedRagResponse)
def routed_rag_answer(request: RoutedRagRequest) -> RoutedRagResponse:
    config = load_router_config()
    response_payload = route_request(
        request.question,
        request.top_k,
        request.user_id,
        request.request_id,
        config,
    )
    return RoutedRagResponse(**response_payload)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    config = load_router_config()
    log_row = build_feedback_log(
        request.request_id,
        request.user_id,
        request.rating,
        request.comment,
    )
    write_feedback_log(config.feedback_log_path, log_row)
    return FeedbackResponse(
        status="logged",
        request_id=request.request_id,
        rating=request.rating,
    )

