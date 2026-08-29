"""
Configuration utilities for the A/B router.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RouterConfig:
    model_a_url: str
    model_b_url: str
    model_a_weight: float
    model_b_weight: float
    request_log_path: Path
    feedback_log_path: Path
    timeout_seconds: float


def read_float(name: str, default_value: float) -> float:
    try:
        return float(os.getenv(name, str(default_value)))
    except ValueError:
        return default_value


def normalise_weight(raw_weight: float) -> float:
    return max(0.0, min(raw_weight, 1.0))


def load_router_config() -> RouterConfig:
    model_a_weight = normalise_weight(read_float("MODEL_A_WEIGHT", 0.5))
    return RouterConfig(
        model_a_url=os.getenv("MODEL_A_URL", "http://localhost:8000/rag-answer"),
        model_b_url=os.getenv("MODEL_B_URL", "http://localhost:8001/rag-answer"),
        model_a_weight=model_a_weight,
        model_b_weight=round(1.0 - model_a_weight, 4),
        request_log_path=Path(read_path("REQUEST_LOG_PATH", "data/logs/ab_requests.jsonl")),
        feedback_log_path=Path(read_path("FEEDBACK_LOG_PATH", "data/logs/ab_feedback.jsonl")),
        timeout_seconds=read_float("ROUTER_TIMEOUT_SECONDS", 15.0),
    )


def read_path(name: str, default_value: str) -> str:
    return os.getenv(name, default_value)

