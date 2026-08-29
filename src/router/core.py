"""
Core A/B routing logic.

The routing decision is deterministic by request ID. This avoids a common
experiment problem where retries accidentally jump between variants.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.router.config import RouterConfig


@dataclass
class RouterDecision:
    request_id: str
    selected_model: str
    selected_url: str


@dataclass
class PredictionLog:
    timestamp_utc: str
    request_id: str
    user_id: str
    selected_model: str
    question: str
    answer: str
    latency_ms: float
    evidence_count: int
    status: str


@dataclass
class FeedbackLog:
    timestamp_utc: str
    request_id: str
    user_id: str
    rating: int
    comment: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_request_id(request_id: str | None) -> str:
    return request_id if request_id else str(uuid4())


def hash_to_unit_interval(value_text: str) -> float:
    digest = hashlib.sha256(value_text.encode("utf-8")).hexdigest()
    integer_value = int(digest[:12], 16)
    return integer_value / float(0xFFFFFFFFFFFF)


def choose_model_version(request_id: str, config: RouterConfig) -> RouterDecision:
    bucket_value = hash_to_unit_interval(request_id)
    if bucket_value < config.model_a_weight:
        return RouterDecision(request_id, "model_a", config.model_a_url)
    return RouterDecision(request_id, "model_b", config.model_b_url)


def call_model_api(
    url: str,
    question: str,
    top_k: int,
    timeout_seconds: float,
) -> dict[str, object]:
    payload = json.dumps({"question": question, "top_k": top_k}).encode("utf-8")
    request = build_http_request(url, payload)
    try:
        return read_http_response(request, timeout_seconds)
    except urllib.error.URLError as url_error:
        return fallback_model_response(question, str(url_error))


def build_http_request(url: str, payload: bytes) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def read_http_response(
    request: urllib.request.Request,
    timeout_seconds: float,
) -> dict[str, object]:
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_text = response.read().decode("utf-8")
    return json.loads(response_text)


def fallback_model_response(question: str, error_text: str) -> dict[str, object]:
    return {
        "answer": "Model API unavailable. This fallback keeps the router testable.",
        "question": question,
        "evidence": [],
        "evidence_count": 0,
        "latency_ms": 0.0,
        "error": error_text,
    }


def route_request(
    question: str,
    top_k: int,
    user_id: str,
    request_id: str | None,
    config: RouterConfig,
) -> dict[str, object]:
    decision = choose_model_version(stable_request_id(request_id), config)
    model_response, latency_ms = fetch_variant_answer(decision, question, top_k, config)
    response_payload = build_router_response(decision, user_id, model_response, latency_ms)
    write_prediction_log(config.request_log_path, prediction_log(response_payload, question))
    return response_payload


def fetch_variant_answer(
    decision: RouterDecision,
    question: str,
    top_k: int,
    config: RouterConfig,
) -> tuple[dict[str, object], float]:
    started_at = time.perf_counter()
    model_response = call_model_api(
        decision.selected_url,
        question,
        top_k,
        config.timeout_seconds,
    )
    return model_response, (time.perf_counter() - started_at) * 1000


def build_router_response(
    decision: RouterDecision,
    user_id: str,
    model_response: dict[str, object],
    latency_ms: float,
) -> dict[str, object]:
    evidence = safe_evidence(model_response.get("evidence", []))
    response = base_router_response(decision, user_id, model_response, latency_ms)
    response["evidence_count"] = len(evidence)
    response["evidence"] = evidence
    return response


def base_router_response(
    decision: RouterDecision,
    user_id: str,
    model_response: dict[str, object],
    latency_ms: float,
) -> dict[str, object]:
    return {
        "request_id": decision.request_id,
        "user_id": user_id,
        "selected_model": decision.selected_model,
        "answer": str(model_response.get("answer", "")),
        "latency_ms": round(latency_ms, 3),
    }


def safe_evidence(raw_evidence: object) -> list[dict[str, object]]:
    if isinstance(raw_evidence, list):
        return [item for item in raw_evidence if isinstance(item, dict)]
    return []


def prediction_log(response_payload: dict[str, object], question: str) -> PredictionLog:
    return PredictionLog(
        timestamp_utc=utc_now(),
        request_id=str(response_payload["request_id"]),
        user_id=str(response_payload["user_id"]),
        selected_model=str(response_payload["selected_model"]),
        question=question,
        answer=str(response_payload["answer"]),
        latency_ms=float(response_payload["latency_ms"]),
        evidence_count=int(response_payload["evidence_count"]),
        status="ok",
    )


def write_prediction_log(file_path: Path, log_row: PredictionLog) -> None:
    append_jsonl(file_path, asdict(log_row))


def write_feedback_log(file_path: Path, log_row: FeedbackLog) -> None:
    append_jsonl(file_path, asdict(log_row))


def append_jsonl(file_path: Path, row: dict[str, object]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_feedback_log(
    request_id: str,
    user_id: str,
    rating: int,
    comment: str,
) -> FeedbackLog:
    return FeedbackLog(utc_now(), request_id, user_id, rating, comment)

