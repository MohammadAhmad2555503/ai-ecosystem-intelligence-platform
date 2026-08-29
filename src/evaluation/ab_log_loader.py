"""
Load and join A/B router prediction and feedback logs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.ab_statistics import VariantSummary


@dataclass
class PredictionRecord:
    request_id: str
    user_id: str
    selected_model: str
    question: str
    answer: str
    latency_ms: float
    evidence_count: int


@dataclass
class FeedbackRecord:
    request_id: str
    user_id: str
    rating: int
    comment: str


@dataclass
class JoinedAbRecord:
    request_id: str
    selected_model: str
    rating: int
    is_positive: bool
    latency_ms: float
    evidence_count: int


def load_jsonl_records(file_path: Path) -> list[dict[str, object]]:
    if not file_path.exists():
        return []
    rows = []
    with file_path.open("r", encoding="utf-8") as input_file:
        for line_text in input_file:
            rows.append(json.loads(line_text))
    return rows


def parse_predictions(file_path: Path) -> list[PredictionRecord]:
    return [prediction_from_row(row) for row in load_jsonl_records(file_path)]


def prediction_from_row(row: dict[str, object]) -> PredictionRecord:
    return PredictionRecord(
        str(row.get("request_id", "")),
        str(row.get("user_id", "")),
        str(row.get("selected_model", "")),
        str(row.get("question", "")),
        str(row.get("answer", "")),
        float(row.get("latency_ms", 0.0)),
        int(row.get("evidence_count", 0)),
    )


def parse_feedback(file_path: Path) -> list[FeedbackRecord]:
    return [feedback_from_row(row) for row in load_jsonl_records(file_path)]


def feedback_from_row(row: dict[str, object]) -> FeedbackRecord:
    return FeedbackRecord(
        str(row.get("request_id", "")),
        str(row.get("user_id", "")),
        int(row.get("rating", 0)),
        str(row.get("comment", "")),
    )


def latest_feedback_by_request(feedback_rows: list[FeedbackRecord]) -> dict[str, FeedbackRecord]:
    feedback_lookup = {}
    for feedback_row in feedback_rows:
        feedback_lookup[feedback_row.request_id] = feedback_row
    return feedback_lookup


def join_prediction_feedback(
    predictions: list[PredictionRecord],
    feedback_rows: list[FeedbackRecord],
) -> list[JoinedAbRecord]:
    feedback_lookup = latest_feedback_by_request(feedback_rows)
    joined_rows = []
    for prediction in predictions:
        joined_row = join_single_prediction(prediction, feedback_lookup)
        if joined_row:
            joined_rows.append(joined_row)
    return joined_rows


def join_single_prediction(
    prediction: PredictionRecord,
    feedback_lookup: dict[str, FeedbackRecord],
) -> JoinedAbRecord | None:
    feedback = feedback_lookup.get(prediction.request_id)
    if feedback is None:
        return None
    return JoinedAbRecord(
        prediction.request_id,
        prediction.selected_model,
        feedback.rating,
        feedback.rating >= 4,
        prediction.latency_ms,
        prediction.evidence_count,
    )


def summarise_variant(rows: list[JoinedAbRecord], model_name: str) -> VariantSummary:
    model_rows = [row for row in rows if row.selected_model == model_name]
    success_count = sum(1 for row in model_rows if row.is_positive)
    sample_size = len(model_rows)
    success_rate = success_count / sample_size if sample_size else 0.0
    return VariantSummary(model_name, sample_size, success_count, success_rate)


def build_variant_summaries(rows: list[JoinedAbRecord]) -> tuple[VariantSummary, VariantSummary]:
    return summarise_variant(rows, "model_a"), summarise_variant(rows, "model_b")

