#!/usr/bin/env python3
"""
Student-style tests for the Step 5 A/B router logic.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.router.config import RouterConfig
from src.router.core import (
    build_feedback_log,
    choose_model_version,
    route_request,
    write_feedback_log,
)


def test_config(folder_path: Path) -> RouterConfig:
    return RouterConfig(
        model_a_url="http://localhost:9999/rag-answer",
        model_b_url="http://localhost:9998/rag-answer",
        model_a_weight=1.0,
        model_b_weight=0.0,
        request_log_path=folder_path / "ab_requests.jsonl",
        feedback_log_path=folder_path / "ab_feedback.jsonl",
        timeout_seconds=0.01,
    )


def run_tests() -> None:
    with tempfile.TemporaryDirectory() as temporary_folder:
        folder_path = Path(temporary_folder)
        config = test_config(folder_path)
        response = route_request("Why is Microsoft influential?", 3, "student", "REQ_001", config)
        decision = choose_model_version("REQ_001", config)
        feedback_row = build_feedback_log("REQ_001", "student", 5, "useful")
        write_feedback_log(config.feedback_log_path, feedback_row)
        request_log_text = config.request_log_path.read_text(encoding="utf-8")
        feedback_text = config.feedback_log_path.read_text(encoding="utf-8")
        run_sunny_day_test(response)
        run_edge_case_test(decision.selected_model)
        run_weird_gotcha_test(request_log_text)
        run_bug_catcher_test(feedback_text)


def run_sunny_day_test(response: dict[str, object]) -> None:
    # Test 1 (Sunny Day): A normal request should return a routed response.
    assert response["request_id"] == "REQ_001"
    print("✅ Sunny day test passed: request was routed and returned.")


def run_edge_case_test(selected_model: str) -> None:
    # Test 2 (Edge Case): Weight 1.0 should send all traffic to model A.
    assert selected_model == "model_a"
    print("✅ Edge case test passed: 100 percent model A weight worked.")


def run_weird_gotcha_test(request_log_text: str) -> None:
    # Test 3 (Weird Gotcha): Prediction logs must keep the selected model for analysis.
    assert "model_a" in request_log_text
    print("✅ Weird gotcha test passed: selected model was logged.")


def run_bug_catcher_test(feedback_text: str) -> None:
    # Test 4 (Bug Catcher): Feedback logs must preserve request IDs for A/B joining.
    feedback_rows = [json.loads(line) for line in feedback_text.splitlines()]
    assert feedback_rows[0]["request_id"] == "REQ_001"
    print("✅ Bug catcher test passed: feedback kept the request ID.")


if __name__ == "__main__":
    run_tests()

