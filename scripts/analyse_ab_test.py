#!/usr/bin/env python3
"""
Analyse A/B router logs and log the result.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.ab_log_loader import (
    build_variant_summaries,
    join_prediction_feedback,
    parse_feedback,
    parse_predictions,
)
from src.evaluation.ab_statistics import run_two_proportion_test
from src.evaluation.mlflow_tracker import log_ab_result_to_mlflow


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse A/B test logs.")
    parser.add_argument("--request-log", default="data/logs/ab_requests.jsonl")
    parser.add_argument("--feedback-log", default="data/logs/ab_feedback.jsonl")
    parser.add_argument("--output-json", default="data/evaluation/ab_test_results.json")
    parser.add_argument("--output-report", default="reports/ab_test_decision_report.md")
    parser.add_argument("--mlflow-fallback", default="data/mlflow/ab_test_tracking_fallback.json")
    parser.add_argument("--experiment-name", default="ai-ecosystem-ab-tests")
    parser.add_argument("--run-tests", action="store_true")
    return parser.parse_args()


def analyse_logs(arguments: argparse.Namespace) -> dict[str, object]:
    predictions = parse_predictions(Path(arguments.request_log))
    feedback_rows = parse_feedback(Path(arguments.feedback_log))
    joined_rows = join_prediction_feedback(predictions, feedback_rows)
    baseline, challenger = build_variant_summaries(joined_rows)
    result = run_two_proportion_test(baseline, challenger)
    tracking_id = track_result(arguments, result)
    payload = build_result_payload(result, len(joined_rows), tracking_id)
    write_outputs(payload, Path(arguments.output_json), Path(arguments.output_report))
    return payload


def track_result(arguments: argparse.Namespace, result: object) -> str:
    return log_ab_result_to_mlflow(
        result,
        Path(arguments.mlflow_fallback),
        arguments.experiment_name,
    )


def build_result_payload(result: object, joined_count: int, tracking_id: str) -> dict[str, object]:
    payload = asdict(result)
    payload["joined_feedback_rows"] = joined_count
    payload["tracking_run_id"] = tracking_id
    return payload


def write_outputs(payload: dict[str, object], json_path: Path, report_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(markdown_report(payload), encoding="utf-8")


def markdown_report(payload: dict[str, object]) -> str:
    return "\n".join(report_lines(payload))


def report_lines(payload: dict[str, object]) -> list[str]:
    return [
        "# A/B Test Decision Report",
        "",
        f"Primary metric: {payload['primary_metric']}",
        f"Decision: **{payload['decision']}**",
        f"P-value: {payload['p_value']:.6f}",
        f"Absolute lift: {payload['absolute_lift']:.4f}",
        f"Relative lift: {payload['relative_lift']:.4f}",
        f"Observed power: {payload['observed_power']:.4f}",
        f"Required sample per variant: {payload['required_sample_per_variant']}",
        "",
        "Shipping requires significance, positive lift, and adequate power.",
    ]


def write_demo_logs(folder_path: Path) -> tuple[Path, Path]:
    request_log = folder_path / "ab_requests.jsonl"
    feedback_log = folder_path / "ab_feedback.jsonl"
    write_demo_jsonl(request_log, demo_prediction_rows())
    write_demo_jsonl(feedback_log, demo_feedback_rows())
    return request_log, feedback_log


def write_demo_jsonl(file_path: Path, rows: list[dict[str, object]]) -> None:
    with file_path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")


def demo_prediction_rows() -> list[dict[str, object]]:
    rows = []
    rows.extend(prediction_rows_for_model("model_a", 10))
    rows.extend(prediction_rows_for_model("model_b", 10))
    return rows


def prediction_rows_for_model(model_name: str, count: int) -> list[dict[str, object]]:
    return [prediction_row(model_name, row_number) for row_number in range(count)]


def prediction_row(model_name: str, row_number: int) -> dict[str, object]:
    return {
        "request_id": f"{model_name}_{row_number}",
        "user_id": "demo",
        "selected_model": model_name,
        "question": "Why is Microsoft influential?",
        "answer": "demo answer",
        "latency_ms": 10.0,
        "evidence_count": 3,
    }


def demo_feedback_rows() -> list[dict[str, object]]:
    rows = []
    rows.extend(feedback_rows_for_model("model_a", 6, 4))
    rows.extend(feedback_rows_for_model("model_b", 8, 2))
    return rows


def feedback_rows_for_model(
    model_name: str,
    positive_count: int,
    negative_count: int,
) -> list[dict[str, object]]:
    positive_rows = build_feedback_rows(model_name, 0, positive_count, 5)
    negative_rows = build_feedback_rows(model_name, positive_count, negative_count, 2)
    return positive_rows + negative_rows


def build_feedback_rows(
    model_name: str,
    start_number: int,
    count: int,
    rating: int,
) -> list[dict[str, object]]:
    return [
        feedback_row(model_name, start_number + row_number, rating)
        for row_number in range(count)
    ]


def feedback_row(model_name: str, row_number: int, rating: int) -> dict[str, object]:
    return {
        "request_id": f"{model_name}_{row_number}",
        "user_id": "demo",
        "rating": rating,
        "comment": "",
    }


def run_tests() -> None:
    with tempfile.TemporaryDirectory() as temporary_folder:
        payload, folder_path = run_demo_analysis(Path(temporary_folder))
        run_sunny_day_test(payload)
        run_edge_case_test(folder_path)
        run_weird_gotcha_test(payload)
        run_bug_catcher_test(payload)


def run_demo_analysis(folder_path: Path) -> tuple[dict[str, object], Path]:
    request_log, feedback_log = write_demo_logs(folder_path)
    arguments = sample_arguments(folder_path, request_log, feedback_log)
    return analyse_logs(arguments), folder_path


def sample_arguments(
    folder_path: Path,
    request_log: Path,
    feedback_log: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        request_log=str(request_log),
        feedback_log=str(feedback_log),
        output_json=str(folder_path / "ab_results.json"),
        output_report=str(folder_path / "ab_report.md"),
        mlflow_fallback=str(folder_path / "mlflow_fallback.json"),
        experiment_name="demo-ab-test",
        run_tests=False,
    )


def run_sunny_day_test(payload: dict[str, object]) -> None:
    # Test 1 (Sunny Day): Normal A/B logs should produce a decision payload.
    assert payload["primary_metric"] == "positive_feedback_rate"
    print("✅ Sunny day test passed: A/B result payload was created.")


def run_edge_case_test(folder_path: Path) -> None:
    # Test 2 (Edge Case): Local MLflow fallback should exist when MLflow is unavailable.
    assert (folder_path / "mlflow_fallback.json").exists()
    print("✅ Edge case test passed: local MLflow fallback was written.")


def run_weird_gotcha_test(payload: dict[str, object]) -> None:
    # Test 3 (Weird Gotcha): P-value must be present because lift alone is not enough.
    assert "p_value" in payload
    print("✅ Weird gotcha test passed: statistical p-value was calculated.")


def run_bug_catcher_test(payload: dict[str, object]) -> None:
    # Test 4 (Bug Catcher): Joined feedback rows must be counted for auditability.
    assert payload["joined_feedback_rows"] == 20
    print("✅ Bug catcher test passed: joined feedback count was preserved.")


def main() -> None:
    arguments = parse_arguments()
    if arguments.run_tests:
        run_tests()
        return
    payload = analyse_logs(arguments)
    print(f"A/B analysis complete. Decision: {payload['decision']}")


if __name__ == "__main__":
    main()

