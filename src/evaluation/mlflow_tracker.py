"""
Optional MLflow tracking for Step 6.

If MLflow is not installed, the code writes a local JSON fallback. This keeps
the project runnable on one machine while still documenting the MLflow path.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.evaluation.ab_statistics import AbTestResult


def result_to_metrics(result: AbTestResult) -> dict[str, float]:
    return {
        "baseline_success_rate": result.baseline.success_rate,
        "challenger_success_rate": result.challenger.success_rate,
        "absolute_lift": result.absolute_lift,
        "relative_lift": result.relative_lift,
        "p_value": result.p_value,
        "effect_size": result.effect_size,
        "observed_power": result.observed_power,
    }


def result_to_params(result: AbTestResult) -> dict[str, Any]:
    return {
        "primary_metric": result.primary_metric,
        "baseline_model": result.baseline.model_name,
        "challenger_model": result.challenger.model_name,
        "required_sample_per_variant": result.required_sample_per_variant,
        "decision": result.decision,
    }


def log_ab_result_to_mlflow(
    result: AbTestResult,
    fallback_path: Path,
    experiment_name: str,
) -> str:
    try:
        import mlflow
    except ImportError:
        return write_local_tracking_fallback(result, fallback_path, experiment_name)
    return log_with_mlflow(result, experiment_name)


def log_with_mlflow(result: AbTestResult, experiment_name: str) -> str:
    import mlflow

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="ab-test-analysis") as active_run:
        mlflow.log_params(result_to_params(result))
        mlflow.log_metrics(result_to_metrics(result))
        mlflow.set_tag("decision", result.decision)
        return str(active_run.info.run_id)


def write_local_tracking_fallback(
    result: AbTestResult,
    fallback_path: Path,
    experiment_name: str,
) -> str:
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tracking_backend": "local_json_fallback",
        "experiment_name": experiment_name,
        "params": result_to_params(result),
        "metrics": result_to_metrics(result),
        "result": asdict(result),
    }
    fallback_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return "local-json-fallback"

