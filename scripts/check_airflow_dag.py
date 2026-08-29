#!/usr/bin/env python3
"""
Student-style checks for the Step 8 Airflow DAG.

This does not import Airflow directly, so it works even before Airflow is
installed on your machine.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DAG_PATH = PROJECT_ROOT / "airflow_dags" / "ai_ecosystem_pipeline_dag.py"


def read_dag_text() -> str:
    return DAG_PATH.read_text(encoding="utf-8")


def required_task_ids() -> list[str]:
    return [
        "validate_dashboard_inputs",
        "build_rag_documents",
        "build_vector_index",
        "test_model_api_logic",
        "test_ab_router_logic",
        "analyse_ab_test",
        "write_pipeline_audit",
    ]


def run_tests() -> None:
    dag_text = read_dag_text()
    run_sunny_day_test(dag_text)
    run_edge_case_test()
    run_weird_gotcha_test(dag_text)
    run_bug_catcher_test(dag_text)


def run_sunny_day_test(dag_text: str) -> None:
    # Test 1: The expected DAG ID should be present.
    assert "ai_ecosystem_intelligence_pipeline" in dag_text
    print("✅ Sunny day test passed: Airflow DAG ID is present.")


def run_edge_case_test() -> None:
    # Test 2: The DAG file must exist for Airflow discovery.
    assert DAG_PATH.exists()
    print("✅ Edge case test passed: Airflow DAG file exists.")


def run_weird_gotcha_test(dag_text: str) -> None:
    # Test 3: Kubernetes support should exist for later production deployment.
    assert "KubernetesPodOperator" in dag_text
    print("✅ Weird gotcha test passed: Kubernetes execution support is present.")


def run_bug_catcher_test(dag_text: str) -> None:
    # Test 4: All orchestration task IDs must appear in the DAG.
    assert all(task_id in dag_text for task_id in required_task_ids())
    print("✅ Bug catcher test passed: all pipeline task IDs are present.")


if __name__ == "__main__":
    run_tests()

