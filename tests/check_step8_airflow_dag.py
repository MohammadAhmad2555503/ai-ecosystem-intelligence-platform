#!/usr/bin/env python3
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


def test_dag_file_exists() -> None:
    assert DAG_PATH.exists()
    print("PASS: DAG file exists.")


def test_dag_id_present(dag_text: str) -> None:
    assert "ai_ecosystem_intelligence_pipeline" in dag_text
    print("PASS: DAG ID is present.")


def test_kubernetes_support_present(dag_text: str) -> None:
    assert "KubernetesPodOperator" in dag_text
    print("PASS: Kubernetes support is present.")


def test_required_tasks_present(dag_text: str) -> None:
    assert all(task_id in dag_text for task_id in required_task_ids())
    print("PASS: all required task IDs are present.")


def run_tests() -> None:
    test_dag_file_exists()
    dag_text = read_dag_text()
    test_dag_id_present(dag_text)
    test_kubernetes_support_present(dag_text)
    test_required_tasks_present(dag_text)


if __name__ == "__main__":
    run_tests()

