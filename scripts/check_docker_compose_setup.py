#!/usr/bin/env python3
"""
Local smoke checks for Step 7 Docker Compose preparation.

The script does not require Docker to be installed. It checks that the
compose file and Dockerfiles needed for local containerisation are present.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def required_files() -> list[Path]:
    return [
        PROJECT_ROOT / "docker-compose.yml",
        PROJECT_ROOT / "docker/model-api/Dockerfile",
        PROJECT_ROOT / "docker/ab-router/Dockerfile",
        PROJECT_ROOT / "docker/data-indexer/Dockerfile",
        PROJECT_ROOT / "docker/mlflow/Dockerfile",
        PROJECT_ROOT / "docker/model-api/requirements.txt",
        PROJECT_ROOT / "docker/ab-router/requirements.txt",
        PROJECT_ROOT / "docker/mlflow/requirements.txt",
    ]


def missing_files() -> list[Path]:
    return [file_path for file_path in required_files() if not file_path.exists()]


def read_compose_text() -> str:
    return (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def contains_required_services(compose_text: str) -> bool:
    service_names = ["data-indexer", "model-api", "ab-router", "mlflow"]
    return all(service_name in compose_text for service_name in service_names)


def run_tests() -> None:
    compose_text = read_compose_text()
    run_sunny_day_test(compose_text)
    run_edge_case_test()
    run_weird_gotcha_test(compose_text)
    run_bug_catcher_test(compose_text)


def run_sunny_day_test(compose_text: str) -> None:
    # Test 1 (Sunny Day): Compose must define the four local services.
    assert contains_required_services(compose_text)
    print("✅ Sunny day test passed: Docker Compose services are present.")


def run_edge_case_test() -> None:
    # Test 2 (Edge Case): Missing Dockerfiles would break a fresh demo machine.
    assert not missing_files()
    print("✅ Edge case test passed: all required Docker files exist.")


def run_weird_gotcha_test(compose_text: str) -> None:
    # Test 3 (Weird Gotcha): The model API must wait for the indexer to finish.
    assert "service_completed_successfully" in compose_text
    print("✅ Weird gotcha test passed: model API waits for indexer completion.")


def run_bug_catcher_test(compose_text: str) -> None:
    # Test 4 (Bug Catcher): Logs must be mounted so A/B evidence survives restarts.
    assert "./data:/app/data" in compose_text
    print("✅ Bug catcher test passed: shared data volume is mounted.")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--run-tests":
        run_tests()
        return
    run_tests()


if __name__ == "__main__":
    main()

