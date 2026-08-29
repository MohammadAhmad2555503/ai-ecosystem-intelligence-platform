#!/usr/bin/env python3
"""
Student-style tests for the Step 4 RAG engine.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.rag_engine import RagEngine
from src.indexing.local_vector_index import (
    VectorDocument,
    build_vector_index,
    write_vector_index,
)


def sample_records_path(folder_path: Path) -> Path:
    index_path = folder_path / "index.json"
    records = build_vector_index(sample_documents())
    write_vector_index(index_path, records)
    return index_path


def sample_documents() -> list[VectorDocument]:
    return [microsoft_document(), topic_document()]


def microsoft_document() -> VectorDocument:
    return VectorDocument(
        "organisation_microsoft",
        "organisation",
        "top_organisations",
        "microsoft",
        "Microsoft is influential in the AI ecosystem through GitHub and agentic workflows.",
        {"primary_id": "CORG_001"},
    )


def topic_document() -> VectorDocument:
    return VectorDocument(
        "topic_agentic_workflows",
        "topic",
        "topic_leadership",
        "Agentic Workflows",
        "Agentic workflows connect tool using external system systems and orchestration projects.",
        {"primary_id": "TOPIC_001"},
    )


def run_tests() -> None:
    with tempfile.TemporaryDirectory() as temporary_folder:
        index_path = sample_records_path(Path(temporary_folder))
        engine = RagEngine.from_index_path(index_path)
        results = engine.search_documents("Microsoft GitHub influence", 1)
        answer, evidence, latency_ms = engine.answer_question("Why Microsoft?", 2)
        summary_counts = engine.document_type_counts()
        run_sunny_day_test(results)
        run_edge_case_test(summary_counts)
        run_weird_gotcha_test(answer)
        run_bug_catcher_test(evidence, latency_ms)


def run_sunny_day_test(results: list[object]) -> None:
    # Test 1 (Sunny Day): Normal queries should retrieve the Microsoft document.
    assert results and results[0].document_id == "organisation_microsoft"
    print("✅ Sunny day test passed: search retrieved the expected organisation.")


def run_edge_case_test(summary_counts: dict[str, int]) -> None:
    # Test 2 (Edge Case): Summary counts should work even with a tiny two-document index.
    assert summary_counts["organisation"] == 1
    print("✅ Edge case test passed: dashboard-style summary counts worked.")


def run_weird_gotcha_test(answer: str) -> None:
    # Test 3 (Weird Gotcha): Answers must mention the dashboard evidence boundary.
    assert "Stage 5 dashboard outputs" in answer
    print("✅ Weird gotcha test passed: answer stayed grounded to dashboard evidence.")


def run_bug_catcher_test(evidence: list[object], latency_ms: float) -> None:
    # Test 4 (Bug Catcher): Evidence must preserve source IDs for future A/B logging.
    evidence_json = json.dumps([result.metadata for result in evidence])
    assert "CORG_001" in evidence_json and latency_ms >= 0
    print("✅ Bug catcher test passed: evidence IDs and latency were preserved.")


if __name__ == "__main__":
    run_tests()



