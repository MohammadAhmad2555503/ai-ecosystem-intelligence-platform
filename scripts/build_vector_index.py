#!/usr/bin/env python3
"""
Build and query the local semantic index.

This is Step 3 of the platform. It prepares the RAG documents for retrieval
before the model API is introduced in the next step.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.indexing.local_vector_index import (
    VectorDocument,
    build_vector_index,
    load_rag_documents,
    read_vector_index,
    search_vector_index,
    write_index_manifest,
    write_search_results,
    write_vector_index,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and search the local vector index.")
    parser.add_argument("--documents", default="data/rag_documents/rag_documents.jsonl")
    parser.add_argument("--index-path", default="data/vector_store/local_vector_index.json")
    parser.add_argument("--manifest-path", default="data/vector_store/vector_index_manifest.json")
    parser.add_argument("--query", default="Why is Microsoft influential in the AI ecosystem?")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--search-output", default="data/vector_store/sample_search_results.json")
    parser.add_argument("--run-tests", action="store_true")
    return parser.parse_args()


def build_index_from_documents(document_path: Path, index_path: Path, manifest_path: Path) -> int:
    documents = load_rag_documents(document_path)
    records = build_vector_index(documents)
    write_vector_index(index_path, records)
    write_index_manifest(manifest_path, records)
    return len(records)


def run_search(index_path: Path, query_text: str, top_k: int, output_path: Path) -> int:
    records = read_vector_index(index_path)
    results = search_vector_index(records, query_text, top_k)
    write_search_results(output_path, results)
    return len(results)


def create_sample_documents(file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sample_documents = build_sample_documents()
    with file_path.open("w", encoding="utf-8") as output_file:
        for document in sample_documents:
            output_file.write(json.dumps(document.__dict__) + "\n")


def build_sample_documents() -> list[VectorDocument]:
    return [sample_organisation_document(), sample_topic_document()]


def sample_organisation_document() -> VectorDocument:
    return VectorDocument(
        "organisation_microsoft",
        "organisation",
        "top_organisations",
        "microsoft",
        "Microsoft is highly influential in agentic AI workflows and GitHub tooling.",
        {"primary_id": "CORG_001"},
    )


def sample_topic_document() -> VectorDocument:
    return VectorDocument(
        "topic_agentic_workflows",
        "topic",
        "topic_leadership",
        "Agentic Workflows",
        "Agentic workflows contain organisations that build tool-using external system systems.",
        {"primary_id": "TOPIC_001"},
    )


def run_tests() -> None:
    with tempfile.TemporaryDirectory() as temporary_folder:
        output_folder = Path(temporary_folder)
        document_path = output_folder / "documents.jsonl"
        index_path = output_folder / "index.json"
        manifest_path = output_folder / "manifest.json"
        search_output = output_folder / "search.json"
        create_sample_documents(document_path)
        record_count = build_index_from_documents(document_path, index_path, manifest_path)
        result_count = run_search(index_path, "Microsoft GitHub influence", 1, search_output)
        search_text = search_output.read_text(encoding="utf-8")
        run_sunny_day_test(record_count)
        run_edge_case_test(index_path, manifest_path)
        run_weird_gotcha_test(result_count)
        run_bug_catcher_test(search_text)


def run_sunny_day_test(record_count: int) -> None:
    # Test 1 (Sunny Day): Normal documents must create vector records for retrieval.
    assert record_count == 2
    print("✅ Sunny day test passed: documents were indexed.")


def run_edge_case_test(index_path: Path, manifest_path: Path) -> None:
    # Test 2 (Edge Case): Index and manifest files must exist for downstream services.
    assert index_path.exists() and manifest_path.exists()
    print("✅ Edge case test passed: index and manifest files were written.")


def run_weird_gotcha_test(result_count: int) -> None:
    # Test 3 (Weird Gotcha): Search should return a result even before a full external system API exists.
    assert result_count == 1
    print("✅ Weird gotcha test passed: semantic search returned a result.")


def run_bug_catcher_test(search_text: str) -> None:
    # Test 4 (Bug Catcher): Retrieved results must preserve IDs for traceability.
    assert "CORG_001" in search_text
    print("✅ Bug catcher test passed: retrieved result kept the source ID.")


def run_default_pipeline(arguments: argparse.Namespace) -> tuple[int, int]:
    record_count = build_index_from_documents(
        Path(arguments.documents),
        Path(arguments.index_path),
        Path(arguments.manifest_path),
    )
    result_count = run_search_from_arguments(arguments)
    return record_count, result_count


def run_search_from_arguments(arguments: argparse.Namespace) -> int:
    return run_search(
        Path(arguments.index_path),
        arguments.query,
        arguments.top_k,
        Path(arguments.search_output),
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.run_tests:
        run_tests()
        return
    record_count, result_count = run_default_pipeline(arguments)
    print(f"Indexed {record_count} documents and wrote {result_count} search results.")


if __name__ == "__main__":
    main()



