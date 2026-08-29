#!/usr/bin/env python3
"""
Build RAG-ready documents from Dataset 3 Stage 5 dashboard outputs.

This Step 2 script keeps the dashboard relevant by using the final Stage 5
CSV files as the first grounded RAG corpus.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.dashboard_loader import load_dashboard_bundle
from src.indexing.rag_document_builder import (
    build_rag_documents,
    write_document_manifest,
    write_jsonl_documents,
    write_markdown_documents,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create RAG documents from dashboard tables.")
    parser.add_argument("--dashboard-folder", default="data/dashboard")
    parser.add_argument("--output-folder", default="data/rag_documents")
    parser.add_argument("--run-tests", action="store_true")
    return parser.parse_args()


def build_documents_from_folders(dashboard_folder: Path, output_folder: Path) -> int:
    dashboard_bundle = load_dashboard_bundle(dashboard_folder)
    rag_documents = build_rag_documents(dashboard_bundle)
    write_outputs(output_folder, rag_documents)
    return len(rag_documents)


def write_outputs(output_folder: Path, rag_documents: list[object]) -> None:
    write_jsonl_documents(output_folder / "rag_documents.jsonl", rag_documents)
    write_markdown_documents(output_folder / "markdown", rag_documents)
    write_document_manifest(output_folder / "rag_document_manifest.json", rag_documents)


def write_sample_csv(
    file_path: Path,
    field_names: list[str],
    rows: list[dict[str, str]],
) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=field_names)
        csv_writer.writeheader()
        csv_writer.writerows(rows)


def create_sample_dashboard(folder_path: Path) -> None:
    # These tiny CSVs prove the pipeline works without needing the full dataset.
    sample_tables = build_sample_tables()
    for file_name, field_names, rows in sample_tables:
        write_sample_csv(folder_path / file_name, field_names, rows)


def build_sample_tables() -> list[tuple[str, list[str], list[dict[str, str]]]]:
    return [
        ("Dataset3_Dashboard_Top_Organisations.csv", org_fields(), org_rows()),
        ("Dataset3_Dashboard_Topic_Leadership.csv", topic_fields(), topic_rows()),
        ("Dataset3_Dashboard_Platform_Dominance.csv", platform_fields(), platform_rows()),
        ("Dataset3_Dashboard_Domain_Influence.csv", domain_fields(), domain_rows()),
        ("Dataset3_Dashboard_Cluster_Bridge.csv", cluster_fields(), cluster_rows()),
        ("Dataset3_Dashboard_KPI_Summary.csv", kpi_fields(), kpi_rows()),
        ("Dataset3_Dashboard_KG_Overview.csv", ["category", "count"], kg_rows()),
        ("Dataset3_Stage5_Dashboard_Audit.csv", ["metric_name", "status"], audit_rows()),
    ]


def org_fields() -> list[str]:
    return [
        "canonical_entity_id", "canonical_name", "influence_score",
        "influence_band", "primary_platform", "dominant_topic",
        "dominant_cluster", "dominant_domain", "artefact_count",
        "repository_count", "model_count", "dataset_count",
    ]


def org_rows() -> list[dict[str, str]]:
    return [sample_org_row()]


def sample_org_row() -> dict[str, str]:
    return {
        "canonical_entity_id": "CORG_001", "canonical_name": "microsoft",
        "influence_score": "79.2", "influence_band": "high",
        "primary_platform": "github", "dominant_topic": "Agentic Workflows",
        "dominant_cluster": "external system Tools", "dominant_domain": "AI Infrastructure",
        "artefact_count": "10", "repository_count": "7",
        "model_count": "2", "dataset_count": "1",
    }


def topic_fields() -> list[str]:
    return [
        "topic_node_id", "topic_display_label", "topic_label",
        "organisation_count", "artefact_count", "total_influence_score",
        "top_organisation_name", "top_organisation_id",
        "topic_interpretability_flag",
    ]


def topic_rows() -> list[dict[str, str]]:
    return [sample_topic_row()]


def sample_topic_row() -> dict[str, str]:
    return {
        "topic_node_id": "TOPIC_001", "topic_display_label": "Agentic Workflows",
        "topic_label": "Agentic Workflows", "organisation_count": "5",
        "artefact_count": "8", "total_influence_score": "120",
        "top_organisation_name": "microsoft", "top_organisation_id": "CORG_001",
        "topic_interpretability_flag": "interpretable_topic",
    }


def platform_fields() -> list[str]:
    return [
        "platform_node_id", "platform_display_label", "platform",
        "organisation_count", "artefact_count", "top_organisation_name",
        "top_organisation_id", "total_influence_score",
    ]


def platform_rows() -> list[dict[str, str]]:
    return [{
        "platform_node_id": "PLATFORM_001", "platform_display_label": "GitHub",
        "platform": "github", "organisation_count": "5", "artefact_count": "8",
        "top_organisation_name": "microsoft", "top_organisation_id": "CORG_001",
        "total_influence_score": "120",
    }]


def domain_fields() -> list[str]:
    return [
        "domain_node_id", "domain_display_label", "domain_label",
        "organisation_count", "total_influence_score",
        "top_organisation_name", "top_organisation_id",
    ]


def domain_rows() -> list[dict[str, str]]:
    return [{
        "domain_node_id": "DOMAIN_001", "domain_display_label": "AI Infrastructure",
        "domain_label": "AI Infrastructure", "organisation_count": "4",
        "total_influence_score": "100", "top_organisation_name": "microsoft",
        "top_organisation_id": "CORG_001",
    }]


def cluster_fields() -> list[str]:
    return [
        "cluster_node_id", "cluster_display_label", "cluster_label",
        "organisation_count", "artefact_count", "top_bridge_organisation_name",
        "top_influence_organisation_name",
    ]


def cluster_rows() -> list[dict[str, str]]:
    return [{
        "cluster_node_id": "CLUSTER_001", "cluster_display_label": "external system Tools",
        "cluster_label": "external system Tools", "organisation_count": "3",
        "artefact_count": "8", "top_bridge_organisation_name": "microsoft",
        "top_influence_organisation_name": "microsoft",
    }]


def kpi_fields() -> list[str]:
    return ["kpi_group", "kpi_name", "kpi_display_value", "interpretation"]


def kpi_rows() -> list[dict[str, str]]:
    return [{
        "kpi_group": "size", "kpi_name": "organisation_rows",
        "kpi_display_value": "4,283",
        "interpretation": "Organisation influence table size.",
    }]


def kg_rows() -> list[dict[str, str]]:
    return [{"category": "ORGANISATION", "count": "2"}]


def audit_rows() -> list[dict[str, str]]:
    return [{"metric_name": "failed", "status": "pass"}]


def run_tests() -> None:
    with tempfile.TemporaryDirectory() as temporary_folder:
        output_folder, document_text, document_count = run_sample_pipeline(temporary_folder)
        run_sunny_day_test(document_count)
        run_edge_case_test(output_folder)
        run_weird_gotcha_test(document_text)
        run_bug_catcher_test(document_text)


def run_sample_pipeline(temporary_folder: str) -> tuple[Path, str, int]:
    dashboard_folder = Path(temporary_folder) / "dashboard"
    output_folder = Path(temporary_folder) / "rag_documents"
    create_sample_dashboard(dashboard_folder)
    document_count = build_documents_from_folders(dashboard_folder, output_folder)
    document_text = (output_folder / "rag_documents.jsonl").read_text(encoding="utf-8")
    return output_folder, document_text, document_count


def run_sunny_day_test(document_count: int) -> None:
    # Test 1 (Sunny Day): Normal dashboard data must become RAG documents for the main path.
    assert document_count == 6
    print("✅ Sunny day test passed: dashboard rows became RAG documents.")


def run_edge_case_test(output_folder: Path) -> None:
    # Test 2 (Edge Case): Fresh output folders must be created automatically to avoid setup crashes.
    assert (output_folder / "rag_documents.jsonl").exists()
    print("✅ Edge case test passed: output files were created automatically.")


def run_weird_gotcha_test(document_text: str) -> None:
    # Test 3 (Weird Gotcha): Influence scores must remain visible because the dashboard is evidence.
    assert "79.2" in document_text
    print("✅ Weird gotcha test passed: influence evidence was preserved.")


def run_bug_catcher_test(document_text: str) -> None:
    # Test 4 (Bug Catcher): Organisation IDs must survive because missing IDs broke earlier outputs.
    assert "CORG_001" in document_text
    print("✅ Bug catcher test passed: organisation IDs were preserved.")


def main() -> None:
    arguments = parse_arguments()
    if arguments.run_tests:
        run_tests()
        return
    document_count = build_documents_from_folders(
        Path(arguments.dashboard_folder),
        Path(arguments.output_folder),
    )
    print(f"Created {document_count} RAG-ready documents.")


if __name__ == "__main__":
    main()



