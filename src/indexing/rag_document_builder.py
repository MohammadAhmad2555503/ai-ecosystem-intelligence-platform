"""
RAG document generation for the AI Ecosystem Intelligence Platform.

The dashboard tables are strong for analytics, but RAG needs explainable
text chunks. This module converts Stage 5 rows into grounded documents that
can later be embedded into a vector database.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.dashboard.dashboard_loader import DashboardBundle, clean_text


@dataclass
class RagDocument:
    """One retrieval-ready document created from a dashboard row."""

    document_id: str
    document_type: str
    source_table: str
    title: str
    text: str
    metadata: dict[str, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_value(row: dict[str, str], column_name: str) -> str:
    return clean_text(row.get(column_name, ""))


def make_document_id(prefix: str, raw_identifier: str, row_number: int) -> str:
    cleaned_identifier = clean_text(raw_identifier).replace(" ", "_")
    safe_identifier = cleaned_identifier or str(row_number)
    return f"{prefix}_{safe_identifier}".lower()


def build_rag_documents(bundle: DashboardBundle) -> list[RagDocument]:
    documents = []
    documents.extend(build_organisation_documents(bundle))
    documents.extend(build_topic_documents(bundle))
    documents.extend(build_platform_documents(bundle))
    documents.extend(build_domain_documents(bundle))
    documents.extend(build_cluster_documents(bundle))
    documents.extend(build_kpi_documents(bundle))
    return documents


def build_organisation_documents(bundle: DashboardBundle) -> list[RagDocument]:
    rows = bundle.get_rows("top_organisations")
    return [organisation_document(row, row_number) for row_number, row in enumerate(rows, start=1)]


def organisation_document(row: dict[str, str], row_number: int) -> RagDocument:
    organisation_name = safe_value(row, "canonical_name")
    organisation_id = safe_value(row, "canonical_entity_id")
    text = organisation_text(row, organisation_name, organisation_id)
    return make_document(
        "organisation",
        "top_organisations",
        organisation_id,
        organisation_name,
        text,
        row,
    )


def organisation_text(row: dict[str, str], organisation_name: str, organisation_id: str) -> str:
    return (
        f"{organisation_name} is an AI ecosystem organisation with ID {organisation_id}. "
        f"It has influence score {safe_value(row, 'influence_score')} and influence band "
        f"{safe_value(row, 'influence_band')}. Its primary platform is "
        f"{safe_value(row, 'primary_platform')}. The dominant topic is "
        f"{safe_value(row, 'dominant_topic')}, dominant cluster is "
        f"{safe_value(row, 'dominant_cluster')}, and dominant domain is "
        f"{safe_value(row, 'dominant_domain')}. It is linked to "
        f"{safe_value(row, 'artefact_count')} artefacts, including "
        f"{safe_value(row, 'repository_count')} repositories, "
        f"{safe_value(row, 'model_count')} models, and "
        f"{safe_value(row, 'dataset_count')} datasets."
    )


def build_topic_documents(bundle: DashboardBundle) -> list[RagDocument]:
    rows = bundle.get_rows("topic_leadership")
    return [topic_document(row, row_number) for row_number, row in enumerate(rows, start=1)]


def topic_document(row: dict[str, str], row_number: int) -> RagDocument:
    topic_name = safe_value(row, "topic_display_label") or safe_value(row, "topic_label")
    topic_id = safe_value(row, "topic_node_id")
    text = topic_text(row, topic_name)
    return make_document("topic", "topic_leadership", topic_id, topic_name, text, row)


def topic_text(row: dict[str, str], topic_name: str) -> str:
    return (
        f"{topic_name} is a topic in the AI ecosystem graph. It has "
        f"{safe_value(row, 'organisation_count')} organisations and "
        f"{safe_value(row, 'artefact_count')} artefacts. The total influence score is "
        f"{safe_value(row, 'total_influence_score')}. The leading organisation is "
        f"{safe_value(row, 'top_organisation_name')} with ID "
        f"{safe_value(row, 'top_organisation_id')}. Interpretability flag: "
        f"{safe_value(row, 'topic_interpretability_flag')}."
    )


def build_platform_documents(bundle: DashboardBundle) -> list[RagDocument]:
    rows = bundle.get_rows("platform_dominance")
    return [platform_document(row, row_number) for row_number, row in enumerate(rows, start=1)]


def platform_document(row: dict[str, str], row_number: int) -> RagDocument:
    platform_name = safe_value(row, "platform_display_label") or safe_value(row, "platform")
    platform_id = safe_value(row, "platform_node_id")
    text = platform_text(row, platform_name)
    return make_document("platform", "platform_dominance", platform_id, platform_name, text, row)


def platform_text(row: dict[str, str], platform_name: str) -> str:
    return (
        f"{platform_name} is a platform represented in the AI ecosystem graph. It has "
        f"{safe_value(row, 'organisation_count')} organisations and "
        f"{safe_value(row, 'artefact_count')} artefacts. The top organisation is "
        f"{safe_value(row, 'top_organisation_name')} with ID "
        f"{safe_value(row, 'top_organisation_id')}. Total influence score is "
        f"{safe_value(row, 'total_influence_score')}."
    )


def build_domain_documents(bundle: DashboardBundle) -> list[RagDocument]:
    rows = bundle.get_rows("domain_influence")
    return [domain_document(row, row_number) for row_number, row in enumerate(rows, start=1)]


def domain_document(row: dict[str, str], row_number: int) -> RagDocument:
    domain_name = safe_value(row, "domain_display_label") or safe_value(row, "domain_label")
    domain_id = safe_value(row, "domain_node_id")
    text = domain_text(row, domain_name)
    return make_document("domain", "domain_influence", domain_id, domain_name, text, row)


def domain_text(row: dict[str, str], domain_name: str) -> str:
    return (
        f"{domain_name} is a domain-level grouping in the AI ecosystem graph. It contains "
        f"{safe_value(row, 'organisation_count')} organisations and has total influence "
        f"score {safe_value(row, 'total_influence_score')}. The leading organisation is "
        f"{safe_value(row, 'top_organisation_name')} with ID "
        f"{safe_value(row, 'top_organisation_id')}."
    )


def build_cluster_documents(bundle: DashboardBundle) -> list[RagDocument]:
    rows = bundle.get_rows("cluster_bridge")
    return [cluster_document(row, row_number) for row_number, row in enumerate(rows, start=1)]


def cluster_document(row: dict[str, str], row_number: int) -> RagDocument:
    cluster_name = safe_value(row, "cluster_display_label") or safe_value(row, "cluster_label")
    cluster_id = safe_value(row, "cluster_node_id")
    text = cluster_text(row, cluster_name)
    return make_document("cluster", "cluster_bridge", cluster_id, cluster_name, text, row)


def cluster_text(row: dict[str, str], cluster_name: str) -> str:
    return (
        f"{cluster_name} is a cluster in the AI ecosystem graph. It contains "
        f"{safe_value(row, 'organisation_count')} organisations and "
        f"{safe_value(row, 'artefact_count')} artefacts. The strongest bridge organisation is "
        f"{safe_value(row, 'top_bridge_organisation_name')}. The highest influence organisation is "
        f"{safe_value(row, 'top_influence_organisation_name')}."
    )


def build_kpi_documents(bundle: DashboardBundle) -> list[RagDocument]:
    rows = bundle.get_rows("kpi_summary")
    return [kpi_document(row, row_number) for row_number, row in enumerate(rows, start=1)]


def kpi_document(row: dict[str, str], row_number: int) -> RagDocument:
    kpi_name = safe_value(row, "kpi_name")
    text = kpi_text(row)
    return make_document("kpi", "kpi_summary", kpi_name, kpi_name, text, row)


def kpi_text(row: dict[str, str]) -> str:
    return (
        f"The KPI {safe_value(row, 'kpi_name')} belongs to the "
        f"{safe_value(row, 'kpi_group')} group. Its value is "
        f"{safe_value(row, 'kpi_display_value')}. Interpretation: "
        f"{safe_value(row, 'interpretation')}."
    )


def make_document(
    document_type: str,
    source_table: str,
    raw_identifier: str,
    title: str,
    text: str,
    row: dict[str, str],
) -> RagDocument:
    metadata = build_metadata(document_type, source_table, row)
    document_id = make_document_id(document_type, raw_identifier, len(text))
    return RagDocument(document_id, document_type, source_table, title, text, metadata)


def build_metadata(
    document_type: str,
    source_table: str,
    row: dict[str, str],
) -> dict[str, str]:
    return {
        "document_type": document_type,
        "source_table": source_table,
        "created_at_utc": utc_now(),
        "primary_id": first_existing_value(row),
    }


def first_existing_value(row: dict[str, str]) -> str:
    possible_columns = ["canonical_entity_id", "topic_node_id", "platform_node_id"]
    possible_columns += ["domain_node_id", "cluster_node_id", "kpi_name"]
    for column_name in possible_columns:
        if safe_value(row, column_name):
            return safe_value(row, column_name)
    return ""


def write_jsonl_documents(file_path: Path, documents: list[RagDocument]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as output_file:
        for document in documents:
            output_file.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


def write_markdown_documents(folder_path: Path, documents: list[RagDocument]) -> None:
    folder_path.mkdir(parents=True, exist_ok=True)
    for document in documents:
        file_path = folder_path / f"{document.document_id}.md"
        file_path.write_text(markdown_document(document), encoding="utf-8")


def markdown_document(document: RagDocument) -> str:
    return (
        f"# {document.title}\n\n"
        f"Document type: {document.document_type}\n\n"
        f"Source table: {document.source_table}\n\n"
        f"{document.text}\n"
    )


def write_document_manifest(file_path: Path, documents: list[RagDocument]) -> None:
    manifest = build_manifest(documents)
    file_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def build_manifest(documents: list[RagDocument]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for document in documents:
        counts[document.document_type] = counts.get(document.document_type, 0) + 1
    return {"document_count": len(documents), "document_type_counts": counts}

