"""
Local vector-style semantic index for Step 3.

This module intentionally uses only Python standard libraries so the project
can run before Docker, ChromaDB, or FAISS are introduced. The interface is
kept close to a vector database: documents are indexed, vectors are stored,
and semantic search returns similarity-scored chunks.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class VectorDocument:
    """A RAG document loaded from Step 2 JSONL output."""

    document_id: str
    document_type: str
    source_table: str
    title: str
    text: str
    metadata: dict[str, str]


@dataclass
class VectorRecord:
    """A stored vector entry with enough metadata for traceable retrieval."""

    document_id: str
    document_type: str
    source_table: str
    title: str
    text: str
    vector: dict[str, float]
    metadata: dict[str, str]


@dataclass
class VectorSearchResult:
    """One semantic search hit returned by the local vector index."""

    document_id: str
    document_type: str
    title: str
    score: float
    text: str
    metadata: dict[str, str]


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "with",
}


def clean_text(raw_value: object) -> str:
    if raw_value is None:
        return ""
    return " ".join(str(raw_value).replace("\n", " ").split()).strip()


def tokenize(text: str) -> list[str]:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def term_counts(tokens: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def load_rag_documents(file_path: Path) -> list[VectorDocument]:
    documents: list[VectorDocument] = []
    with file_path.open("r", encoding="utf-8") as input_file:
        for line_text in input_file:
            documents.append(parse_document_line(line_text))
    return documents


def parse_document_line(line_text: str) -> VectorDocument:
    try:
        row = json.loads(line_text)
    except json.JSONDecodeError as json_error:
        raise ValueError("RAG document JSONL contains an invalid line.") from json_error
    return VectorDocument(
        document_id=clean_text(row.get("document_id")),
        document_type=clean_text(row.get("document_type")),
        source_table=clean_text(row.get("source_table")),
        title=clean_text(row.get("title")),
        text=clean_text(row.get("text")),
        metadata=row.get("metadata", {}),
    )


def document_frequency(documents: list[VectorDocument]) -> dict[str, int]:
    frequency: dict[str, int] = {}
    for document in documents:
        unique_tokens = set(tokenize(document.text))
        update_document_frequency(frequency, unique_tokens)
    return frequency


def update_document_frequency(frequency: dict[str, int], tokens: set[str]) -> None:
    for token in tokens:
        frequency[token] = frequency.get(token, 0) + 1


def inverse_document_frequency(total_documents: int, document_count: int) -> float:
    return math.log((1 + total_documents) / (1 + document_count)) + 1


def normalise_vector(vector: dict[str, float]) -> dict[str, float]:
    magnitude = math.sqrt(sum(value * value for value in vector.values()))
    if magnitude == 0:
        return vector
    return {term: value / magnitude for term, value in vector.items()}


def build_tfidf_vector(
    text: str,
    frequencies: dict[str, int],
    total_documents: int,
) -> dict[str, float]:
    counts = term_counts(tokenize(text))
    weighted_vector = {}
    for token, count_value in counts.items():
        document_count = frequencies.get(token, 0)
        idf_value = inverse_document_frequency(total_documents, document_count)
        weighted_vector[token] = count_value * idf_value
    return normalise_vector(weighted_vector)


def build_vector_index(documents: list[VectorDocument]) -> list[VectorRecord]:
    frequencies = document_frequency(documents)
    total_documents = len(documents)
    return [build_vector_record(document, frequencies, total_documents) for document in documents]


def build_vector_record(
    document: VectorDocument,
    frequencies: dict[str, int],
    total_documents: int,
) -> VectorRecord:
    vector = build_tfidf_vector(document.text, frequencies, total_documents)
    return VectorRecord(
        document_id=document.document_id,
        document_type=document.document_type,
        source_table=document.source_table,
        title=document.title,
        text=document.text,
        vector=vector,
        metadata=document.metadata,
    )


def cosine_similarity(left_vector: dict[str, float], right_vector: dict[str, float]) -> float:
    shared_terms = set(left_vector).intersection(right_vector)
    return sum(left_vector[term] * right_vector[term] for term in shared_terms)


def search_vector_index(
    records: list[VectorRecord],
    query_text: str,
    top_k: int,
) -> list[VectorSearchResult]:
    query_vector = build_query_vector(records, query_text)
    scored_results = score_records(records, query_vector)
    return sorted(scored_results, key=lambda result: result.score, reverse=True)[:top_k]


def build_query_vector(records: list[VectorRecord], query_text: str) -> dict[str, float]:
    frequencies = rebuild_frequency_from_records(records)
    return build_tfidf_vector(query_text, frequencies, max(len(records), 1))


def rebuild_frequency_from_records(records: list[VectorRecord]) -> dict[str, int]:
    frequency: dict[str, int] = {}
    for record in records:
        for token in record.vector:
            frequency[token] = frequency.get(token, 0) + 1
    return frequency


def score_records(
    records: list[VectorRecord],
    query_vector: dict[str, float],
) -> list[VectorSearchResult]:
    return [score_single_record(record, query_vector) for record in records]


def score_single_record(
    record: VectorRecord,
    query_vector: dict[str, float],
) -> VectorSearchResult:
    return VectorSearchResult(
        document_id=record.document_id,
        document_type=record.document_type,
        title=record.title,
        score=cosine_similarity(query_vector, record.vector),
        text=record.text,
        metadata=record.metadata,
    )


def write_vector_index(file_path: Path, records: list[VectorRecord]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_vector_index(file_path: Path) -> list[VectorRecord]:
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    return [VectorRecord(**record) for record in payload]


def write_search_results(file_path: Path, results: list[VectorSearchResult]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_index_manifest(records: list[VectorRecord]) -> dict[str, object]:
    type_counts: dict[str, int] = {}
    for record in records:
        type_counts[record.document_type] = type_counts.get(record.document_type, 0) + 1
    return {"record_count": len(records), "document_type_counts": type_counts}


def write_index_manifest(file_path: Path, records: list[VectorRecord]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_index_manifest(records)
    file_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

