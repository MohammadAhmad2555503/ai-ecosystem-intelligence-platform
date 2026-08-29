"""
RAG inference engine for the Step 4 model API.

The first version is deterministic so it can later become the baseline model
for A/B testing against a stronger generative version.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from src.indexing.local_vector_index import (
    VectorRecord,
    VectorSearchResult,
    read_vector_index,
    search_vector_index,
)


MODEL_VERSION = "baseline-rag-v1"


@dataclass
class RagEngine:
    """Loads the local vector index and builds grounded answers from it."""

    records: list[VectorRecord]
    model_version: str = MODEL_VERSION

    @classmethod
    def from_index_path(cls, index_path: Path) -> "RagEngine":
        records = read_vector_index(index_path)
        return cls(records=records)

    def search_documents(self, query_text: str, top_k: int) -> list[VectorSearchResult]:
        safe_top_k = max(1, min(top_k, 10))
        return search_vector_index(self.records, query_text, safe_top_k)

    def answer_question(
        self,
        question: str,
        top_k: int,
    ) -> tuple[str, list[VectorSearchResult], float]:
        start_time = time.perf_counter()
        evidence = self.search_documents(question, top_k)
        answer_text = build_grounded_answer(question, evidence)
        latency_ms = (time.perf_counter() - start_time) * 1000
        return answer_text, evidence, latency_ms

    def document_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.document_type] = counts.get(record.document_type, 0) + 1
        return counts


def build_grounded_answer(question: str, evidence: list[VectorSearchResult]) -> str:
    useful_evidence = [result for result in evidence if result.score > 0]
    if not useful_evidence:
        return fallback_answer(question)
    return compose_answer(question, useful_evidence)


def fallback_answer(question: str) -> str:
    return (
        "I could not find strong matching evidence in the indexed dashboard corpus. "
        "The safe next step is to rephrase the question or refresh the vector index. "
        f"Question received: {question}"
    )


def compose_answer(question: str, evidence: list[VectorSearchResult]) -> str:
    lead_sentence = answer_lead(question, evidence[0])
    supporting_sentence = supporting_evidence_sentence(evidence)
    caution_sentence = "This answer is grounded only in the indexed Stage 5 dashboard outputs."
    return " ".join([lead_sentence, supporting_sentence, caution_sentence])


def answer_lead(question: str, top_result: VectorSearchResult) -> str:
    return (
        f"For the question '{question}', the strongest evidence comes from "
        f"{top_result.document_type} document '{top_result.title}'. "
        f"{top_result.text}"
    )


def supporting_evidence_sentence(evidence: list[VectorSearchResult]) -> str:
    titles = [result.title for result in evidence[:3]]
    joined_titles = ", ".join(titles)
    return f"Additional retrieved evidence came from: {joined_titles}."


def serialise_search_result(result: VectorSearchResult) -> dict[str, object]:
    return {
        "document_id": result.document_id,
        "document_type": result.document_type,
        "title": result.title,
        "score": round(float(result.score), 6),
        "text": result.text,
        "metadata": result.metadata,
    }

