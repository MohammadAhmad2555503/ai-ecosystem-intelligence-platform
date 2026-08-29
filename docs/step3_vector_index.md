# Step 3: Local Vector Index and Semantic Search

Step 3 converts the Step 2 RAG-ready documents into a searchable local vector-style index.

## Language

Python 3.11

## Libraries

The default Step 3 implementation uses only Python standard libraries.

## Why not ChromaDB yet?

ChromaDB will be introduced later when the Docker service layer is built. For this step, the priority is to create a fully runnable local semantic retrieval layer on one machine without requiring heavy dependencies.

## Inputs

```text
data/rag_documents/rag_documents.jsonl
```

## Outputs

```text
data/vector_store/local_vector_index.json
data/vector_store/vector_index_manifest.json
data/vector_store/sample_search_results.json
```

## Run Step 3

```bash
python scripts/build_vector_index.py
```

## Run Step 3 tests

```bash
python scripts/build_vector_index.py --run-tests
```

## What this proves

This step proves that the dashboard-derived RAG documents can be retrieved semantically before the external system API is added.

The index preserves:

```text
document_id
document_type
source_table
title
text
metadata
similarity score
```

That means future RAG answers can cite the exact retrieved evidence instead of giving ungrounded responses.



