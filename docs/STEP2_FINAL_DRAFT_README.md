# Step 2 Final Draft: Dashboard Loader and RAG Document Generator

This folder is the cleaned final draft for Step 2 of the AI Ecosystem Intelligence Platform.

## What Step 2 does

Step 2 keeps the Dataset 3 Stage 5 dashboard relevant by converting the final dashboard CSV files into RAG-ready documents.

The output documents will later be embedded into a vector database during Step 3.

## Language

Python 3.11

## Libraries

Step 2 uses only Python standard libraries.

## Main files

```text
src/dashboard/dashboard_loader.py
src/indexing/rag_document_builder.py
scripts/build_rag_documents.py
docs/step2_rag_document_generation.md
docs/step2_final_audit_report.json
```

## Inputs

The source-of-truth dashboard CSV files are stored in:

```text
data/dashboard/
```

## Outputs

```text
data/rag_documents/rag_documents.jsonl
data/rag_documents/rag_document_manifest.json
data/rag_documents/markdown/
```

## Run Step 2

```bash
python scripts/build_rag_documents.py
```

## Run Step 2 tests

```bash
python scripts/build_rag_documents.py --run-tests
```

## Expected verified result

```text
Created 166 RAG-ready documents.
```

Document breakdown:

```text
organisation: 100
topic: 25
platform: 11
domain: 10
cluster: 6
kpi: 14
```

## Why this matters

This step bridges the dashboard and the future RAG assistant. The RAG system will not answer from random text; it will answer from the same verified dashboard evidence used in the dissertation analytics.

