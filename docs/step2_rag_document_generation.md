# Step 2: Dashboard Loader and RAG Document Generator

Step 2 keeps the Stage 5 dashboard outputs relevant by turning them into the first RAG-ready document corpus.

## Language

Python 3.11.

## Libraries

Only Python standard libraries are used in Step 2.

## Why this step exists

The dashboard tables are excellent for structured analytics, but a RAG system needs readable text chunks. This step converts each important dashboard row into a retrieval-ready document.

## Inputs

Place the final Stage 5 files in:

```text
data/dashboard/
```

Required CSV files:

```text
Dataset3_Dashboard_KPI_Summary.csv
Dataset3_Dashboard_Top_Organisations.csv
Dataset3_Dashboard_Topic_Leadership.csv
Dataset3_Dashboard_Platform_Dominance.csv
Dataset3_Dashboard_Domain_Influence.csv
Dataset3_Dashboard_Cluster_Bridge.csv
Dataset3_Dashboard_KG_Overview.csv
Dataset3_Stage5_Dashboard_Audit.csv
```

## Outputs

```text
data/rag_documents/rag_documents.jsonl
data/rag_documents/rag_document_manifest.json
data/rag_documents/markdown/*.md
```

## Run

```bash
python scripts/build_rag_documents.py
```

## Run tests

```bash
python scripts/build_rag_documents.py --run-tests
```

## Why the dashboard is still central

The RAG documents are generated from the dashboard tables. That means the future RAG assistant will answer questions using the same influence scores, topic leadership, platform dominance, domain influence, and cluster bridge outputs produced by Dataset 3 Stage 5.

