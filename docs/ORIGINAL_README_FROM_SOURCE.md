# AI Ecosystem Intelligence Platform

## Final Project Status

This repository contains the completed **AI Ecosystem Intelligence Platform** developed for the MSc Artificial Intelligence dissertation project.

The platform combines:

- Research literature analysis
- GitHub and Hugging Face ecosystem data
- Organisation-level knowledge graph construction
- Influence scoring and graph analytics
- Traditional machine learning in R
- Deep learning in Python
- ML vs DL comparison
- R Shiny research dashboard
- FastAPI application backend
- Retrieval-Augmented Generation, also known as RAG
- A/B routing and feedback logging
- MLflow experiment-tracking support
- Airflow DAG validation
- Docker Compose deployment
- React and TypeScript frontend

The final completed application folder is:

```text
ai_ecosystem_intelligence_platform_step7_DOCKER_COMPOSE_FINAL/ai-ecosystem-intelligence-platform
```

---

## Project Purpose

The purpose of this project is to build an end-to-end AI ecosystem intelligence system that can analyse how modern AI technologies, organisations, open-source artefacts, and research topics are connected.

The project investigates questions such as:

- Which organisations are most influential in the AI ecosystem?
- How are organisations connected to GitHub repositories and Hugging Face artefacts?
- Which topics, platforms, and domains dominate the ecosystem?
- Can graph-derived features predict high-influence organisations?
- How do traditional machine learning models compare with deep learning models on this task?
- Can the final knowledge base be exposed through a practical RAG-based application?

The project is divided into two main layers:

1. **Research Layer**  
   Contains the academic data preparation, machine learning, deep learning, model comparison, and R Shiny dashboard.

2. **Application Layer**  
   Contains the production-style platform, including FastAPI services, RAG, A/B router, Docker Compose, MLflow, Airflow validation, and React frontend.

---

## Final Verified Dataset Statistics

The final project package includes three main datasets.

### Dataset 1: Research Literature Dataset

Dataset 1 contains AI research literature records.

Final verified files include:

```text
data_main/raw/Dataset1_Raw_Public_Final.csv
data_main/final/Dataset1_Final_Visualisation_Ready.csv
```

Final verified statistics:

```text
Raw research records: 10,192
Final visualisation-ready research records: 9,052
```

Dataset 1 covers AI research areas such as:

- external systems
- Retrieval-Augmented Generation
- Hallucination
- Reasoning
- Agents
- Evaluation
- Multimodal AI
- Machine learning and deep learning research trends

---

### Dataset 2: GitHub and Hugging Face Ecosystem Dataset

Dataset 2 contains open-source AI ecosystem artefacts collected from GitHub and Hugging Face.

Final verified files include:

```text
data_main/raw/Dataset2_Raw_Ecosystem.csv
data_main/processed/Dataset2_Final_Visualisation_Ready.csv
```

Final verified statistics:

```text
Raw ecosystem records: 1,885
Final visualisation-ready ecosystem records: 1,885
```

Dataset 2 includes information such as:

- GitHub repositories
- Hugging Face models
- Hugging Face datasets
- Stars
- Forks
- Likes
- Downloads
- Licences
- Platform information
- Topic and cluster features

---

### Dataset 3: Organisation Knowledge Graph Dataset

Dataset 3 is the main organisation-level intelligence dataset.

Final verified files include:

```text
data_main/raw/Dataset3_Raw_Organisations.csv
data_main/raw/Dataset3_Raw_Organisation_Artefact_Links.csv
data_main/processed/Dataset3_Canonical_Organisations.csv
data_main/processed/Dataset3_Canonical_Organisation_Artefact_Links.csv
data_main/graph/Dataset3_KG_Nodes.csv
data_main/graph/Dataset3_KG_Edges.csv
data_main/analytics/Dataset3_Organisation_Influence_Scores.csv
```

Final verified statistics:

```text
Raw organisation records: 6,836
Raw organisation-artefact links: 7,883
Canonical organisations: 4,283
Canonical organisation-artefact links: 7,119
Knowledge graph nodes: 10,900
Knowledge graph edges: 49,418
Organisation influence rows: 4,283
Topic leadership rows: 25
Platform dominance rows: 11
Domain influence rows: 10
Cluster bridge rows: 6
```

The knowledge graph links organisations to:

- AI artefacts
- GitHub repositories
- Hugging Face models
- Hugging Face datasets
- Topics
- Platforms
- Domains
- Influence scores
- Graph centrality measures

---

## Organisation Influence Results

The project calculates organisation influence using graph-derived and ecosystem-level features.

Influence bands in the final dataset:

```text
High influence organisations: 2
Medium influence organisations: 50
Emerging influence organisations: 1,039
Low influence organisations: 3,192
```

Top organisations identified in the final influence analysis include:

```text
microsoft
NVIDIA
google
research lab
meta-llama
agentscope-ai
deepseek-ai
OpenBMB
jinaai
NousResearch
```

---

## Research Layer

The research layer is located in:

```text
research_layer/
```

Important files include:

```text
research_layer/02_machine_learning_models.R
research_layer/03_deep_learning_model.py
research_layer/04.R
research_layer/05_prepare_shiny_dashboard_data.R
research_layer/outputs/
research_layer/shiny_dashboard/
```

The research task is a binary classification problem:

```text
Predict whether an organisation belongs to the medium/high influence group.
```

The target variable is based on organisation influence level.

The dataset is highly imbalanced, so the evaluation does not rely only on accuracy. The project uses:

- Precision
- Recall
- F1-score
- Specificity
- Balanced accuracy
- AUC
- Confusion matrix

---

## Traditional Machine Learning Methods

Traditional machine learning was implemented in R.

The following models were trained:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine
5. Naive Bayes
6. Majority-class baseline

Main ML output files:

```text
research_layer/outputs/dataset3_ml_all_model_metrics.csv
research_layer/outputs/dataset3_ml_all_predictions.csv
research_layer/outputs/dataset3_ml_feature_importance.csv
research_layer/outputs/dataset3_ml_model_comparison_report.csv
research_layer/outputs/dataset3_ml_selected_features.csv
research_layer/outputs/dataset3_ml_trained_models.rds
```

Best traditional machine learning model:

```text
Random Forest
```

Random Forest performance:

```text
Accuracy: 99.81%
Precision: 92.31%
Recall: 92.31%
F1-score: 92.31%
AUC: 99.96%
```

---

## Deep Learning Methods

Deep learning was implemented in Python.

The following neural network models were trained:

1. Shallow Neural Network
2. Deep Neural Network
3. Wide Neural Network
4. Narrow Deep Neural Network
5. Dropout Neural Network
6. Majority-class baseline

Main DL output files:

```text
research_layer/outputs/dataset3_dl_all_model_metrics.csv
research_layer/outputs/dataset3_dl_all_predictions.csv
research_layer/outputs/dataset3_dl_training_history.csv
research_layer/outputs/dataset3_dl_selected_features.csv
research_layer/outputs/dataset3_dl_model_comparison_report.csv
research_layer/outputs/dataset3_dl_trained_models.json
```

Best deep learning models:

```text
Shallow Neural Network
Wide Neural Network
```

Best deep learning performance:

```text
F1-score: 80.00%
```

The Wide Neural Network achieved the strongest deep learning AUC:

```text
AUC: 99.93%
```

---

## ML vs DL Comparison

The ML and DL results were combined and compared.

Main comparison output files:

```text
research_layer/outputs/dataset3_combined_model_metrics.csv
research_layer/outputs/dataset3_overall_model_ranking.csv
research_layer/outputs/dataset3_ml_vs_dl_family_summary.csv
research_layer/outputs/dataset3_best_model_summary.csv
research_layer/outputs/dataset3_step4_comparison_report.csv
research_layer/outputs/dataset3_combined_model_predictions.csv
```

Final best overall model:

```text
Machine Learning - Random Forest
```

Best model summary:

```text
Best model: Machine Learning - Random Forest
F1-score: 92.31%
AUC: 99.96%
```

Family-level comparison:

```text
Best ML F1-score: 92.31%
Best DL F1-score: 80.00%
Mean ML F1-score: 61.04%
Mean DL F1-score: 77.23%
```

Interpretation:

```text
Random Forest achieved the strongest individual model performance.
Deep learning models showed stronger average consistency across architectures.
```

---

## R Shiny Research Dashboard

The R Shiny dashboard is located in:

```text
research_layer/shiny_dashboard/
```

Important files:

```text
research_layer/shiny_dashboard/app.R
research_layer/shiny_dashboard/dashboard_data/
```

The Shiny dashboard visualises the academic research results, including:

- Dashboard KPIs
- ML vs DL comparison
- Model ranking
- F1-score and AUC comparison
- Confusion matrix summaries
- Prediction distributions
- Feature importance
- Top organisations by influence

To run the Shiny dashboard in R or RStudio:

```r
setwd("C:/Users/HP/Desktop/New folder (5)/New folder (2)/AI Ecosystem/ai_ecosystem_intelligence_platform_step7_DOCKER_COMPOSE_FINAL/ai-ecosystem-intelligence-platform/research_layer")

shiny::runApp("shiny_dashboard")
```

---

## Application Layer

The application layer is the production-style system.

It includes:

- FastAPI Model API
- A/B Router
- RAG document generator
- Local vector-style semantic index
- Dashboard summary endpoint
- Semantic search endpoint
- RAG answer endpoint
- Feedback endpoint
- Docker Compose deployment
- MLflow service
- Airflow DAG validation
- React frontend

Important folders:

```text
src/
scripts/
docker/
data/
data/dashboard/
data/rag_documents/
data/vector_store/
reports/evidence/
frontend/
airflow_dags/
```

---

## RAG System

RAG means **Retrieval-Augmented Generation**.

In this project, the RAG system allows the user to ask natural-language questions about the AI ecosystem, organisation influence, knowledge graph outputs, topics, platforms, domains, and dashboard results.

The RAG process is:

1. The user asks a question.
2. The system searches the indexed knowledge base.
3. Relevant records are retrieved from the local vector-style index.
4. The retrieved evidence is used to construct a grounded answer.
5. The answer is returned with source information.

The live application RAG is generated from the final structured Dataset 3 dashboard and knowledge graph outputs.

Important RAG files:

```text
data/rag_documents/rag_documents.jsonl
data/rag_documents/rag_document_manifest.json
data/rag_documents/markdown/
data/vector_store/local_vector_index.json
data/vector_store/vector_index_manifest.json
```

Final verified RAG index size:

```text
RAG documents: 166
Vector index records: 166
```

Example questions:

```text
Which organisations are most influential in the AI ecosystem?
What are the main knowledge graph statistics?
Which topics dominate the AI ecosystem?
Which platforms are most important?
Which organisations have high influence scores?
```

Note:

```text
Dataset 1 research literature files are included in the project package for research-literature analysis. The current Docker RAG demonstrator indexes the generated structured knowledge graph and dashboard documents. Full PDF-level paper chunking and paper-specific summarisation can be added as a future extension if full paper text is indexed.
```

---

## Backend Services

The Docker Compose stack contains the following services:

```text
data-indexer
model-api
ab-router
mlflow
```

Main service URLs:

```text
Model API Swagger: http://localhost:8000/docs
A/B Router Swagger: http://localhost:8080/docs
MLflow: http://localhost:5000
React Frontend: http://localhost:5173
```

---

## FastAPI Model API

The Model API provides the main inference and analytics service.

Available endpoints:

```text
GET  /health
GET  /ready
GET  /dashboard/summary
POST /semantic-search
POST /rag-answer
```

The health endpoint should return:

```json
{
  "status": "ok",
  "service": "model-api"
}
```

---

## A/B Router

The A/B router provides production-style request routing and feedback logging.

Available endpoints:

```text
GET  /health
GET  /router/summary
POST /rag-answer
POST /feedback
```

The health endpoint should return:

```json
{
  "status": "ok",
  "service": "ab-router"
}
```

The router can send RAG requests to configured model variants and log feedback for later analysis.

---

## MLflow

MLflow is included as an experiment-tracking and evaluation-support service.

The MLflow service runs at:

```text
http://localhost:5000
```

MLflow-related files are located in:

```text
data/mlflow/
docker/mlflow/
src/evaluation/mlflow_tracker.py
```

---

## Airflow DAG Validation

The project includes Airflow DAG validation files.

Important files:

```text
airflow_dags/ai_ecosystem_pipeline_dag.py
scripts/check_airflow_dag.py
tests/check_step8_airflow_dag.py
reports/evidence/step8_airflow_dag_check.txt
```

The DAG represents how the full AI ecosystem intelligence pipeline could be orchestrated in a production environment.

---

## React Frontend

The frontend is located in:

```text
frontend/
```

It is built with:

```text
React
TypeScript
Vite
CSS
```

The frontend provides:

- Dark AI research assistant interface
- Sidebar navigation
- RAG chat panel
- Backend health indicators
- Source cards
- Knowledge base overview
- A/B router toggle
- Live connection to Model API and A/B Router through Vite proxy

Important frontend files:

```text
frontend/src/App.tsx
frontend/src/App.css
frontend/src/index.css
frontend/vite.config.ts
frontend/package.json
```

The Vite proxy maps:

```text
/api/model  -> http://localhost:8000
/api/router -> http://localhost:8080
```

This avoids browser CORS problems during local development.

---

## Evidence Files

The project includes application evidence in:

```text
reports/evidence/
```

Evidence files include:

```text
ab_router_feedback.json
ab_router_health.json
ab_router_openapi.json
ab_router_rag_answer.json
ab_router_summary.json
dashboard_summary.json
docker_compose_services.txt
model_api_health.json
model_api_openapi.json
model_api_rag_answer.json
step8_airflow_dag_check.txt
```

These files demonstrate that the backend services, RAG endpoints, A/B router, Docker services, and Airflow validation were tested.

---

## How to Run the Application

### 1. Open PowerShell

Go to the final application folder:

```powershell
cd "[private workspace path]"
```

### 2. Start Docker Compose

```powershell
docker compose up -d
```

### 3. Check Running Containers

```powershell
docker compose ps
```

Expected services:

```text
ai-ecosystem-model-api
ai-ecosystem-ab-router
ai-ecosystem-mlflow
```

### 4. Check Model API Health

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected result:

```text
status service
------ -------
ok     model-api
```

### 5. Check A/B Router Health

```powershell
Invoke-RestMethod http://localhost:8080/health
```

Expected result:

```text
status service
------ -------
ok     ab-router
```

### 6. Open Backend Documentation

Open these in a browser:

```text
http://localhost:8000/docs
http://localhost:8080/docs
http://localhost:5000
```

---

## Test RAG Through PowerShell

Test the A/B router RAG endpoint:

```powershell
$body = @{
  question = "Which organisations are most influential in the AI ecosystem?"
  top_k = 3
  user_id = "demo-user"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://localhost:8080/rag-answer" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Test the Model API RAG endpoint:

```powershell
$body = @{
  question = "Which organisations are most influential in the AI ecosystem?"
  top_k = 3
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://localhost:8000/rag-answer" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

---

## Run the React Frontend

Open a second PowerShell terminal.

Go to the frontend folder:

```powershell
cd "[private workspace path]"
```

Install dependencies if needed:

```powershell
npm.cmd install
```

Start the frontend:

```powershell
npm.cmd run dev
```

Open the frontend in a browser:

```text
http://localhost:5173
```

If PowerShell blocks `npm`, use `npm.cmd` as shown above.

---

## Stop the Application

From the main application folder:

```powershell
docker compose down
```

---

## Manual Script Commands

The project also includes scripts for rebuilding or testing individual parts of the application layer.

Run from the main application folder:

```powershell
python scripts/build_rag_documents.py
python scripts/build_vector_index.py
python scripts/test_model_api_logic.py
python scripts/test_ab_router_logic.py
python scripts/analyse_ab_test.py --run-tests
python scripts/check_docker_compose_setup.py --run-tests
python scripts/check_airflow_dag.py
```

---

## Important Project Folders

```text
data/
```

Contains application-layer data used by the Docker services, including dashboard files, RAG documents, vector index, logs, evaluation files, and MLflow data.

```text
data_main/
```

Contains the main final dataset outputs for Dataset 1, Dataset 2, and Dataset 3, including raw, processed, graph, analytics, and dashboard outputs.

```text
research_layer/
```

Contains R and Python research scripts, ML/DL outputs, and the R Shiny dashboard.

```text
frontend/
```

Contains the React and TypeScript frontend.

```text
src/
```

Contains reusable backend source code for dashboard loading, indexing, inference, routing, and evaluation.

```text
scripts/
```

Contains runnable utility scripts for RAG generation, vector indexing, API testing, A/B analysis, Docker checks, and Airflow DAG checks.

```text
docker/
```

Contains Dockerfiles and service-level requirements for the data indexer, model API, A/B router, MLflow, and Airflow.

```text
reports/evidence/
```

Contains JSON and text evidence generated from testing the application layer.

---

## Submission Notes

For dissertation submission, this repository should be submitted together with:

```text
Final dissertation report PDF
Final dissertation report DOCX, if required
Screenshot evidence folder
```

Recommended screenshot evidence:

```text
Docker Compose running
Model API Swagger page
A/B Router Swagger page
Model API health check
A/B Router health check
React frontend
RAG answer output
A/B router RAG answer output
R Shiny dashboard
MLflow page
Airflow DAG check output
```

If a clean submission zip is required, `frontend/node_modules/` can be removed because it can be regenerated using:

```powershell
cd frontend
npm.cmd install
```

---

## Limitations

The project has the following limitations:

- The organisation influence classification task is highly imbalanced.
- The deployed RAG demonstrator indexes structured Dataset 3 dashboard and knowledge graph evidence rather than full PDF paper chunks.
- Dataset 1 research literature files are included, but full paper-level RAG requires additional PDF text extraction and chunk indexing.
- The frontend is a local prototype interface, not a cloud-hosted multi-user system.
- Docker deployment is local rather than deployed to a public cloud provider.
- Influence scores depend on the available public data sources and entity-resolution quality.

---

## Final Summary

This project is a full AI Ecosystem Intelligence Platform that collects and analyses AI research and ecosystem data, builds an organisation knowledge graph, calculates influence scores, compares traditional machine learning and deep learning models, visualises results through an R Shiny research dashboard, and exposes the final intelligence layer through a Dockerised FastAPI RAG application with an A/B router, MLflow support, Airflow DAG validation, and a React frontend.

The project demonstrates both academic AI research skills and practical full-stack AI system development.





