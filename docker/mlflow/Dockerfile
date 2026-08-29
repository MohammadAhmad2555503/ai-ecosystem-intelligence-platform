FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /mlflow

RUN useradd --create-home --shell /bin/bash appuser

COPY docker/mlflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /mlflow/mlruns /mlflow/artifacts
RUN chown -R appuser:appuser /mlflow

USER appuser

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]

