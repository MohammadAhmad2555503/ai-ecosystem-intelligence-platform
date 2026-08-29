FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY docker/model-api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app/src
COPY scripts /app/scripts
COPY data/vector_store /app/data/vector_store

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["python", "scripts/run_model_api.py"]

