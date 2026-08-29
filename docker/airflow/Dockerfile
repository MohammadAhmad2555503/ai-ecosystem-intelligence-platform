FROM apache/airflow:2.10.5-python3.11

USER root

RUN mkdir -p /opt/airflow/project
RUN chown -R airflow:0 /opt/airflow

USER airflow

COPY docker/airflow/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

