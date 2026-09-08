"""Daily load. Reads yesterday's event files from S3, overwrites `raw.events`."""

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipeline.load import load_events

with DAG(
    "events_daily",
    schedule="@daily",
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
) as dag:
    PythonOperator(task_id="load_events", python_callable=load_events)
