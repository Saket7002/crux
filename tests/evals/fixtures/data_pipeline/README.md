# data_pipeline
Airflow DAGs loading events from S3 into a Postgres warehouse. Daily schedule,
full-table overwrite on every run, dbt-style SQL models under `sql/`.
