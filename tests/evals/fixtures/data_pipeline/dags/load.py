"""Loads a day of events. Not idempotent: a rerun for the same day appends."""

import boto3
import pandas as pd
import psycopg

BUCKET = "events-archive"


def load_events(ds: str) -> None:
    frames = [
        pd.read_parquet(f"s3://{BUCKET}/dt={ds}/{obj['Key']}")
        for obj in boto3.client("s3").list_objects_v2(Bucket=BUCKET, Prefix=f"dt={ds}/")["Contents"]
    ]
    events = pd.concat(frames)
    with psycopg.connect("postgresql://warehouse") as conn:
        events.to_sql("events", conn, schema="raw", if_exists="append")
