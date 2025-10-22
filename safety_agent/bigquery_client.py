import os
from typing import List, Dict, Any
from google.cloud import bigquery


class BigQueryClient:
    def __init__(self, project: str = None, dataset: str = None):
        self.project = project or os.getenv("SAFETY_GCP_PROJECT_ID")
        self.dataset = dataset or os.getenv("BIGQUERY_DATASET")
        if not self.dataset:
            raise ValueError("BIGQUERY_DATASET environment variable must be set")
        self.client = bigquery.Client(project=self.project)

    def table_ref(self, table_name: str):
        return f"{self.project}.{self.dataset}.{table_name}"

    def insert_rows(self, table_name: str, rows: List[Dict[str, Any]]):
        table_id = self.table_ref(table_name)
        errors = self.client.insert_rows_json(table_id, rows)
        if errors:
            raise RuntimeError(f"BigQuery insert errors: {errors}")

    def run_query(self, query: str):
        job = self.client.query(query)
        return [dict(row) for row in job.result()]

    def get_ingestion_aggregation(self, table_name: str):
        # Group by file_name and aggregate file records
        table_id = self.table_ref(table_name)
        q = f"SELECT file_name, ARRAY_AGG(STRUCT(file_name AS file_name, file_type AS file_type, file_size AS file_size, source AS source, last_modified AS last_modified, ingested_at AS ingested_at, is_present AS is_present)) AS file_records FROM `{table_id}` GROUP BY file_name"
        return self.run_query(q)


class BigQueryByteStore:
    """Minimal byte store wrapper backed by BigQuery. Intended as a light
    replacement for a MongoDBByteStore used in the project. This provides a
    simple API the repo expects: init and insert/get operations.
    """
    def __init__(self, table_name: str = None):
        self.table = table_name or os.getenv("BIGQUERY_BYTESTORE_TABLE")
        self.bq = BigQueryClient()

    def add_documents(self, documents: List[Dict[str, Any]]):
        # documents are dictionaries representing rows; ensure required fields
        self.bq.insert_rows(self.table, documents)

    def find_by_id(self, doc_id: str):
        table_id = self.bq.table_ref(self.table)
        q = f"SELECT * FROM `{table_id}` WHERE doc_id = @doc_id LIMIT 1"
        job = self.bq.client.query(q, job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("doc_id", "STRING", doc_id)]
        ))
        rows = [dict(r) for r in job.result()]
        return rows[0] if rows else None
