"""
airflow/dags/ingestion_dag.py
Daily ingestion DAG — runs the Phase-1 pipeline automatically.

Schedule: every day at 02:00 UTC
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ── Default task arguments ────────────────────────────────────────────────────
default_args = {
    "owner": "ragflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="ragflow_ingestion",
    default_args=default_args,
    description="Daily ingestion of PDFs, Confluence pages, and Git repos.",
    schedule_interval="0 2 * * *",   # 02:00 UTC every day
    start_date=days_ago(1),
    catchup=False,
    tags=["ragflow", "ingestion", "phase1"],
) as dag:

    # ── Task helpers ──────────────────────────────────────────────────────────

    def _ingest_pdfs(**context) -> dict:
        from ingestion.run import run_ingestion
        summary = run_ingestion(skip_existing=True)
        context["task_instance"].xcom_push(key="summary", value=summary)
        return summary

    def _validate_results(**context) -> None:
        """Fail the DAG if zero documents were ingested across all connectors."""
        ti = context["task_instance"]
        summary = ti.xcom_pull(task_ids="ingest_all_sources", key="summary") or {}

        total_docs = sum(v.get("documents", 0) for v in summary.values())
        total_chunks = sum(v.get("chunks", 0) for v in summary.values())

        print(f"Ingestion summary: {summary}")
        print(f"Total documents: {total_docs} | Total chunks: {total_chunks}")

        if total_docs == 0:
            raise ValueError(
                "No new documents were ingested. "
                "Check connector configuration or source availability."
            )

    # ── Tasks ─────────────────────────────────────────────────────────────────

    ingest_task = PythonOperator(
        task_id="ingest_all_sources",
        python_callable=_ingest_pdfs,
        provide_context=True,
    )

    validate_task = PythonOperator(
        task_id="validate_results",
        python_callable=_validate_results,
        provide_context=True,
    )

    # ── Dependencies ──────────────────────────────────────────────────────────
    ingest_task >> validate_task
