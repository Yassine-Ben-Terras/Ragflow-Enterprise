"""
airflow/dags/feedback_loop_dag.py
Daily DAG that processes negative feedback and runs RAGAs evaluation.

Schedule: every day at 03:00 UTC (after ingestion at 02:00)
"""

from __future__ import annotations

from datetime import timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "ragflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ragflow_feedback_loop",
    default_args=default_args,
    description="Daily feedback loop: process thumbs-down, run RAGAs eval, write improvement report.",
    schedule_interval="0 3 * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["ragflow", "monitoring", "phase5"],
) as dag:

    def _run_feedback_loop(**context):
        from monitoring.feedback.feedback_loop import run_feedback_loop
        summary = run_feedback_loop()
        context["task_instance"].xcom_push(key="summary", value=summary)
        return summary

    def _log_summary(**context):
        ti = context["task_instance"]
        summary = ti.xcom_pull(task_ids="process_feedback", key="summary") or {}
        print(f"Feedback loop summary: {summary}")
        if summary.get("retrieval_failures", 0) > 5:
            print("⚠️  High retrieval failure count — consider re-tuning chunking or retrieval params.")
        if summary.get("prompt_failures", 0) > 5:
            print("⚠️  High prompt failure count — consider revising the system prompt.")

    process_task = PythonOperator(
        task_id="process_feedback",
        python_callable=_run_feedback_loop,
        provide_context=True,
    )

    log_task = PythonOperator(
        task_id="log_improvement_signals",
        python_callable=_log_summary,
        provide_context=True,
    )

    process_task >> log_task
