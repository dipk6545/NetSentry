"""
NetSentry v2 — Production Airflow Orchestration DAG (`dags/netsentry_pipeline_dag.py`).
========================================================================================
Automates the full 14-day continuous retraining lifecycle using DockerOperator:
1. Ingest & Validate raw network flows (Polars).
2. Domain Feature Engineering (71 features).
3. Baseline Multi-Model Tournament (CatBoost, LightGBM, XGBoost, etc.).
4. Optuna Bayesian HPO on the winning architecture.
5. Decision Threshold Calibration (tau*) & Untouched Test Set Evaluation.
6. Quality Gate Auditing & MLflow Model Registry @champion Promotion.

Schedule: Every 14 days (0 2 */14 * *)
"""

from datetime import datetime, timedelta
import os
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Default configurations for all tasks in this DAG
default_args = {
    "owner": "netsentry_mlops",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

# Host project root (mapped to volume mounts in DockerOperator)
PROJECT_DIR = os.getenv("NETSENTRY_HOST_DIR", "/app")
DOCKER_IMAGE = "netsentry-pipeline:latest"

# Standard host mounts so the container reads/writes directly to persistent storage
standard_mounts = [
    Mount(source=f"{PROJECT_DIR}/data", target="/app/data", type="bind"),
    Mount(source=f"{PROJECT_DIR}/artifacts", target="/app/artifacts", type="bind"),
    Mount(source=f"{PROJECT_DIR}/configs", target="/app/configs", type="bind"),
    Mount(source=f"{PROJECT_DIR}/mlruns", target="/app/mlruns", type="bind"),
    Mount(source=f"{PROJECT_DIR}/mlruns.db", target="/app/mlruns.db", type="bind"),
]

with DAG(
    dag_id="netsentry_continuous_training_pipeline",
    default_args=default_args,
    description="Automated 14-day network intrusion detection retraining & @champion promotion",
    schedule="0 2 */14 * *",  # Run every 14 days at 2:00 AM UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "security", "netsentry", "training"],
) as dag:

    # --------------------------------------------------------------------------
    # Task 1: Baseline Tournament
    # --------------------------------------------------------------------------
    train_baselines = DockerOperator(
        task_id="train_baseline_tournament",
        image=DOCKER_IMAGE,
        api_version="auto",
        auto_remove=True,
        command="scripts/train_baselines.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=standard_mounts,
        mount_tmp_dir=False,
    )

    # --------------------------------------------------------------------------
    # Task 2: Multi-Candidate Tuning, Threshold Search, & Champion Promotion
    # --------------------------------------------------------------------------
    run_full_pipeline = DockerOperator(
        task_id="run_tuning_evaluation_and_promotion",
        image=DOCKER_IMAGE,
        api_version="auto",
        auto_remove=True,
        command="scripts/run_pipeline.py --top-k 3",
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        mounts=standard_mounts,
        mount_tmp_dir=False,
    )

    # --------------------------------------------------------------------------
    # Workflow Execution Graph
    # --------------------------------------------------------------------------
    train_baselines >> run_full_pipeline