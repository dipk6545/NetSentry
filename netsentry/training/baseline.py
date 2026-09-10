"""
Baseline Training Orchestrator (`netsentry.training.baseline`).
--------------------------------------------------------------
Discovers candidate baseline model configurations, coordinates training
across all algorithms via ModelFactory and Trainer, ranks models by validation
Recall (with secondary tie-breaking), and emits a machine-readable summary artifact.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Union
import numpy as np
import polars as pl

from netsentry.models.config import ModelConfig, load_model_config
from netsentry.models.factory import create_model
from netsentry.training.config import TrainingConfig
from netsentry.training.trainer import Trainer, TrainingResult
from netsentry.training.tracking import MLflowTracker

logger = logging.getLogger("NetSentryBaselineOrchestrator")


def get_git_commit_sha() -> str:
    """Retrieves current Git commit SHA, or 'unknown' if not available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class BaselineModelEvaluation:
    """Evaluation summary for a single baseline candidate model."""
    name: str
    config_path: str
    run_id: Optional[str]
    status: str  # "SUCCESS" or "FAILED"
    metrics: Dict[str, float]
    error_message: Optional[str] = None


@dataclass(frozen=True)
class BaselineOrchestrationResult:
    """Overall outcome of the baseline comparison tournament."""
    winner: str
    primary_metric: str
    direction: str
    models: List[BaselineModelEvaluation]
    git_sha: str
    output_artifact_path: Optional[str] = None


def discover_baseline_configs(config_dir: Union[str, Path] = "configs/models") -> List[Path]:
    """
    Discovers all baseline model YAML files in the given directory.
    Explicitly excludes tuned configurations (*_tuned.yaml).
    """
    p = Path(config_dir)
    if not p.exists():
        return []

    # Filter only .yaml and .yml files that do not end with _tuned.yaml
    configs = [
        f for f in sorted(p.glob("*.y*ml"))
        if not f.name.endswith("_tuned.yaml") and not f.name.endswith("_tuned.yml")
    ]
    return configs


class BaselineOrchestrator:
    """Orchestrates multi-model baseline training, ranking, and artifact persistence."""

    def __init__(
        self,
        config_dir: Union[str, Path] = "configs/models",
        primary_metric: str = "recall",
        direction: str = "maximize",
        tracker: Optional[MLflowTracker] = None,
        artifacts_output_dir: Union[str, Path] = "artifacts/baselines",
    ):
        self.config_dir = Path(config_dir)
        self.primary_metric = primary_metric.lower()
        self.direction = direction.lower()
        self.tracker = tracker or MLflowTracker(
            experiment_name="netsentry-baseline-orchestration",
            tracking_uri="sqlite:///mlruns.db",
            enabled=True,
        )
        self.artifacts_output_dir = Path(artifacts_output_dir)

    def run(
        self,
        X_train: Union[np.ndarray, pl.DataFrame],
        y_train: Union[np.ndarray, pl.Series],
        X_val: Union[np.ndarray, pl.DataFrame],
        y_val: Union[np.ndarray, pl.Series],
    ) -> BaselineOrchestrationResult:
        """
        Executes baseline tournament on training and validation splits only.
        Test data MUST NOT be passed.
        """
        yaml_paths = discover_baseline_configs(self.config_dir)
        if not yaml_paths:
            raise RuntimeError(f"No baseline model YAML configurations found in: {self.config_dir}")

        evaluations: List[BaselineModelEvaluation] = []
        git_sha = get_git_commit_sha()

        total = len(yaml_paths)
        print(f"\nStarting baseline training tournament across {total} models...")

        for idx, y_path in enumerate(yaml_paths, 1):
            try:
                model_cfg = load_model_config(y_path)
                model_name = model_cfg.name
                print(f"[{idx}/{total}] Training {model_name} ({y_path.name})...")

                model = create_model(model_cfg)
                trainer = Trainer(
                    model=model,
                    model_config=model_cfg,
                    training_config=TrainingConfig(
                        experiment_name=self.tracker.experiment_name,
                    ),
                    tracker=self.tracker,
                )

                train_res: TrainingResult = trainer.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    run_name=f"baseline_{model_name}",
                )

                m = train_res.metrics
                evaluations.append(
                    BaselineModelEvaluation(
                        name=model_name,
                        config_path=str(y_path),
                        run_id=train_res.run_id,
                        status="SUCCESS",
                        metrics=m,
                    )
                )

                p_val = m.get(self.primary_metric, 0.0)
                f1_val = m.get("f1", 0.0)
                roc_val = m.get("roc_auc", 0.0)
                print(f"      {self.primary_metric.capitalize()}: {p_val:.4f} | F1: {f1_val:.4f} | ROC-AUC: {roc_val:.4f} | Run: {train_res.run_id}")

            except Exception as e:
                logger.error(f"Baseline training failed for {y_path}: {e}", exc_info=True)
                print(f"      FAILED: {e}")
                evaluations.append(
                    BaselineModelEvaluation(
                        name=y_path.stem,
                        config_path=str(y_path),
                        run_id=None,
                        status="FAILED",
                        metrics={},
                        error_message=str(e),
                    )
                )

        # Audit successful runs
        successful = [e for e in evaluations if e.status == "SUCCESS"]
        if not successful:
            raise RuntimeError("All baseline model runs failed. Baseline tournament cannot proceed.")

        # Tie-breaking ranking key:
        # 1. Primary metric (e.g. Recall)
        # 2. F1
        # 3. Precision
        # 4. ROC-AUC
        def ranking_key(item: BaselineModelEvaluation):
            m = item.metrics
            return (
                m.get(self.primary_metric, 0.0),
                m.get("f1", 0.0),
                m.get("precision", 0.0),
                m.get("roc_auc", 0.0),
            )

        ranked = sorted(successful, key=ranking_key, reverse=(self.direction == "maximize"))
        winner = ranked[0].name

        # Persist machine-readable comparison artifact
        self.artifacts_output_dir.mkdir(parents=True, exist_ok=True)
        artifact_file = self.artifacts_output_dir / "latest.json"

        artifact_content = {
            "winner": winner,
            "metric": self.primary_metric,
            "direction": self.direction,
            "git_commit_sha": git_sha,
            "models": [
                {
                    "name": e.name,
                    "status": e.status,
                    "metrics": e.metrics,
                    "run_id": e.run_id,
                    "config_path": e.config_path,
                }
                for e in evaluations
            ],
        }

        with open(artifact_file, "w", encoding="utf-8") as f:
            json.dump(artifact_content, f, indent=2)

        print("\n" + "=" * 50)
        print("BASELINE COMPARISON SUMMARY")
        print("=" * 50)
        print(f"Winner: {winner}")
        print(f"Selection Metric: {self.primary_metric.capitalize()} ({self.direction})")
        print(f"Artifact Persisted: {artifact_file}")
        print("=" * 50 + "\n")

        return BaselineOrchestrationResult(
            winner=winner,
            primary_metric=self.primary_metric,
            direction=self.direction,
            models=evaluations,
            git_sha=git_sha,
            output_artifact_path=str(artifact_file),
        )
