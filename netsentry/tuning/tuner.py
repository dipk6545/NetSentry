"""
Hyperparameter Tuner & Baseline Selection (`netsentry.tuning.tuner`).
---------------------------------------------------------------------
Coordinates baseline comparison, Optuna study execution, tuned YAML generation,
and fresh final retraining on Train + Val.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import optuna
import polars as pl
import yaml

from netsentry.models.config import ModelConfig, load_model_config
from netsentry.models.factory import create_model
from netsentry.training.config import TrainingConfig
from netsentry.training.trainer import Trainer
from netsentry.training.tracking import MLflowTracker
from netsentry.tuning.config import TuningConfig, load_tuning_config
from netsentry.tuning.objective import GenericObjective
from netsentry.tuning.result import BaselineComparisonReport, BaselineModelResult, TuningResult


def compare_baselines(
    model_configs: List[Union[str, Path, ModelConfig]],
    X_train: Union[np.ndarray, pl.DataFrame],
    y_train: Union[np.ndarray, pl.Series],
    X_val: Union[np.ndarray, pl.DataFrame],
    y_val: Union[np.ndarray, pl.Series],
    selection_metric: str = "roc_auc",
    tracker: Optional[MLflowTracker] = None,
) -> BaselineComparisonReport:
    """
    Trains all candidate baseline models on X_train, evaluates on X_val,
    and returns a structured comparison report ranking them by selection_metric.
    """
    results: List[BaselineModelResult] = []

    for cfg_item in model_configs:
        cfg = cfg_item if isinstance(cfg_item, ModelConfig) else load_model_config(cfg_item)
        model = create_model(cfg)

        trainer = Trainer(
            model=model,
            model_config=cfg,
            training_config=TrainingConfig(
                experiment_name=tracker.experiment_name if tracker else "netsentry-baselines",
                evaluation={"primary_metric": selection_metric},
            ),
            tracker=tracker,
        )

        res = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            run_name=f"baseline_{cfg.name}",
        )

        score = res.metrics.get(selection_metric, 0.0)
        results.append(
            BaselineModelResult(
                model_name=cfg.name,
                run_id=res.run_id,
                validation_metrics=res.metrics,
                selection_metric=selection_metric,
                selection_score=score,
                selected=False,
            )
        )

    # Rank by score descending
    sorted_results = sorted(results, key=lambda r: r.selection_score, reverse=True)
    best_candidate = sorted_results[0]

    # Mark selected winner
    final_results = [
        BaselineModelResult(
            model_name=r.model_name,
            run_id=r.run_id,
            validation_metrics=r.validation_metrics,
            selection_metric=r.selection_metric,
            selection_score=r.selection_score,
            selected=(r.model_name == best_candidate.model_name),
        )
        for r in sorted_results
    ]

    return BaselineComparisonReport(
        results=final_results,
        best_baseline=final_results[0],
        selection_metric=selection_metric,
    )


class Tuner:
    """Coordinates Optuna study, tuned configuration emission, and final retraining."""

    def __init__(
        self,
        tuning_config: Union[str, Path, TuningConfig],
        tracker: Optional[MLflowTracker] = None,
    ):
        self.tuning_config = (
            tuning_config if isinstance(tuning_config, TuningConfig) else load_tuning_config(tuning_config)
        )
        self.base_model_config = load_model_config(self.tuning_config.model_config_path)
        self.tracker = tracker or MLflowTracker(
            experiment_name=f"hpo-{self.base_model_config.name}",
            tracking_uri="sqlite:///mlruns.db",
            enabled=True,
        )

    def tune(
        self,
        X_train: Union[np.ndarray, pl.DataFrame],
        y_train: Union[np.ndarray, pl.Series],
        X_val: Union[np.ndarray, pl.DataFrame],
        y_val: Union[np.ndarray, pl.Series],
        retrain_on_train_val: bool = True,
        output_dir: str = "configs/models",
    ) -> TuningResult:
        """
        Executes Optuna study on X_train/X_val, writes <model>_tuned.yaml,
        and retrains fresh model on Train + Val if specified.
        """
        # Array conversion
        X_tr = X_train.to_numpy() if isinstance(X_train, pl.DataFrame) else np.asarray(X_train)
        y_tr = y_train.to_numpy() if isinstance(y_train, pl.Series) else np.asarray(y_train)
        X_v = X_val.to_numpy() if isinstance(X_val, pl.DataFrame) else np.asarray(X_val)
        y_v = y_val.to_numpy() if isinstance(y_val, pl.Series) else np.asarray(y_val)

        # 1. Setup Optuna Study
        study = optuna.create_study(
            study_name=f"study-{self.base_model_config.name}",
            direction=self.tuning_config.tuning.direction,
        )

        objective = GenericObjective(
            base_model_config=self.base_model_config,
            tuning_config=self.tuning_config,
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_v,
            y_val=y_v,
            tracker=self.tracker,
        )

        # Suppress excessive Optuna logging during tests/runs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(
            objective,
            n_trials=self.tuning_config.tuning.n_trials,
            timeout=self.tuning_config.tuning.timeout_seconds,
        )

        best_params = study.best_params
        best_score = study.best_value

        # 2. Emit <model>_tuned.yaml without modifying baseline YAML
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tuned_yaml_path = out_dir / f"{self.base_model_config.name}_tuned.yaml"

        merged_tuned_params = {**self.base_model_config.parameters, **best_params}
        tuned_yaml_content = {
            "name": f"{self.base_model_config.name}_tuned",
            "implementation": {
                "class_path": self.base_model_config.implementation.class_path,
            },
            "preprocessing": {
                "scaling": self.base_model_config.preprocessing.scaling,
            },
            "parameters": merged_tuned_params,
        }

        with open(tuned_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(tuned_yaml_content, f, sort_keys=False)

        # 3. Final Retraining Strategy B: Train + Validation
        final_model_cfg = load_model_config(tuned_yaml_path)
        final_model = create_model(final_model_cfg)

        if retrain_on_train_val:
            X_combined = np.vstack([X_tr, X_v])
            y_combined = np.concatenate([y_tr, y_v])
            final_model.fit(X_combined, y_combined)
        else:
            final_model.fit(X_tr, y_tr)

        return TuningResult(
            study=study,
            best_params=best_params,
            best_score=best_score,
            tuned_model_config_path=str(tuned_yaml_path),
            final_model=final_model,
            retrained_on_train_val=retrain_on_train_val,
        )
