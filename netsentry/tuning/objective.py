"""
Optuna Objective Module (`netsentry.tuning.objective`).
-------------------------------------------------------
Evaluates a single hyperparameter trial using Model Factory and Trainer.
Tracks trial iterations in MLflow.
"""

from typing import Any, Dict, Optional
import numpy as np
import optuna

from netsentry.models.config import ModelConfig
from netsentry.models.factory import create_model
from netsentry.training.config import TrainingConfig
from netsentry.training.trainer import Trainer
from netsentry.training.tracking import MLflowTracker
from netsentry.tuning.config import ParameterSearchSpace, TuningConfig


def sample_parameters(trial: optuna.Trial, search_space: Dict[str, ParameterSearchSpace]) -> Dict[str, Any]:
    """Generates hyperparameter samples from the defined search space."""
    params = {}
    for p_name, space in search_space.items():
        if space.param_type == "int":
            params[p_name] = trial.suggest_int(
                name=p_name,
                low=int(space.low),
                high=int(space.high),
                step=int(space.step) if space.step else 1,
                log=space.log,
            )
        elif space.param_type == "float":
            params[p_name] = trial.suggest_float(
                name=p_name,
                low=float(space.low),
                high=float(space.high),
                step=float(space.step) if space.step else None,
                log=space.log,
            )
        elif space.param_type == "categorical":
            params[p_name] = trial.suggest_categorical(name=p_name, choices=space.choices)
    return params


class GenericObjective:
    """Model-agnostic Optuna objective function."""

    def __init__(
        self,
        base_model_config: ModelConfig,
        tuning_config: TuningConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tracker: Optional[MLflowTracker] = None,
    ):
        self.base_model_config = base_model_config
        self.tuning_config = tuning_config
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.tracker = tracker

    def __call__(self, trial: optuna.Trial) -> float:
        # 1. Sample hyperparameters for this trial
        trial_params = sample_parameters(trial, self.tuning_config.search_space)

        # 2. Merge sampled parameters over base parameters
        merged_params = {**self.base_model_config.parameters, **trial_params}
        trial_model_cfg = ModelConfig(
            name=f"{self.base_model_config.name}_trial_{trial.number}",
            implementation=self.base_model_config.implementation,
            preprocessing=self.base_model_config.preprocessing,
            parameters=merged_params,
        )

        # 3. Build model pipeline
        model = create_model(trial_model_cfg)

        # 4. Train model using Phase 5 Trainer
        training_cfg = TrainingConfig(
            experiment_name=self.tracker.experiment_name if self.tracker else "netsentry-hpo",
        )
        trainer = Trainer(
            model=model,
            model_config=trial_model_cfg,
            training_config=training_cfg,
            tracker=self.tracker,
        )

        run_name = f"{self.base_model_config.name}_trial_{trial.number}"
        res = trainer.train(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            run_name=run_name,
        )

        # 5. Extract optimization target metric
        target_metric = self.tuning_config.tuning.metric
        score = res.metrics.get(target_metric, 0.0)

        # Log trial score into Optuna user attributes
        trial.set_user_attr("all_metrics", res.metrics)
        return float(score)
