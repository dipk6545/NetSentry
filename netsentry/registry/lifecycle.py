"""
Model Lifecycle Manager (`netsentry.registry.lifecycle`).
--------------------------------------------------------
Orchestrates the entire production promotion workflow:
Register -> Assign Challenger -> Retrieve Champion -> Compare -> Promote or Retain -> MLflow Audit.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import numpy as np
import polars as pl

from netsentry.registry.aliases import AliasManager
from netsentry.registry.client import RegistryClient
from netsentry.registry.comparator import ChampionChallengerComparator, ModelComparisonMetrics
from netsentry.registry.config import RegistryConfig, load_registry_config
from netsentry.registry.promotion import PromotionDecision, evaluate_promotion
from netsentry.registry.registration import ModelRegistrar, RegistrationResult
from netsentry.training.tracking import MLflowTracker


@dataclass(frozen=True)
class LifecyclePromotionResult:
    """Summary of the complete lifecycle registration and promotion process."""
    registered_version: str
    decision: PromotionDecision
    champion_version_after: str
    is_new_champion: bool


class ModelLifecycleManager:
    """End-to-end coordinator for registry operations and production promotion."""

    def __init__(
        self,
        config: Optional[Union[str, RegistryConfig]] = None,
        registry_client: Optional[RegistryClient] = None,
        tracker: Optional[MLflowTracker] = None,
    ):
        if config is None:
            self.config = RegistryConfig()
        elif isinstance(config, str):
            self.config = load_registry_config(config)
        else:
            self.config = config

        self.client = registry_client or RegistryClient()
        self.registrar = ModelRegistrar(client=self.client)
        self.alias_manager = AliasManager(
            client=self.client,
            model_name=self.config.model_name,
            config=self.config.aliases,
        )
        self.comparator = ChampionChallengerComparator(comp_config=self.config.comparison)
        self.tracker = tracker

    def promote_candidate(
        self,
        candidate_model: Any,
        run_id: str,
        X_val: Union[np.ndarray, pl.DataFrame],
        y_val: Union[np.ndarray, pl.Series],
        X_test: Union[np.ndarray, pl.DataFrame],
        y_test: Union[np.ndarray, pl.Series],
        labels_test: Union[np.ndarray, pl.Series, list],
        metadata: Optional[Dict[str, Any]] = None,
        optimal_threshold: Optional[float] = None,
    ) -> LifecyclePromotionResult:
        """
        Executes registration, comparison vs incumbent champion, and conditional promotion.
        """
        # 1. Register candidate as new immutable version
        reg: RegistrationResult = self.registrar.register(
            model_name=self.config.model_name,
            run_id=run_id,
            metadata=metadata,
            optimal_threshold=optimal_threshold,
        )
        challenger_ver = reg.version

        # 2. Check current Champion
        current_champ_ver = self.alias_manager.get_champion_version()

        # Case A: First Release
        if current_champ_ver is None:
            # Self-compare to evaluate base constraints
            comp = self.comparator.compare(
                champion_model=candidate_model,
                challenger_model=candidate_model,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                labels_test=labels_test,
            )
            decision = evaluate_promotion(
                comparison=comp,
                champion_version=None,
                challenger_version=challenger_ver,
                config=self.config.promotion,
            )
            if decision.passed:
                self.alias_manager.assign_champion(challenger_ver)
                champ_after = challenger_ver
                is_new = True
            else:
                champ_after = "none"
                is_new = False

            return LifecyclePromotionResult(
                registered_version=challenger_ver,
                decision=decision,
                champion_version_after=champ_after,
                is_new_champion=is_new,
            )

        # Case B: Second+ Release with Incumbent Champion
        # Assign @challenger alias
        self.alias_manager.assign_challenger(challenger_ver)

        # Load Champion model artifact
        try:
            champion_model = self.client.load_model_by_version(
                name=self.config.model_name,
                version=current_champ_ver,
            )
        except Exception as e:
            # If incumbent champion artifact is missing or corrupted on disk, fall back cleanly
            import logging
            logging.getLogger("ModelLifecycleManager").warning(
                f"Could not load incumbent champion v{current_champ_ver} artifact ({e}). Treating as first release."
            )
            champion_model = None
            current_champ_ver = None

        if champion_model is None:
            comp = self.comparator.compare(
                champion_model=candidate_model,
                challenger_model=candidate_model,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                labels_test=labels_test,
            )
            decision = evaluate_promotion(
                comparison=comp,
                champion_version=None,
                challenger_version=challenger_ver,
                config=self.config.promotion,
            )
            if decision.passed:
                self.alias_manager.assign_champion(challenger_ver)
                champ_after = challenger_ver
                is_new = True
            else:
                champ_after = "none"
                is_new = False

            return LifecyclePromotionResult(
                registered_version=challenger_ver,
                decision=decision,
                champion_version_after=champ_after,
                is_new_champion=is_new,
            )

        # Execute side-by-side comparison on identical test protocol
        comp = self.comparator.compare(
            champion_model=champion_model,
            challenger_model=candidate_model,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            labels_test=labels_test,
        )

        # Evaluate promotion policy
        decision = evaluate_promotion(
            comparison=comp,
            champion_version=current_champ_ver,
            challenger_version=challenger_ver,
            config=self.config.promotion,
        )

        if decision.passed:
            self.alias_manager.promote_challenger_to_champion(challenger_ver)
            champ_after = challenger_ver
            is_new = True
        else:
            # Challenger rejected; retain incumbent champion and remove challenger alias
            self.client.delete_model_alias(self.config.model_name, self.config.aliases.challenger)
            champ_after = current_champ_ver
            is_new = False

        # Log promotion run to MLflow if tracking active
        if self.tracker and self.tracker.enabled:
            with self.tracker.start_run(run_name=f"promotion_v{challenger_ver}_vs_v{current_champ_ver}"):
                self.tracker.log_params({
                    "promotion_decision": "PROMOTE" if decision.passed else "REJECT",
                    "challenger_version": challenger_ver,
                    "champion_version": current_champ_ver,
                    "reason": decision.reason,
                })
                # Log metrics deltas
                deltas_to_log = {f"delta_{k}": v for k, v in comp.deltas.items()}
                self.tracker.log_metrics(deltas_to_log)

        return LifecyclePromotionResult(
            registered_version=challenger_ver,
            decision=decision,
            champion_version_after=champ_after,
            is_new_champion=is_new,
        )
