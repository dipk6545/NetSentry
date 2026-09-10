"""NetSentry Model Registry & Promotion Layer."""
from netsentry.registry.config import (
    RegistryConfig,
    AliasConfig,
    ComparisonConfig,
    PromotionConfig,
    PromotionRule,
    load_registry_config,
)
from netsentry.registry.client import RegistryClient
from netsentry.registry.registration import ModelRegistrar, RegistrationResult
from netsentry.registry.aliases import AliasManager, AliasState
from netsentry.registry.comparator import ChampionChallengerComparator, ModelComparisonMetrics
from netsentry.registry.promotion import PromotionDecision, evaluate_promotion
from netsentry.registry.lifecycle import ModelLifecycleManager, LifecyclePromotionResult

__all__ = [
    "RegistryConfig",
    "AliasConfig",
    "ComparisonConfig",
    "PromotionConfig",
    "PromotionRule",
    "load_registry_config",
    "RegistryClient",
    "ModelRegistrar",
    "RegistrationResult",
    "AliasManager",
    "AliasState",
    "ChampionChallengerComparator",
    "ModelComparisonMetrics",
    "PromotionDecision",
    "evaluate_promotion",
    "ModelLifecycleManager",
    "LifecyclePromotionResult",
]
