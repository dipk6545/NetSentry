"""
Registry Configuration Schema (`netsentry.registry.config`).
-----------------------------------------------------------
Defines typed configuration for MLflow Model Registry, lifecycle aliases,
champion-challenger metrics comparison, and promotion policies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


class RegistryConfigValidationError(ValueError):
    """Raised when registry configuration fails schema validation."""
    pass


@dataclass(frozen=True)
class AliasConfig:
    """Registry lifecycle alias names."""
    champion: str = "champion"
    challenger: str = "challenger"


@dataclass(frozen=True)
class ComparisonConfig:
    """Comparison metric specification."""
    primary_metric: str = "recall"
    secondary_metrics: List[str] = field(
        default_factory=lambda: [
            "precision",
            "f1",
            "roc_auc",
            "pr_auc",
            "fnr",
            "fpr",
            "latency_p95_ms",
        ]
    )


@dataclass(frozen=True)
class PromotionRule:
    """Constraints for an individual metric during promotion evaluation."""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    improvement_required: bool = False


@dataclass(frozen=True)
class PromotionConfig:
    """Promotion requirements and operational constraints."""
    require_quality_gates: bool = True
    require_challenger_better: bool = True
    rules: Dict[str, PromotionRule] = field(default_factory=dict)


@dataclass(frozen=True)
class RegistryConfig:
    """Root configuration for model registry and promotion pipeline."""
    model_name: str = "NetSentry"
    aliases: AliasConfig = field(default_factory=AliasConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)

    def __post_init__(self):
        if not self.model_name or not isinstance(self.model_name, str):
            raise RegistryConfigValidationError("Registry 'model_name' must be a non-empty string.")


def load_registry_config(config_source: Union[str, Path, Dict[str, Any]]) -> RegistryConfig:
    """Loads and validates a RegistryConfig from a YAML file or dict."""
    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        if not path.exists():
            raise FileNotFoundError(f"Registry config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
    elif isinstance(config_source, dict):
        raw_cfg = config_source
    else:
        raise TypeError(f"Expected file path or dict, got {type(config_source)}")

    if not isinstance(raw_cfg, dict):
        raise RegistryConfigValidationError("Registry config root must be a mapping.")

    # 1. Registry info
    reg_raw = raw_cfg.get("registry", {})
    model_name = str(reg_raw.get("model_name", "NetSentry"))

    # 2. Aliases
    aliases_raw = raw_cfg.get("aliases", {})
    alias_cfg = AliasConfig(
        champion=str(aliases_raw.get("champion", "champion")),
        challenger=str(aliases_raw.get("challenger", "challenger")),
    )

    # 3. Comparison
    comp_raw = raw_cfg.get("comparison", {})
    comp_cfg = ComparisonConfig(
        primary_metric=str(comp_raw.get("primary_metric", "recall")).lower(),
        secondary_metrics=[
            str(m).lower()
            for m in comp_raw.get(
                "secondary_metrics",
                ["precision", "f1", "roc_auc", "pr_auc", "fnr", "fpr", "latency_p95_ms"],
            )
        ],
    )

    # 4. Promotion rules
    prom_raw = raw_cfg.get("promotion", {})
    require_qg = bool(prom_raw.get("require_quality_gates", True))
    require_better = bool(prom_raw.get("require_challenger_better", True))

    rules_dict: Dict[str, PromotionRule] = {}
    rules_raw = prom_raw.get("rules", {})
    if isinstance(rules_raw, dict):
        for metric_name, rule_spec in rules_raw.items():
            if not isinstance(rule_spec, dict):
                raise RegistryConfigValidationError(f"Rule for '{metric_name}' must be a mapping.")
            rules_dict[metric_name.lower()] = PromotionRule(
                minimum=float(rule_spec["minimum"]) if "minimum" in rule_spec else None,
                maximum=float(rule_spec["maximum"]) if "maximum" in rule_spec else None,
                improvement_required=bool(rule_spec.get("improvement_required", False)),
            )

    prom_cfg = PromotionConfig(
        require_quality_gates=require_qg,
        require_challenger_better=require_better,
        rules=rules_dict,
    )

    return RegistryConfig(
        model_name=model_name,
        aliases=alias_cfg,
        comparison=comp_cfg,
        promotion=prom_cfg,
    )
