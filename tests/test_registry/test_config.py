"""Tests for RegistryConfig parser and validation rules."""
from pathlib import Path
import pytest
from netsentry.registry.config import load_registry_config, RegistryConfigValidationError


def test_load_valid_promotion_yaml():
    cfg = load_registry_config("configs/registry/promotion.yaml")
    assert cfg.model_name == "NetSentry"
    assert cfg.aliases.champion == "champion"
    assert cfg.aliases.challenger == "challenger"
    assert cfg.comparison.primary_metric == "recall"
    assert "recall" in cfg.promotion.rules
    assert cfg.promotion.rules["recall"].minimum == 0.90
    assert cfg.promotion.rules["recall"].improvement_required is True


def test_invalid_registry_empty_name_raises(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("registry:\n  model_name: ''\n")
    with pytest.raises(RegistryConfigValidationError):
        load_registry_config(bad_yaml)
