"""Tests for preprocessing transformer builder."""
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from netsentry.models.config import PreprocessingConfig
from netsentry.models.preprocessing import build_preprocessor


def test_build_preprocessor_none():
    prep = build_preprocessor(PreprocessingConfig(scaling="none"))
    assert prep is None


def test_build_preprocessor_standard():
    prep = build_preprocessor(PreprocessingConfig(scaling="standard"))
    assert isinstance(prep, StandardScaler)


def test_build_preprocessor_minmax():
    prep = build_preprocessor(PreprocessingConfig(scaling="minmax"))
    assert isinstance(prep, MinMaxScaler)


def test_build_preprocessor_robust():
    prep = build_preprocessor(PreprocessingConfig(scaling="robust"))
    assert isinstance(prep, RobustScaler)
