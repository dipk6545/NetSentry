"""Tests for Model Interface, capabilities, and zero data leakage verification."""
import numpy as np
import pytest
from netsentry.models.config import load_model_config
from netsentry.models.factory import create_model
from netsentry.models.interface import ModelInterface, ProbabilisticModelInterface, has_predict_proba


@pytest.fixture
def mock_classification_data():
    np.random.seed(42)
    X_train = np.random.randn(100, 10) * 100.0 + 50.0  # Large spread to verify scaling
    y_train = np.random.choice([0, 1], size=100)
    X_val = np.random.randn(20, 10) * 100.0 + 50.0
    return X_train, y_train, X_val


def test_fit_predict_and_capabilities(mock_classification_data):
    X_train, y_train, X_val = mock_classification_data
    config = load_model_config("configs/models/logistic_regression.yaml")
    model = create_model(config)

    # Fits the pipeline
    fitted_model = model.fit(X_train, y_train)
    assert fitted_model is model

    # Check contract conformance
    assert isinstance(model, ModelInterface)
    assert isinstance(model, ProbabilisticModelInterface)
    assert has_predict_proba(model) is True

    # Predictions
    preds = model.predict(X_val)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (20,)
    assert set(np.unique(preds)).issubset({0, 1})

    # Probabilities
    probs = model.predict_proba(X_val)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (20, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_preprocessing_fitted_only_on_train(mock_classification_data):
    """
    CRITICAL TEST (Rule 5):
    Verifies that the preprocessor scaler's mean_ and var_ are calculated ONLY
    from X_train and remain identical during X_val evaluation (zero leakage).
    """
    X_train, y_train, X_val = mock_classification_data
    config = load_model_config("configs/models/logistic_regression.yaml")
    model = create_model(config)

    model.fit(X_train, y_train)

    scaler = model.named_steps["preprocessor"]
    fitted_mean_train = np.copy(scaler.mean_)
    fitted_var_train = np.copy(scaler.var_)

    # Confirm scaler matches X_train statistics exactly
    assert np.allclose(fitted_mean_train, np.mean(X_train, axis=0))

    # Evaluate validation data
    _ = model.predict(X_val)
    _ = model.predict_proba(X_val)

    # Verify scaler statistics did NOT change when predicting on validation data
    assert np.array_equal(scaler.mean_, fitted_mean_train)
    assert np.array_equal(scaler.var_, fitted_var_train)
