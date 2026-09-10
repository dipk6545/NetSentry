"""Tests for inference latency profiler."""
import numpy as np
from sklearn.dummy import DummyClassifier
from netsentry.evaluation.latency import measure_inference_latency


def test_measure_inference_latency():
    model = DummyClassifier(strategy="constant", constant=1)
    X = np.random.randn(50, 4)
    y = np.ones(50)
    model.fit(X, y)

    profile = measure_inference_latency(model, X, iterations=20, warmup_iterations=2)
    assert profile.samples_evaluated == 20
    assert profile.p95_ms >= 0.0
    assert profile.mean_ms <= 100.0  # Dummy classifier is sub-millisecond
