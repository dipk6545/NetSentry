"""Tests for attack slice metrics."""
import pytest
from netsentry.evaluation.slices import compute_slice_metrics


def test_compute_slice_metrics():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 0]
    labels = ["BENIGN", "BENIGN", "DDoS", "DDoS", "PortScan"]

    slices = compute_slice_metrics(y_true, y_pred, labels)

    assert "BENIGN" in slices
    assert "DDoS" in slices
    assert "PortScan" in slices

    assert slices["BENIGN"].sample_count == 2
    assert slices["BENIGN"].accuracy == 1.0

    assert slices["DDoS"].sample_count == 2
    assert slices["DDoS"].recall == 1.0

    assert slices["PortScan"].sample_count == 1
    assert slices["PortScan"].recall == 0.0  # Misclassified
