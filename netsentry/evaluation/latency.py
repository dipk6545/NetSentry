"""
Inference Latency Profiler (`netsentry.evaluation.latency`).
-----------------------------------------------------------
Measures model inference execution latency per batch and per sample,
computing distribution percentiles (mean, median, p95, p99).
"""

from dataclasses import dataclass
import time
from typing import Any, List, Union
import numpy as np
import polars as pl


@dataclass(frozen=True)
class LatencyProfile:
    """Latency distribution summary in milliseconds."""
    samples_evaluated: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def measure_inference_latency(
    model: Any,
    X: Union[np.ndarray, pl.DataFrame],
    iterations: int = 50,
    warmup_iterations: int = 5,
) -> LatencyProfile:
    """
    Profiles inference latency per sample on input features X.
    """
    features = X.to_numpy() if isinstance(X, pl.DataFrame) else np.asarray(X)
    n_samples = len(features)
    if n_samples == 0:
        return LatencyProfile(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Warmup
    for _ in range(warmup_iterations):
        _ = model.predict(features[: min(10, n_samples)])

    latencies_per_sample_ms: List[float] = []

    # Benchmark single-sample or batch inference
    for _ in range(iterations):
        # Sample random row
        idx = np.random.randint(0, n_samples)
        sample = features[idx : idx + 1]

        t0 = time.perf_counter()
        _ = model.predict(sample)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        latencies_per_sample_ms.append(duration_ms)

    arr = np.array(latencies_per_sample_ms)
    return LatencyProfile(
        samples_evaluated=iterations,
        mean_ms=round(float(np.mean(arr)), 4),
        median_ms=round(float(np.median(arr)), 4),
        p95_ms=round(float(np.percentile(arr, 95)), 4),
        p99_ms=round(float(np.percentile(arr, 99)), 4),
        min_ms=round(float(np.min(arr)), 4),
        max_ms=round(float(np.max(arr)), 4),
    )
