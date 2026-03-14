"""Tests for the MetricsStore in-memory buffer."""

import pytest
import os
import sys
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.metrics_store import MetricsStore


@pytest.fixture
def store():
    """Create a fresh MetricsStore."""
    return MetricsStore(max_size=100)


def make_result(model="champion", prediction=5.5, latency=2.0, request_id="req-001"):
    """Helper to create a prediction result dict."""
    return {
        "model": model,
        "prediction": [prediction],
        "latency_ms": latency,
        "timestamp": time.time(),
        "request_id": request_id,
    }


class TestMetricsStore:
    def test_record_and_count(self, store):
        """Recording predictions should increase the count."""
        store.record(make_result("champion"))
        store.record(make_result("challenger"))
        store.record(make_result("champion"))

        summary = store.get_summary()
        assert summary["champion"]["count"] == 2
        assert summary["challenger"]["count"] == 1
        assert summary["total_requests"] == 3

    def test_empty_summary(self, store):
        """Empty store should return zeroed stats."""
        summary = store.get_summary()
        assert summary["champion"]["count"] == 0
        assert summary["challenger"]["count"] == 0
        assert summary["champion"]["avg_latency"] == 0
        assert summary["total_requests"] == 0

    def test_latency_stats(self, store):
        """Average and P95 latency should be computed correctly."""
        for lat in [1.0, 2.0, 3.0, 4.0, 5.0]:
            store.record(make_result("champion", latency=lat))

        summary = store.get_summary()
        assert summary["champion"]["avg_latency"] == 3.0
        assert summary["champion"]["p95_latency"] > 4.0  # P95 of [1,2,3,4,5]

    def test_max_size_enforced(self):
        """Buffer should respect max_size."""
        store = MetricsStore(max_size=10)
        for i in range(20):
            store.record(make_result("champion", request_id=f"req-{i}"))

        summary = store.get_summary()
        assert summary["total_requests"] == 10  # Only last 10 kept

    def test_record_outcome(self, store):
        """Recording outcomes should be retrievable in summary."""
        store.record(make_result("champion", prediction=5.5, request_id="req-001"))
        store.record_outcome("req-001", 6.0)

        summary = store.get_summary()
        assert summary["total_outcomes"] == 1

    def test_accuracy_computation(self, store):
        """When outcomes are recorded, MAE should be computed."""
        store.record(make_result("champion", prediction=5.0, request_id="req-1"))
        store.record(make_result("champion", prediction=6.0, request_id="req-2"))
        store.record_outcome("req-1", 5.5)  # error = 0.5
        store.record_outcome("req-2", 5.0)  # error = 1.0

        summary = store.get_summary()
        accuracy = summary["champion"]["accuracy"]
        assert accuracy["count"] == 2
        assert accuracy["mae"] == pytest.approx(0.75, abs=0.01)

    def test_predictions_limited_to_50(self, store):
        """Summary should only include last 50 predictions per model."""
        for i in range(100):
            store.record(make_result("champion", prediction=float(i), request_id=f"r-{i}"))

        summary = store.get_summary()
        assert len(summary["champion"]["predictions"]) == 50
        assert len(summary["champion"]["timestamps"]) == 50

    def test_clear(self, store):
        """clear() should reset all data."""
        store.record(make_result("champion"))
        store.record_outcome("req-001", 5.0)
        store.clear()

        summary = store.get_summary()
        assert summary["total_requests"] == 0
        assert summary["total_outcomes"] == 0

    def test_thread_safety(self, store):
        """Concurrent writes should not cause data corruption."""
        errors = []

        def writer(model, count):
            try:
                for i in range(count):
                    store.record(make_result(model, request_id=f"{model}-{i}"))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("champion", 500)),
            threading.Thread(target=writer, args=("challenger", 500)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        summary = store.get_summary()
        assert summary["total_requests"] == 100  # max_size=100

    def test_traffic_split(self, store):
        """Traffic split should reflect actual routing."""
        for _ in range(7):
            store.record(make_result("champion"))
        for _ in range(3):
            store.record(make_result("challenger"))

        summary = store.get_summary()
        assert summary["traffic_split"]["champion"] == 7
        assert summary["traffic_split"]["challenger"] == 3
