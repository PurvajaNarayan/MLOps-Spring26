"""
MetricsStore — In-memory buffer for dashboard metrics.

Thread-safe deque-based storage with rolling statistics computation
for latency, prediction distributions, and traffic counts.
"""

from collections import deque
import numpy as np
import threading


class MetricsStore:
    def __init__(self, max_size=1000):
        self.predictions = deque(maxlen=max_size)
        self.outcomes = {}
        self.lock = threading.Lock()

    def record(self, result):
        """Record a prediction result."""
        with self.lock:
            self.predictions.append(result)

    def record_outcome(self, request_id, actual):
        """Record a ground-truth outcome for a previous prediction."""
        with self.lock:
            self.outcomes[request_id] = actual

    def get_summary(self):
        """
        Compute rolling summary statistics for the dashboard.

        Returns a dict with champion/challenger stats, traffic split,
        and accuracy info if outcomes are available.
        """
        with self.lock:
            champion_preds = [p for p in self.predictions if p["model"] == "champion"]
            challenger_preds = [p for p in self.predictions if p["model"] == "challenger"]

            summary = {
                "champion": self._model_stats(champion_preds),
                "challenger": self._model_stats(challenger_preds),
                "traffic_split": {
                    "champion": len(champion_preds),
                    "challenger": len(challenger_preds),
                },
                "total_requests": len(self.predictions),
                "total_outcomes": len(self.outcomes),
            }

            # Add accuracy metrics if we have outcomes
            if self.outcomes:
                summary["champion"]["accuracy"] = self._compute_accuracy(
                    champion_preds
                )
                summary["challenger"]["accuracy"] = self._compute_accuracy(
                    challenger_preds
                )

            return summary

    def _model_stats(self, preds):
        """Compute stats for a single model's predictions."""
        if not preds:
            return {
                "count": 0,
                "avg_latency": 0,
                "p95_latency": 0,
                "predictions": [],
                "timestamps": [],
                "latencies": [],
            }

        latencies = [p["latency_ms"] for p in preds]
        return {
            "count": len(preds),
            "avg_latency": round(float(np.mean(latencies)), 2),
            "p95_latency": round(float(np.percentile(latencies, 95)), 2),
            "predictions": [p["prediction"][0] for p in preds[-50:]],
            "timestamps": [p["timestamp"] for p in preds[-50:]],
            "latencies": [p["latency_ms"] for p in preds[-50:]],
        }

    def _compute_accuracy(self, preds):
        """Compute MAE for predictions that have ground truth."""
        errors = []
        for p in preds:
            request_id = p.get("request_id")
            if request_id and request_id in self.outcomes:
                actual = self.outcomes[request_id]
                predicted = p["prediction"][0]
                errors.append(abs(actual - predicted))

        if not errors:
            return {"mae": None, "count": 0}

        return {
            "mae": round(float(np.mean(errors)), 4),
            "count": len(errors),
        }

    def clear(self):
        """Clear all stored data."""
        with self.lock:
            self.predictions.clear()
            self.outcomes.clear()
