"""
ABTracker — Logs predictions and outcomes to MLflow as time-series metrics.

Uses background threads for fire-and-forget logging so prediction latency
isn't affected by MLflow write times.
"""

import os
import mlflow
from threading import Thread


def _check_mlflow_server(uri, timeout=2):
    """Quick check if an MLflow server is reachable."""
    if not uri.startswith("http"):
        return True  # Local file URI, always works
    try:
        import urllib.request
        req = urllib.request.Request(f"{uri}/health", method="GET")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except Exception:
        return False


class ABTracker:
    def __init__(self, experiment_name="ab-test-wine-quality"):
        self.experiment_name = experiment_name
        self.champion_step = 0
        self.challenger_step = 0
        self.enabled = True

        try:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")

            # If a remote URI is set, check if it's reachable
            if tracking_uri and tracking_uri.startswith("http"):
                if _check_mlflow_server(tracking_uri):
                    mlflow.set_tracking_uri(tracking_uri)
                    print(f"✅ MLflow server connected: {tracking_uri}")
                else:
                    # Fall back to local file-based tracking
                    print(f"⚠️  MLflow server at {tracking_uri} not reachable, using local tracking")
                    mlflow.set_tracking_uri("mlruns")
            else:
                # Use local file-based tracking (no server needed)
                mlflow.set_tracking_uri("mlruns")
                print("📁 Using local MLflow tracking (mlruns/)")

            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"⚠️  MLflow tracking disabled: {e}")
            self.enabled = False

    def log_prediction(self, result):
        """Fire-and-forget logging in a background thread."""
        if not self.enabled:
            return
        Thread(target=self._log_prediction, args=(result,), daemon=True).start()

    def _log_prediction(self, result):
        """Log a single prediction as a step-level metric."""
        try:
            model = result["model"]
            with mlflow.start_run(run_name=f"ab-test-{model}", nested=True):
                step = self.champion_step if model == "champion" else self.challenger_step

                mlflow.log_metric(f"{model}_latency_ms", result["latency_ms"], step=step)
                mlflow.log_metric(
                    f"{model}_prediction", result["prediction"][0], step=step
                )

                if model == "champion":
                    self.champion_step += 1
                else:
                    self.challenger_step += 1
        except Exception as e:
            # Silently fail — don't crash the app because of logging issues
            pass

    def log_outcome(self, request_id, actual_value):
        """Log a ground-truth outcome for a previous prediction."""
        if not self.enabled:
            return
        Thread(
            target=self._log_outcome,
            args=(request_id, actual_value),
            daemon=True,
        ).start()

    def _log_outcome(self, request_id, actual_value):
        """Log outcome to MLflow."""
        try:
            with mlflow.start_run(run_name=f"feedback-{request_id[:8]}", nested=True):
                mlflow.log_metric("actual_value", actual_value)
                mlflow.log_param("request_id", request_id)
        except Exception as e:
            pass
