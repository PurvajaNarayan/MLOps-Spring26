"""
Feedback sender — sends ground truth labels to previous prediction requests.

Usage:
    python scripts/send_feedback.py [--url URL] [--count N]
"""

import requests
import random
import argparse


def send_feedback(url, count=50, verbose=True):
    """
    First collect request IDs by making predictions,
    then send ground truth feedback for each.
    """
    print(f"🔄 Step 1: Making {count} predictions to collect request IDs...\n")

    # Wine feature ranges for sampling
    feature_ranges = {
        "fixed acidity": (4.6, 15.9),
        "volatile acidity": (0.12, 1.58),
        "citric acid": (0.0, 1.0),
        "residual sugar": (0.9, 15.5),
        "chlorides": (0.012, 0.611),
        "free sulfur dioxide": (1.0, 72.0),
        "total sulfur dioxide": (6.0, 289.0),
        "density": (0.9901, 1.0037),
        "pH": (2.74, 4.01),
        "sulphates": (0.33, 2.0),
        "alcohol": (8.4, 14.9),
    }

    predictions = []
    for i in range(count):
        sample = {k: round(random.uniform(lo, hi), 4) for k, (lo, hi) in feature_ranges.items()}
        try:
            resp = requests.post(f"{url}/predict", json={"features": sample}, timeout=10)
            data = resp.json()
            predictions.append(data)
        except Exception as e:
            print(f"  [{i+1}] Prediction error: {e}")

    print(f"\n📬 Step 2: Sending ground truth feedback for {len(predictions)} predictions...\n")

    success = 0
    for i, pred in enumerate(predictions):
        request_id = pred.get("request_id")
        if not request_id:
            continue

        # Simulate ground truth: wine quality is typically 3-8
        actual_quality = round(random.uniform(3, 8), 1)

        try:
            resp = requests.post(
                f"{url}/feedback",
                json={"request_id": request_id, "actual": actual_quality},
                timeout=10,
            )
            result = resp.json()
            if verbose:
                predicted = pred.get("prediction", [None])[0]
                model = pred.get("model", "?")
                error = abs(actual_quality - predicted) if predicted else None
                print(
                    f"  [{i+1:3d}] {model:>10s} | "
                    f"predicted={predicted:5.2f} actual={actual_quality:4.1f} "
                    f"error={error:5.2f}"
                )
            success += 1
        except Exception as e:
            print(f"  [{i+1:3d}] Feedback error: {e}")

    print(f"\n✅ Sent feedback for {success}/{len(predictions)} predictions")


def main():
    parser = argparse.ArgumentParser(description="Send ground truth feedback")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--count", type=int, default=50, help="Number of predictions to make + feedback to send")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-request output")
    args = parser.parse_args()

    send_feedback(args.url, args.count, verbose=not args.quiet)


if __name__ == "__main__":
    main()
