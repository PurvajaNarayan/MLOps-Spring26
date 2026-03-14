"""
Traffic simulator — sends fake prediction requests to test the system.

Usage:
    python scripts/simulate_traffic.py [--url URL] [--count N] [--rps RPS]
"""

import requests
import random
import time
import argparse
import sys
import os

# Wine quality feature ranges (based on the red wine dataset)
FEATURE_RANGES = {
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


def generate_sample():
    """Generate a random wine sample within realistic feature ranges."""
    return {
        feature: round(random.uniform(lo, hi), 4)
        for feature, (lo, hi) in FEATURE_RANGES.items()
    }


def simulate(url, count, rps, verbose=True):
    """Send prediction requests to the API."""
    delay = 1.0 / rps if rps > 0 else 0
    results = {"champion": 0, "challenger": 0, "errors": 0}
    latencies = {"champion": [], "challenger": []}

    print(f"🚀 Sending {count} requests to {url}/predict at {rps} req/s\n")

    for i in range(count):
        sample = generate_sample()
        try:
            resp = requests.post(
                f"{url}/predict",
                json={"features": sample},
                timeout=10,
            )
            if resp.status_code != 200:
                raise Exception(f"HTTP {resp.status_code}: {resp.text[:100]}")
            data = resp.json()
            model = data.get("model", "unknown")
            latency = data.get("latency_ms", 0)

            results[model] = results.get(model, 0) + 1
            if model in latencies:
                latencies[model].append(latency)

            if verbose:
                pred = data.get("prediction", [None])[0]
                print(
                    f"  [{i+1:4d}/{count}] {model:>10s} | "
                    f"pred={pred:6.3f} | latency={latency:6.2f}ms"
                )
        except Exception as e:
            results["errors"] += 1
            if verbose:
                print(f"  [{i+1:4d}/{count}] ERROR: {e}")

        if delay > 0:
            time.sleep(delay)

    # Summary
    print(f"\n{'='*50}")
    print(f"  Total requests:  {count}")
    print(f"  Champion:        {results['champion']}")
    print(f"  Challenger:      {results['challenger']}")
    print(f"  Errors:          {results['errors']}")
    if latencies["champion"]:
        avg_c = sum(latencies["champion"]) / len(latencies["champion"])
        print(f"  Champion avg:    {avg_c:.2f} ms")
    if latencies["challenger"]:
        avg_ch = sum(latencies["challenger"]) / len(latencies["challenger"])
        print(f"  Challenger avg:  {avg_ch:.2f} ms")
    print(f"{'='*50}")

    return results


def main():
    parser = argparse.ArgumentParser(description="A/B test traffic simulator")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--count", type=int, default=500, help="Number of requests")
    parser.add_argument("--rps", type=float, default=10, help="Requests per second")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-request output")
    args = parser.parse_args()

    simulate(args.url, args.count, args.rps, verbose=not args.quiet)


if __name__ == "__main__":
    main()
