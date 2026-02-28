import apache_beam as beam
import numpy as np

class ExtractSessionFeatures(beam.DoFn):
    def process(self, element):
        user_id, transactions = element
        txns = list(transactions)

        amounts    = [t["amount"] for t in txns]
        categories = [t["merchant_category"] for t in txns]
        locations  = [t["location"] for t in txns]

        features = {
            "user_id": user_id,
            "session_txn_count":      len(txns),
            "total_amount":           sum(amounts),
            "avg_amount":             np.mean(amounts),
            "max_amount":             max(amounts),
            "amount_std":             np.std(amounts) if len(amounts) > 1 else 0,
            "unique_categories":      len(set(categories)),
            "unique_locations":       len(set(locations)),
            "location_switches":      sum(1 for i in range(1, len(locations))
                                         if locations[i] != locations[i-1]),
            # High location switching = suspicious
            "high_value_txn_ratio":   sum(1 for a in amounts if a > 1000) / len(amounts),
        }

        # Attach each original transaction + its session features for scoring
        for txn in txns:
            yield {**txn, **features}