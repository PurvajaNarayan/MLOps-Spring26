import json
import os
import pickle
import numpy as np
import apache_beam as beam
from apache_beam import window
from apache_beam.options.pipeline_options import PipelineOptions

# ── DoFns ─────────────────────────────────────────────────────────────────────

class ParseTransaction(beam.DoFn):
    def process(self, element):
        try:
            txn = json.loads(element)
            required = ["transaction_id", "user_id", "amount", "timestamp"]
            if not all(k in txn for k in required):
                yield beam.pvalue.TaggedOutput("dead_letter", element)
                return
            yield txn
        except Exception:
            yield beam.pvalue.TaggedOutput("dead_letter", element)


class AddTimestamp(beam.DoFn):
    def process(self, element):
        import apache_beam as beam
        from datetime import datetime
        ts = datetime.fromisoformat(element["timestamp"])
        yield beam.window.TimestampedValue(element, ts.timestamp())


class ExtractSessionFeatures(beam.DoFn):
    def process(self, element):
        user_id, transactions = element
        txns = list(transactions)

        amounts   = [t["amount"] for t in txns]
        cats      = [t["merchant_category"] for t in txns]
        locations = [t["location"] for t in txns]

        session_feats = {
            "session_txn_count":    len(txns),
            "total_amount":         sum(amounts),
            "avg_amount":           np.mean(amounts),
            "max_amount":           max(amounts),
            "amount_std":           np.std(amounts) if len(amounts) > 1 else 0,
            "unique_categories":    len(set(cats)),
            "unique_locations":     len(set(locations)),
            "location_switches":    sum(1 for i in range(1, len(locations))
                                        if locations[i] != locations[i - 1]),
            "high_value_txn_ratio": sum(1 for a in amounts if a > 1000) / len(amounts),
        }

        for txn in txns:
            yield {**txn, **session_feats}


class ScoreWithModel(beam.DoFn):
    FEATURE_COLS = [
        "amount", "session_txn_count", "total_amount",
        "avg_amount", "max_amount", "amount_std",
        "unique_categories", "unique_locations",
        "location_switches", "high_value_txn_ratio"
    ]

    def setup(self):
        with open("model/fraud_model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def process(self, element):
        import pandas as pd
        features = pd.DataFrame([[element.get(col, 0) for col in self.FEATURE_COLS]], columns=self.FEATURE_COLS)
        fraud_prob = self.model.predict_proba(features)[0][1]
        element["fraud_score"]     = round(float(fraud_prob), 4)
        element["fraud_predicted"] = bool(fraud_prob > 0.7)  # numpy.bool_ → Python bool
        yield element


class RouteTransaction(beam.DoFn):
    def process(self, element):
        if element["fraud_predicted"]:
            yield beam.pvalue.TaggedOutput("flagged", element)
        else:
            yield beam.pvalue.TaggedOutput("clean", element)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run():
    os.makedirs("output", exist_ok=True)

    # Make all numpy types JSON-safe
    def safe_json(obj):
        def convert(v):
            if isinstance(v, (np.integer,)): return int(v)
            if isinstance(v, (np.floating,)): return float(v)
            if isinstance(v, (np.bool_,)): return bool(v)
            return v
        return json.dumps({k: convert(v) for k, v in obj.items()})

    options = PipelineOptions(runner="DirectRunner", streaming=False)

    with beam.Pipeline(options=options) as p:

        # 1. Read
        raw = p | "ReadTransactions" >> beam.io.ReadFromText("transactions.jsonl")

        # 2. Parse
        parse_result = raw | "Parse" >> beam.ParDo(ParseTransaction()).with_outputs(
            "dead_letter", main="valid"
        )
        parsed     = parse_result["valid"]
        dead_letters = parse_result["dead_letter"]

        # 3. Session windowing + feature engineering
        session_features = (
            parsed
            | "AddTimestamp"    >> beam.ParDo(AddTimestamp())
            | "KeyByUser"       >> beam.Map(lambda t: (t["user_id"], t))
            | "SessionWindows"  >> beam.WindowInto(window.Sessions(gap_size=1800))
            | "GroupByUser"     >> beam.GroupByKey()
            | "ExtractFeatures" >> beam.ParDo(ExtractSessionFeatures())
        )

        # 4. Score with ML model
        scored = session_features | "ScoreModel" >> beam.ParDo(ScoreWithModel())

        # 5. Route flagged vs clean
        routed = scored | "Route" >> beam.ParDo(RouteTransaction()).with_outputs(
            "flagged", "clean"
        )

        # 6. Write flagged → review queue (file sink)
        (
            routed["flagged"]
            | "SerializeFlagged" >> beam.Map(safe_json)
            | "WriteFlagged"     >> beam.io.WriteToText("output/flagged_transactions")
        )

        # 7. Write clean → storage
        (
            routed["clean"]
            | "SerializeClean" >> beam.Map(safe_json)
            | "WriteClean"     >> beam.io.WriteToText("output/clean_transactions")
        )

        # 8. Dead letter queue
        (
            dead_letters
            | "WriteDeadLetter" >> beam.io.WriteToText("output/dead_letters")
        )


if __name__ == "__main__":
    run()