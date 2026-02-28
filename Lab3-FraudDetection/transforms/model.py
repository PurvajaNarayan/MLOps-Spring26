import apache_beam as beam
import pickle
import numpy as np

class ScoreWithModel(beam.DoFn):

    FEATURE_COLS = [
        "amount", "session_txn_count", "total_amount",
        "avg_amount", "max_amount", "amount_std",
        "unique_categories", "unique_locations",
        "location_switches", "high_value_txn_ratio"
    ]

    def setup(self):
        # Called once per worker — load model into memory
        with open("model/fraud_model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def process(self, element):
        features = np.array([[element.get(col, 0) for col in self.FEATURE_COLS]])
        fraud_prob = self.model.predict_proba(features)[0][1]

        element["fraud_score"]    = round(fraud_prob, 4)
        element["fraud_predicted"] = fraud_prob > 0.7  # threshold
        yield element