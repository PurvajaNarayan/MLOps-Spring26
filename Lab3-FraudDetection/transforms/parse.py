import apache_beam as beam
import json
from datetime import datetime

class ParseTransaction(beam.DoFn):
    def process(self, element):
        try:
            txn = json.loads(element)
            # Validate required fields
            required = ["transaction_id", "user_id", "amount", "timestamp"]
            if not all(k in txn for k in required):
                return  # drop malformed records

            # Attach event timestamp for windowing
            txn["event_time"] = txn["timestamp"]
            yield txn

        except Exception as e:
            yield beam.pvalue.TaggedOutput("dead_letter", element)