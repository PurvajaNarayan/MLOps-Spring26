# Lab 3 — Real-Time Fraud Detection with Apache Beam

A machine-learning fraud-detection pipeline built on **Apache Beam** (DirectRunner). Raw credit-card transactions are parsed, windowed into user sessions, enriched with session-level features, scored by a trained Random Forest classifier, and routed to either a flagged or clean output sink.

---

## Project Structure

```
Lab3-FraudDetection/
├── pipeline.py               # Main Apache Beam pipeline
├── simulate_transactions.py  # Synthetic data generator
├── train_model.py            # Offline model training
├── transactions.jsonl        # Generated transaction data
├── model/
│   └── fraud_model.pkl       # Serialized Random Forest model
├── output/
│   ├── flagged_transactions-* # Beam output: fraud suspects
│   ├── clean_transactions-*   # Beam output: legitimate transactions
│   └── dead_letters-*         # Beam output: malformed records
└── transforms/               # Modular Apache Beam DoFns
    ├── parse.py              # ParseTransaction
    ├── features.py           # ExtractSessionFeatures
    ├── model.py              # ScoreWithModel
    └── router.py             # RouteTransaction
```

---

## Beyond the Reference Lab

The [course reference notebook](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Data_Labs/Apache_Beam_Labs) (`Try_Apache_Beam_Python.ipynb`) is a single-cell **word count** exercise on a Shakespeare text file — a "Hello World" introduction to Beam. This lab goes significantly further:

| Feature | Reference Lab | This Lab |
|---|---|---|
| **Problem** | Word count on a text file | End-to-end fraud detection on financial transactions |
| **Data** | Pre-made `.txt` downloaded from GCS | Synthetically generated JSONL via `simulate_transactions.py` |
| **Windowing** | ❌ None | ✅ Session windows (30-min inactivity gap) |
| **Event-time** | ❌ None | ✅ `TimestampedValue` for correct windowing |
| **Multiple outputs** | ❌ Single output | ✅ flagged / clean / dead-letter sinks |
| **Dead letter queue** | ❌ | ✅ Malformed records isolated without crashing |
| **ML model** | ❌ None | ✅ Random Forest trained offline, loaded in `DoFn.setup()` |
| **Feature engineering** | ❌ | ✅ 9 session-level features computed inside the pipeline |
| **Code structure** | ~15 lines in one notebook cell | Modular `transforms/` package with 4 `DoFn` classes |

In short — the reference lab teaches **what Beam is**; this lab demonstrates **how to use Beam in a real MLOps pipeline**.

## Apache Beam — Core Concepts Used

Apache Beam is a unified programming model for both batch and streaming data pipelines. This lab uses the **DirectRunner** (local execution) to demonstrate Beam fundamentals.

### 1. `PCollection`
Every data transformation in Beam operates on a `PCollection` (parallel collection) — an immutable, distributed dataset. The pipeline chains `PCollection → transform → PCollection` steps using the `|` and `>>` operators.

```python
raw = p | "ReadTransactions" >> beam.io.ReadFromText("transactions.jsonl")
```

### 2. `DoFn` — The Building Block of Transforms
A `DoFn` (Do Function) defines the per-element processing logic. Each transform in `transforms/` is its own `DoFn` class:

| Class | File | Role |
|---|---|---|
| `ParseTransaction` | `transforms/parse.py` | JSON parse + field validation |
| `ExtractSessionFeatures` | `transforms/features.py` | Session-level feature engineering |
| `ScoreWithModel` | `transforms/model.py` | ML model inference |
| `RouteTransaction` | `transforms/router.py` | Tag-based output routing |

`DoFn` supports lifecycle hooks — `setup()` is called **once per worker** before `process()`, ideal for loading a model:

```python
class ScoreWithModel(beam.DoFn):
    def setup(self):
        with open("model/fraud_model.pkl", "rb") as f:
            self.model = pickle.load(f)   # loaded once, reused across elements

    def process(self, element):
        ...
        yield element
```

### 3. `beam.ParDo` — Applying a DoFn
`beam.ParDo` applies a `DoFn` to every element in a `PCollection` in parallel:

```python
parsed = raw | "Parse" >> beam.ParDo(ParseTransaction())
```

### 4. Tagged Outputs — Multiple Output Branches
`with_outputs()` lets a single `DoFn` emit to **multiple named PCollections**. This is used for both the dead-letter queue and the final flagged/clean split:

```python
parse_result = raw | "Parse" >> beam.ParDo(ParseTransaction()).with_outputs(
    "dead_letter", main="valid"
)
parsed       = parse_result["valid"]
dead_letters = parse_result["dead_letter"]
```

Inside the `DoFn`, tagged output is produced with:
```python
yield beam.pvalue.TaggedOutput("dead_letter", element)
```

### 5. Session Windows
Session windowing groups elements by key into **activity sessions** separated by a gap of inactivity. Here, transactions from the same user within a 30-minute idle gap are grouped into one session:

```python
session_features = (
    parsed
    | "AddTimestamp"   >> beam.ParDo(AddTimestamp())
    | "KeyByUser"      >> beam.Map(lambda t: (t["user_id"], t))
    | "SessionWindows" >> beam.WindowInto(window.Sessions(gap_size=1800))  # 30 min
    | "GroupByUser"    >> beam.GroupByKey()
    | "ExtractFeatures" >> beam.ParDo(ExtractSessionFeatures())
)
```

**Why sessions?** Fraud often manifests as a burst of high-value purchases from different locations within a short window. Session windows capture exactly this behavior.

`AddTimestamp` assigns each transaction's `timestamp` field as the Beam event time, enabling correct windowing:
```python
yield beam.window.TimestampedValue(element, ts.timestamp())
```

### 6. `beam.Map` — Simple 1-to-1 Transforms
`beam.Map` is a lightweight shorthand for a `DoFn` with a single output. Used for keying and serialization:

```python
| "KeyByUser" >> beam.Map(lambda t: (t["user_id"], t))
| "SerializeFlagged" >> beam.Map(safe_json)
```

### 7. I/O — Reading and Writing
Beam provides built-in connectors. This lab uses the text file connector:

```python
# Source
raw = p | "ReadTransactions" >> beam.io.ReadFromText("transactions.jsonl")

# Sinks
routed["flagged"] | "WriteFlagged" >> beam.io.WriteToText("output/flagged_transactions")
routed["clean"]   | "WriteClean"   >> beam.io.WriteToText("output/clean_transactions")
dead_letters      | "WriteDeadLetter" >> beam.io.WriteToText("output/dead_letters")
```

---

## Full Pipeline Flow

```
transactions.jsonl
       │
       ▼
  ReadFromText          ← I/O Source
       │
       ▼
 ParseTransaction       ← Validate JSON, tag malformed as dead_letters
       │
   ┌───┴────────────────────────────┐
   ▼                                ▼
 valid PCollection           dead_letter PCollection
   │                                │
   ▼                                ▼
 AddTimestamp               WriteToText (dead_letters)
   │
   ▼
 KeyByUser  →  SessionWindows (30 min gap)  →  GroupByKey
   │
   ▼
 ExtractSessionFeatures     ← Compute 9 session-level features
   │
   ▼
 ScoreWithModel             ← Random Forest predict_proba, threshold 0.7
   │
   ▼
 RouteTransaction
   │
   ├──▶ flagged PCollection → WriteToText (flagged_transactions)
   └──▶ clean   PCollection → WriteToText (clean_transactions)
```

---

## Session Features Engineered

| Feature | Description |
|---|---|
| `session_txn_count` | Number of transactions in the session |
| `total_amount` | Sum of all transaction amounts |
| `avg_amount` | Mean transaction amount |
| `max_amount` | Single largest transaction |
| `amount_std` | Standard deviation of amounts |
| `unique_categories` | Count of distinct merchant categories |
| `unique_locations` | Count of distinct geographic locations |
| `location_switches` | Number of times location changed between consecutive transactions |
| `high_value_txn_ratio` | Fraction of transactions exceeding $1,000 |

---

## ML Model

- **Algorithm**: `RandomForestClassifier` (100 estimators, `class_weight="balanced"`)
- **Training data**: Session features aggregated from `transactions.jsonl`
- **Features**: The 10 columns listed in `ScoreWithModel.FEATURE_COLS`
- **Fraud threshold**: `fraud_prob > 0.7` → flagged
- **Saved artifact**: `model/fraud_model.pkl`

---

## Setup & Running

### Prerequisites
```bash
pip install apache-beam numpy pandas scikit-learn faker
```

### Step 1 — Generate synthetic transactions
```bash
python simulate_transactions.py
# Generates transactions.jsonl with ~200 users, ~20% fraud users
```

### Step 2 — Train the model
```bash
python train_model.py
# Saves model/fraud_model.pkl
```

### Step 3 — Run the Beam pipeline
```bash
python pipeline.py
```

Output files will be written to the `output/` directory:
- `output/flagged_transactions-*` — suspected fraud
- `output/clean_transactions-*` — legitimate transactions
- `output/dead_letters-*` — malformed / unparseable records

---

## Dead Letter Queue

Any record that fails JSON parsing or is missing required fields (`transaction_id`, `user_id`, `amount`, `timestamp`) is tagged and written to the dead-letter sink rather than crashing the pipeline. This pattern ensures **fault tolerance** — bad data is isolated, logged, and can be reprocessed later without affecting the main flow.

---

## Key Beam Concepts Summary

| Concept | Used For |
|---|---|
| `PCollection` | Distributed, immutable data at each pipeline stage |
| `DoFn` + `ParDo` | Per-element transformation logic |
| `TaggedOutput` | Splitting a stream into multiple named branches |
| `SessionWindows` | Grouping user transactions by session activity |
| `TimestampedValue` | Assigning event-time for windowing |
| `GroupByKey` | Aggregating keyed elements within a window |
| `beam.Map` | Lightweight 1-to-1 transformations |
| `ReadFromText` / `WriteToText` | File-based I/O connectors |
| `DoFn.setup()` | One-time per-worker initialization (model loading) |

---
