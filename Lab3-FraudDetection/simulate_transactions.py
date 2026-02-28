import json
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)

def generate_normal_transaction(user_id, timestamp):
    return {
        "transaction_id": fake.uuid4(),
        "user_id": user_id,
        "amount": round(random.uniform(5, 300), 2),        # small everyday amounts
        "merchant_category": random.choice(["grocery", "restaurant", "pharmacy"]),
        "location": "Boston",                               # stays in one city
        "timestamp": timestamp.isoformat(),
        "is_fraud": False
    }

def generate_fraud_transaction(user_id, timestamp):
    return {
        "transaction_id": fake.uuid4(),
        "user_id": user_id,
        "amount": round(random.uniform(1500, 5000), 2),    # large amounts
        "merchant_category": random.choice(["electronics", "travel", "luxury"]),
        "location": random.choice(["Miami", "Las Vegas", "New York", "Chicago"]),  # location hopping
        "timestamp": timestamp.isoformat(),
        "is_fraud": True
    }

transactions = []
base_time = datetime.utcnow() - timedelta(hours=6)

# Generate 200 users with spread-out timestamps
for user_num in range(1, 201):
    user_id = f"user_{user_num}"
    is_fraud_user = random.random() < 0.20   # ~20% fraud users → ~5-8% fraud transactions

    # Each user gets 3-8 transactions spread over a few hours
    n_txns = random.randint(3, 8)
    user_start = base_time + timedelta(minutes=random.randint(0, 300))

    for i in range(n_txns):
        ts = user_start + timedelta(minutes=i * random.randint(2, 20))

        if is_fraud_user and i >= n_txns - 3:
            # Last 2-3 transactions are fraudulent (rapid high-value purchases)
            txn = generate_fraud_transaction(user_id, ts)
        else:
            txn = generate_normal_transaction(user_id, ts)

        transactions.append(txn)

random.shuffle(transactions)

with open("transactions.jsonl", "w") as f:
    for t in transactions:
        f.write(json.dumps(t) + "\n")

fraud_count = sum(1 for t in transactions if t["is_fraud"])
print(f"Generated {len(transactions)} transactions")
print(f"Fraud: {fraud_count} ({fraud_count/len(transactions):.1%})")
print(f"Clean: {len(transactions) - fraud_count}")