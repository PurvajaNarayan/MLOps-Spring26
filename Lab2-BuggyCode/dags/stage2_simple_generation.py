"""
Stage 2: Synthetic Data Generation DAG
Generates buggy Python code samples using free LLM models in parallel.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys
import pandas as pd

# Add plugins to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugins.utils.code_generator import generate_batch


# Define paths
DATA_DIR = '/opt/airflow/data'
SYNTHETIC_DIR = os.path.join(DATA_DIR, 'synthetic_batches')


# Default arguments
default_args = {
    'owner': 'purvaja',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Define the DAG
dag = DAG(
    'stage2_simple_generation',
    default_args=default_args,
    description='Generate synthetic buggy code samples using free LLMs',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['bug-detection', 'stage2', 'generation'],
)


def generate_and_save_batch(batch_id: int, **context):
    """
    Generate a batch of synthetic samples and save to CSV.
    
    Args:
        batch_id: Identifier for this batch (0-4 for 5 batches)
    """
    print("=" * 50)
    print(f"GENERATING BATCH {batch_id}")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    
    # Generate samples
    samples = generate_batch(batch_id=batch_id, samples_per_batch=10)
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    # Save to CSV
    output_path = os.path.join(SYNTHETIC_DIR, f"synthetic_batch_{batch_id}.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Batch {batch_id} saved to {output_path}")
    print(f"  Samples: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print("=" * 50)
    
    return f"Batch {batch_id} complete"


# Create 5 parallel tasks (one for each batch)
batch_tasks = []

for batch_id in range(5):  # 5 batches = 50 samples total
    task = PythonOperator(
        task_id=f'generate_batch_{batch_id}',
        python_callable=generate_and_save_batch,
        op_kwargs={'batch_id': batch_id},
        dag=dag,
    )
    batch_tasks.append(task)


# All tasks run in parallel (no dependencies between them)
# This is the power of Airflow - parallel execution!