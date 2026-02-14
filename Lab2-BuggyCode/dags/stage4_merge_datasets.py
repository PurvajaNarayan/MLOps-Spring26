"""
Stage 4: Merge Datasets DAG
Combines base CodeXGLUE data with synthetic filtered data and creates splits.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Add plugins to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Define paths
DATA_DIR = '/opt/airflow/data'
BASE_DATA_DIR = os.path.join(DATA_DIR, 'base')
SYNTHETIC_FILTERED_PATH = os.path.join(DATA_DIR, 'synthetic_filtered.csv')
FINAL_TRAIN_PATH = os.path.join(DATA_DIR, 'final_train.csv')
FINAL_VAL_PATH = os.path.join(DATA_DIR, 'final_val.csv')
FINAL_TEST_PATH = os.path.join(DATA_DIR, 'final_test.csv')
DATASET_STATS_PATH = os.path.join(DATA_DIR, 'dataset_stats.csv')


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
    'stage4_merge_datasets',
    default_args=default_args,
    description='Merge base and synthetic data, create train/val/test splits',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['bug-detection', 'stage4', 'data-prep'],
)


def task_load_and_merge(**context):
    """
    Task 1: Load base and synthetic data, then merge them.
    """
    print("=" * 50)
    print("TASK 1: Loading and Merging Datasets")
    print("=" * 50)
    
    # Load base training data
    base_train_path = os.path.join(BASE_DATA_DIR, 'train.csv')
    print(f"Loading base training data from {base_train_path}...")
    base_df = pd.read_csv(base_train_path)
    print(f"  Base samples: {len(base_df)}")
    
    # Load synthetic filtered data
    print(f"\nLoading synthetic data from {SYNTHETIC_FILTERED_PATH}...")
    synthetic_df = pd.read_csv(SYNTHETIC_FILTERED_PATH)
    print(f"  Synthetic samples: {len(synthetic_df)}")
    
    # Ensure both datasets have the same columns
    # Base dataset should have: func, target
    # Synthetic should have: func, target, bug_type, source, sample_id, batch_id
    
    # Keep only essential columns for base data
    if 'func' in base_df.columns and 'target' in base_df.columns:
        base_df = base_df[['func', 'target']].copy()
        base_df['source'] = 'codexglue'
    else:
        raise ValueError("Base dataset missing required columns: func, target")
    
    # Keep essential columns for synthetic data
    if 'func' in synthetic_df.columns and 'target' in synthetic_df.columns:
        synthetic_df = synthetic_df[['func', 'target', 'source']].copy()
    else:
        raise ValueError("Synthetic dataset missing required columns: func, target")
    
    # Combine datasets
    print("\nCombining datasets...")
    combined_df = pd.concat([base_df, synthetic_df], ignore_index=True)
    
    # Shuffle the data
    print("Shuffling combined data...")
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✓ Combined dataset created:")
    print(f"  Total samples: {len(combined_df)}")
    print(f"  Base samples: {len(base_df)}")
    print(f"  Synthetic samples: {len(synthetic_df)}")
    print(f"  Buggy samples: {combined_df['target'].sum()}")
    print(f"  Clean samples: {len(combined_df) - combined_df['target'].sum()}")
    print("=" * 50)
    
    # Save temporarily
    temp_path = os.path.join(DATA_DIR, 'combined_data.csv')
    combined_df.to_csv(temp_path, index=False)
    
    return f"Merged {len(base_df)} base + {len(synthetic_df)} synthetic = {len(combined_df)} total"


def task_create_splits(**context):
    """
    Task 2: Create stratified train/val/test splits (80/10/10).
    """
    print("=" * 50)
    print("TASK 2: Creating Train/Val/Test Splits")
    print("=" * 50)
    
    # Load combined data
    temp_path = os.path.join(DATA_DIR, 'combined_data.csv')
    df = pd.read_csv(temp_path)
    
    print(f"Creating stratified splits from {len(df)} samples...")
    print("  Target distribution:")
    print(f"    Buggy (1): {df['target'].sum()}")
    print(f"    Clean (0): {len(df) - df['target'].sum()}")
    
    # First split: 80% train, 20% temp (which will become val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['target']
    )
    
    # Second split: Split the 20% into 10% val and 10% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,  # 50% of 20% = 10% of total
        random_state=42,
        stratify=temp_df['target']
    )
    
    # Save splits
    train_df.to_csv(FINAL_TRAIN_PATH, index=False)
    val_df.to_csv(FINAL_VAL_PATH, index=False)
    test_df.to_csv(FINAL_TEST_PATH, index=False)
    
    print(f"\n✓ Splits created and saved:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"    Buggy: {train_df['target'].sum()} ({train_df['target'].sum()/len(train_df)*100:.1f}%)")
    print(f"    Clean: {len(train_df) - train_df['target'].sum()}")
    
    print(f"\n  Validation: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"    Buggy: {val_df['target'].sum()} ({val_df['target'].sum()/len(val_df)*100:.1f}%)")
    print(f"    Clean: {len(val_df) - val_df['target'].sum()}")
    
    print(f"\n  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"    Buggy: {test_df['target'].sum()} ({test_df['target'].sum()/len(test_df)*100:.1f}%)")
    print(f"    Clean: {len(test_df) - test_df['target'].sum()}")
    
    print("\n✓ Files saved:")
    print(f"  {FINAL_TRAIN_PATH}")
    print(f"  {FINAL_VAL_PATH}")
    print(f"  {FINAL_TEST_PATH}")
    print("=" * 50)
    
    return f"Created splits: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test"


def task_generate_statistics(**context):
    """
    Task 3: Generate dataset statistics.
    """
    print("=" * 50)
    print("TASK 3: Generating Dataset Statistics")
    print("=" * 50)
    
    # Load all splits
    train_df = pd.read_csv(FINAL_TRAIN_PATH)
    val_df = pd.read_csv(FINAL_VAL_PATH)
    test_df = pd.read_csv(FINAL_TEST_PATH)
    
    # Calculate statistics
    stats_data = []
    
    for split_name, df in [('train', train_df), ('validation', val_df), ('test', test_df)]:
        total = len(df)
        buggy = df['target'].sum()
        clean = total - buggy
        bug_ratio = buggy / total if total > 0 else 0
        
        # Count synthetic samples
        synthetic_count = (df['source'] == 'synthetic_llm').sum() if 'source' in df.columns else 0
        synthetic_pct = (synthetic_count / total * 100) if total > 0 else 0
        
        stats_data.append({
            'split': split_name,
            'total_samples': total,
            'buggy_samples': buggy,
            'clean_samples': clean,
            'bug_ratio': round(bug_ratio, 3),
            'synthetic_samples': synthetic_count,
            'synthetic_percentage': round(synthetic_pct, 2)
        })
    
    # Create statistics DataFrame
    stats_df = pd.DataFrame(stats_data)
    
    # Save statistics
    stats_df.to_csv(DATASET_STATS_PATH, index=False)
    
    print("\n📊 Dataset Statistics:")
    print(stats_df.to_string(index=False))
    
    # Summary
    total_samples = stats_df['total_samples'].sum()
    total_synthetic = stats_df['synthetic_samples'].sum()
    
    print(f"\n📈 Overall Summary:")
    print(f"  Total samples: {total_samples}")
    print(f"  Synthetic samples: {total_synthetic} ({total_synthetic/total_samples*100:.2f}%)")
    print(f"  Base samples: {total_samples - total_synthetic}")
    
    print(f"\n✓ Statistics saved to {DATASET_STATS_PATH}")
    print("\n✅ Stage 4 completed successfully!")
    print("=" * 50)
    
    return "Statistics generated"


# Define tasks
merge_task = PythonOperator(
    task_id='load_and_merge',
    python_callable=task_load_and_merge,
    dag=dag,
)

split_task = PythonOperator(
    task_id='create_splits',
    python_callable=task_create_splits,
    dag=dag,
)

stats_task = PythonOperator(
    task_id='generate_statistics',
    python_callable=task_generate_statistics,
    dag=dag,
)


# Set task dependencies
merge_task >> split_task >> stats_task