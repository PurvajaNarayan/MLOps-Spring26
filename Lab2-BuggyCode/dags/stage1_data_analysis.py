"""
Stage 1: Dataset Analysis DAG
Loads CodeXGLUE dataset, saves splits, and generates analysis report.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys

# Add plugins to path so we can import our utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugins.utils.data_loader import (
    load_codexglue_dataset,
    save_dataset_splits,
    generate_dataset_analysis,
    validate_dataset
)


# Define paths
DATA_DIR = '/opt/airflow/data'
BASE_DATA_DIR = os.path.join(DATA_DIR, 'base')
ANALYSIS_REPORT_PATH = os.path.join(DATA_DIR, 'analysis_report.csv')


# Default arguments for the DAG
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
    'stage1_dataset_analysis',
    default_args=default_args,
    description='Load and analyze CodeXGLUE defect detection dataset',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['bug-detection', 'stage1', 'data-loading'],
)


def task_load_dataset(**context):
    """
    Task 1: Load CodeXGLUE dataset from HuggingFace.
    Stores the dataset in XCom for the next task.
    """
    print("=" * 50)
    print("TASK 1: Loading CodeXGLUE Dataset")
    print("=" * 50)
    
    # Load the dataset
    datasets = load_codexglue_dataset()
    
    # Validate each split
    for split_name, df in datasets.items():
        validate_dataset(df)
        print(f"✓ Validated {split_name} split: {len(df)} samples")
    
    # Save to disk immediately (don't rely on XCom for large data)
    save_dataset_splits(datasets, BASE_DATA_DIR)
    
    print(f"\n✓ Dataset loaded and saved to {BASE_DATA_DIR}")
    print("=" * 50)
    
    return "Dataset loaded successfully"


def task_generate_analysis(**context):
    """
    Task 2: Generate analysis report from the saved datasets.
    """
    print("=" * 50)
    print("TASK 2: Generating Analysis Report")
    print("=" * 50)
    
    # Load datasets from disk
    import pandas as pd
    datasets = {}
    for split_name in ['train', 'validation', 'test']:
        file_path = os.path.join(BASE_DATA_DIR, f"{split_name}.csv")
        datasets[split_name] = pd.read_csv(file_path)
        print(f"Loaded {split_name}: {len(datasets[split_name])} samples")
    
    # Generate analysis
    analysis_df = generate_dataset_analysis(datasets)
    
    # Save analysis report
    analysis_df.to_csv(ANALYSIS_REPORT_PATH, index=False)
    
    print("\nAnalysis Report:")
    print(analysis_df.to_string(index=False))
    print(f"\n✓ Analysis report saved to {ANALYSIS_REPORT_PATH}")
    print("=" * 50)
    
    return "Analysis complete"


def task_print_summary(**context):
    """
    Task 3: Print final summary.
    """
    print("=" * 50)
    print("STAGE 1 COMPLETE - SUMMARY")
    print("=" * 50)
    
    import pandas as pd
    analysis_df = pd.read_csv(ANALYSIS_REPORT_PATH)
    
    print("\n📊 Dataset Statistics:")
    print(analysis_df.to_string(index=False))
    
    total_samples = analysis_df['total_samples'].sum()
    total_buggy = analysis_df['buggy_samples'].sum()
    
    print(f"\n📈 Total samples across all splits: {total_samples}")
    print(f"🐛 Total buggy samples: {total_buggy}")
    print(f"✨ Total clean samples: {total_samples - total_buggy}")
    
    print("\n✅ Stage 1 completed successfully!")
    print("=" * 50)
    
    return "Summary printed"


# Define tasks
load_task = PythonOperator(
    task_id='load_dataset',
    python_callable=task_load_dataset,
    dag=dag,
)

analysis_task = PythonOperator(
    task_id='generate_analysis',
    python_callable=task_generate_analysis,
    dag=dag,
)

summary_task = PythonOperator(
    task_id='print_summary',
    python_callable=task_print_summary,
    dag=dag,
)


# Set task dependencies
# This means: load_task must complete before analysis_task can run
# and analysis_task must complete before summary_task can run
load_task >> analysis_task >> summary_task