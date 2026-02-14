"""
Stage 3: Quality Filtering DAG
Validates and filters synthetic code samples for quality.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys
import pandas as pd
import glob

# Add plugins to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from plugins.utils.quality_filters import (
    apply_quality_filters,
    generate_quality_report,
    get_code_statistics
)


# Define paths
DATA_DIR = '/opt/airflow/data'
SYNTHETIC_DIR = os.path.join(DATA_DIR, 'synthetic_batches')
FILTERED_PATH = os.path.join(DATA_DIR, 'synthetic_filtered.csv')
QUALITY_REPORT_PATH = os.path.join(DATA_DIR, 'quality_report.csv')


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
    'stage3_quality_filtering',
    default_args=default_args,
    description='Filter and validate synthetic code samples',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['bug-detection', 'stage3', 'quality'],
)


def task_load_synthetic_data(**context):
    """
    Task 1: Load all synthetic batches and combine them.
    """
    print("=" * 50)
    print("TASK 1: Loading Synthetic Batches")
    print("=" * 50)
    
    # Find all batch files
    batch_files = glob.glob(os.path.join(SYNTHETIC_DIR, 'synthetic_batch_*.csv'))
    
    if not batch_files:
        raise FileNotFoundError(f"No synthetic batch files found in {SYNTHETIC_DIR}")
    
    print(f"Found {len(batch_files)} batch files")
    
    # Load and combine all batches
    dfs = []
    for batch_file in sorted(batch_files):
        df = pd.read_csv(batch_file)
        print(f"  Loaded {os.path.basename(batch_file)}: {len(df)} samples")
        dfs.append(df)
    
    # Combine all batches
    combined_df = pd.concat(dfs, ignore_index=True)
    
    print(f"\n✓ Total synthetic samples loaded: {len(combined_df)}")
    print(f"  Columns: {list(combined_df.columns)}")
    print("=" * 50)
    
    # Save combined data temporarily
    temp_path = os.path.join(DATA_DIR, 'synthetic_combined.csv')
    combined_df.to_csv(temp_path, index=False)
    
    return f"Loaded {len(combined_df)} samples from {len(batch_files)} batches"


def task_apply_filters(**context):
    """
    Task 2: Apply quality filters to synthetic data.
    """
    print("=" * 50)
    print("TASK 2: Applying Quality Filters")
    print("=" * 50)
    
    # Load combined data
    temp_path = os.path.join(DATA_DIR, 'synthetic_combined.csv')
    df = pd.read_csv(temp_path)
    
    print(f"Filtering {len(df)} samples...")
    print("\nApplying filters:")
    print("  1. Python syntax validation")
    print("  2. Minimum code length (>50 chars)")
    print("  3. Contains function definition")
    
    # Apply filters
    df_filtered = apply_quality_filters(df)
    
    # Show results for each filter
    print("\nFilter Results:")
    print(f"  Syntax Valid:    {df_filtered['syntax_valid'].sum()}/{len(df_filtered)}")
    print(f"  Min Length:      {df_filtered['min_length'].sum()}/{len(df_filtered)}")
    print(f"  Is Function:     {df_filtered['is_function'].sum()}/{len(df_filtered)}")
    print(f"  Has Docstring:   {df_filtered['has_docstring'].sum()}/{len(df_filtered)}")
    print(f"  Passed All:      {df_filtered['passes_filter'].sum()}/{len(df_filtered)}")
    
    # Keep only samples that passed filters
    df_passed = df_filtered[df_filtered['passes_filter']].copy()
    
    # Drop the filter columns before saving (keep original data clean)
    columns_to_drop = ['syntax_valid', 'min_length', 'is_function', 'has_docstring', 'passes_filter']
    df_passed = df_passed.drop(columns=columns_to_drop)
    
    # Save filtered data
    df_passed.to_csv(FILTERED_PATH, index=False)
    
    print(f"\n✓ Filtered data saved to {FILTERED_PATH}")
    print(f"  Samples retained: {len(df_passed)}/{len(df)}")
    print("=" * 50)
    
    return f"Filtered: {len(df_passed)} of {len(df)} samples passed"


def task_generate_report(**context):
    """
    Task 3: Generate quality report.
    """
    print("=" * 50)
    print("TASK 3: Generating Quality Report")
    print("=" * 50)
    
    # Load combined data with filters
    temp_path = os.path.join(DATA_DIR, 'synthetic_combined.csv')
    df = pd.read_csv(temp_path)
    
    # Reapply filters to get metrics
    df_with_filters = apply_quality_filters(df)
    
    # Generate report
    report_df = generate_quality_report(df_with_filters)
    
    # Get code statistics
    stats = get_code_statistics(df_with_filters)
    
    # Save report
    report_df.to_csv(QUALITY_REPORT_PATH, index=False)
    
    print("\n📊 Quality Report:")
    print(report_df.to_string(index=False))
    
    print("\n📈 Code Statistics:")
    print(f"  Total samples:        {stats['total_samples']}")
    print(f"  Avg code length:      {stats['avg_code_length']:.1f} chars")
    print(f"  Min code length:      {stats['min_code_length']} chars")
    print(f"  Max code length:      {stats['max_code_length']} chars")
    print(f"  With docstrings:      {stats['samples_with_docstring']}")
    
    print(f"\n✓ Quality report saved to {QUALITY_REPORT_PATH}")
    print("\n✅ Stage 3 completed successfully!")
    print("=" * 50)
    
    return "Quality report generated"


# Define tasks
load_task = PythonOperator(
    task_id='load_synthetic_data',
    python_callable=task_load_synthetic_data,
    dag=dag,
)

filter_task = PythonOperator(
    task_id='apply_quality_filters',
    python_callable=task_apply_filters,
    dag=dag,
)

report_task = PythonOperator(
    task_id='generate_quality_report',
    python_callable=task_generate_report,
    dag=dag,
)


# Set task dependencies
load_task >> filter_task >> report_task