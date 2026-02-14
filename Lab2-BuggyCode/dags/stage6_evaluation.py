"""
Stage 6: Model Evaluation DAG
Evaluates the fine-tuned model and generates comparison report.
MINI-EVAL MODE: Uses 20 samples for fast completion.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import sys
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add plugins to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Define paths
DATA_DIR = '/opt/airflow/data'
MODELS_DIR = '/opt/airflow/models'
RESULTS_DIR = '/opt/airflow/results'
TEST_PATH = os.path.join(DATA_DIR, 'final_test.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'codebert_bug_detector_final')
MODEL_RESULTS_PATH = os.path.join(RESULTS_DIR, 'codebert_results.csv')
BASELINE_RESULTS_PATH = os.path.join(RESULTS_DIR, 'baseline_results.csv')
FINAL_COMPARISON_PATH = os.path.join(RESULTS_DIR, 'final_comparison.csv')


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
    'stage6_evaluation',
    default_args=default_args,
    description='Evaluate trained model and generate comparison report (MINI MODE)',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['bug-detection', 'stage6', 'evaluation'],
)


def task_evaluate_trained_model(**context):
    """
    Task 1: Evaluate the fine-tuned model on test set.
    Saves progress after each batch for fault tolerance.
    """
    print("=" * 50)
    print("TASK 1: Evaluating Trained Model")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run Stage 5 first!")
    
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    
    # Load test data
    print(f"\nLoading test data from {TEST_PATH}...")
    test_df = pd.read_csv(TEST_PATH)
    print(f"Original test samples: {len(test_df)}")
    
    # MINI-EVAL MODE: Use only 20 samples for fast evaluation
    print("\n🔧 MINI-EVALUATION MODE ACTIVATED")
    print("   Using 20 samples for faster evaluation on CPU...")
    test_df = test_df.sample(n=20, random_state=42).reset_index(drop=True)
    print(f"Mini test set: {len(test_df)} samples")
    
    # Make predictions in batches with checkpointing
    print("\n🔧 BATCH CHECKPOINTING ENABLED")
    print("   Saving progress after every 5 samples...")
    
    BATCH_SIZE = 5
    predictions = []
    true_labels = test_df['target'].tolist()
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(RESULTS_DIR, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_start in range(0, len(test_df), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(test_df))
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(test_df) + BATCH_SIZE - 1) // BATCH_SIZE
            
            print(f"\n📦 Batch {batch_num}/{total_batches} (samples {batch_start}-{batch_end-1})")
            
            batch_predictions = []
            
            for idx in range(batch_start, batch_end):
                row = test_df.iloc[idx]
                print(f"  Processing sample {idx+1}/{len(test_df)}...")
                
                # Tokenize
                inputs = tokenizer(
                    row['func'],
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Predict
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=-1).item()
                batch_predictions.append(pred)
                predictions.append(pred)
            
            # Save checkpoint after each batch
            checkpoint_path = os.path.join(checkpoint_dir, f'predictions_batch_{batch_num}.csv')
            checkpoint_df = pd.DataFrame({
                'sample_idx': list(range(batch_start, batch_end)),
                'prediction': batch_predictions,
                'true_label': true_labels[batch_start:batch_end]
            })
            checkpoint_df.to_csv(checkpoint_path, index=False)
            
            print(f"  ✓ Batch {batch_num} checkpoint saved: {checkpoint_path}")
            print(f"  Progress: {len(predictions)}/{len(test_df)} samples complete ({len(predictions)/len(test_df)*100:.0f}%)")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    print("\n📊 Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Save final results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results_df = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1_score'],
        'score': [accuracy, precision, recall, f1]
    })
    
    results_df.to_csv(MODEL_RESULTS_PATH, index=False)
    
    print(f"\n✓ Final results saved to {MODEL_RESULTS_PATH}")
    print("=" * 50)
    
    return f"Model evaluated: F1={f1:.4f}"


def task_evaluate_baseline(**context):
    """
    Task 2: Create a simple baseline for comparison.
    Uses a majority class classifier.
    """
    print("=" * 50)
    print("TASK 2: Evaluating Baseline (Majority Class)")
    print("=" * 50)
    
    # Load test data
    test_df = pd.read_csv(TEST_PATH)
    
    # MINI-EVAL MODE: Use same 20 samples as model evaluation
    print("\n🔧 MINI-EVALUATION MODE")
    test_df = test_df.sample(n=20, random_state=42).reset_index(drop=True)
    print(f"Evaluating on {len(test_df)} samples")
    
    true_labels = test_df['target'].tolist()
    
    # Baseline: Always predict majority class
    majority_class = test_df['target'].mode()[0]
    predictions = [majority_class] * len(test_df)
    
    print(f"\nBaseline strategy: Always predict class {majority_class}")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    print("\n📊 Baseline Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'metric': ['accuracy', 'precision', 'recall', 'f1_score'],
        'score': [accuracy, precision, recall, f1]
    })
    
    results_df.to_csv(BASELINE_RESULTS_PATH, index=False)
    
    print(f"\n✓ Results saved to {BASELINE_RESULTS_PATH}")
    print("=" * 50)
    
    return f"Baseline evaluated: F1={f1:.4f}"


def task_generate_comparison(**context):
    """
    Task 3: Generate final comparison report.
    """
    print("=" * 50)
    print("TASK 3: Generating Final Comparison")
    print("=" * 50)
    
    # Load results
    trained_results = pd.read_csv(MODEL_RESULTS_PATH)
    baseline_results = pd.read_csv(BASELINE_RESULTS_PATH)
    
    # Combine into comparison
    comparison_df = pd.DataFrame({
        'metric': trained_results['metric'],
        'trained_model': trained_results['score'],
        'baseline': baseline_results['score']
    })
    
    # Calculate improvement
    comparison_df['improvement'] = comparison_df['trained_model'] - comparison_df['baseline']
    comparison_df['improvement_pct'] = (comparison_df['improvement'] / comparison_df['baseline'] * 100).round(2)
    
    # Save comparison
    comparison_df.to_csv(FINAL_COMPARISON_PATH, index=False)
    
    print("\n" + "=" * 70)
    print("🎉 FINAL RESULTS - MODEL COMPARISON")
    print("=" * 70)
    print("\n📊 Performance Metrics:")
    print(comparison_df.to_string(index=False))
    
    # Summary
    f1_improvement = comparison_df[comparison_df['metric'] == 'f1_score']['improvement'].values[0]
    accuracy_improvement = comparison_df[comparison_df['metric'] == 'accuracy']['improvement'].values[0]
    
    print(f"\n🎯 Key Takeaways:")
    print(f"  F1 Score Improvement:  {f1_improvement:+.4f}")
    print(f"  Accuracy Improvement:  {accuracy_improvement:+.4f}")
    
    if f1_improvement > 0.05:
        print("\n✅ Model shows significant improvement over baseline!")
    elif f1_improvement > 0:
        print("\n✓ Model shows modest improvement over baseline")
    else:
        print("\n⚠️ Model performance similar to baseline")
    
    print(f"\n✓ Final comparison saved to {FINAL_COMPARISON_PATH}")
    print("\n" + "=" * 70)
    print("🎊 STAGE 6 COMPLETE - FULL PIPELINE FINISHED!")
    print("=" * 70)
    
    return "Comparison complete"


# Define tasks
eval_model_task = PythonOperator(
    task_id='evaluate_trained_model',
    python_callable=task_evaluate_trained_model,
    dag=dag,
)

eval_baseline_task = PythonOperator(
    task_id='evaluate_baseline',
    python_callable=task_evaluate_baseline,
    dag=dag,
)

comparison_task = PythonOperator(
    task_id='generate_comparison',
    python_callable=task_generate_comparison,
    dag=dag,
)


# Set task dependencies - eval_model and eval_baseline run in parallel!
[eval_model_task, eval_baseline_task] >> comparison_task