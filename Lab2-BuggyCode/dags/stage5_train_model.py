# """
# Stage 5: Model Training DAG
# Prepares data and fine-tunes Phi-3 model for bug detection.

# """

# from datetime import datetime, timedelta
# from airflow import DAG
# from airflow.operators.python import PythonOperator
# import os
# import sys

# # Add plugins to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# from plugins.utils.model_trainer import (
#     prepare_tokenized_datasets,
#     train_model
# )


# # Define paths
# DATA_DIR = '/opt/airflow/data'
# MODELS_DIR = '/opt/airflow/models'
# TRAIN_PATH = os.path.join(DATA_DIR, 'final_train.csv')
# VAL_PATH = os.path.join(DATA_DIR, 'final_val.csv')
# MODEL_OUTPUT_DIR = os.path.join(MODELS_DIR, 'codebert_bug_detector_final')

# # Model configuration
# MODEL_NAME = "microsoft/codebert-base"
# MAX_LENGTH = 512
# NUM_EPOCHS = 1
# BATCH_SIZE = 8
# LEARNING_RATE = 2e-5


# # Default arguments
# default_args = {
#     'owner': 'purvaja',
#     'depends_on_past': False,
#     'email_on_failure': False,
#     'email_on_retry': False,
#     'retries': 0,  
#     'execution_timeout': timedelta(hours=2),  
# }


# # Define the DAG
# dag = DAG(
#     'stage5_train_model',
#     default_args=default_args,
#     description='Fine-tune Phi-3 model for bug detection (LONG RUNNING!)',
#     schedule_interval=None,
#     start_date=datetime(2024, 1, 1),
#     catchup=False,
#     tags=['bug-detection', 'stage5', 'training', 'long-running'],
# )


# def task_prepare_data(**context):
#     """
#     Task 1: Load and tokenize training and validation data.
#     """
#     print("=" * 50)
#     print("TASK 1: Preparing and Tokenizing Data")
#     print("=" * 50)
#     print(f"\n⚠️  This will download the Phi-3-mini model (~7GB)")
#     print("First run will take extra time for model download.\n")
    
#     print(f"Loading data from:")
#     print(f"  Train: {TRAIN_PATH}")
#     print(f"  Val: {VAL_PATH}")
    
#     # Prepare datasets
#     train_dataset, val_dataset, tokenizer = prepare_tokenized_datasets(
#         train_path=TRAIN_PATH,
#         val_path=VAL_PATH,
#         model_name=MODEL_NAME,
#         max_length=MAX_LENGTH
#     )
    
#     print(f"\n✓ Data prepared:")
#     print(f"  Train samples: {len(train_dataset)}")
#     print(f"  Val samples: {len(val_dataset)}")
#     print(f"  Max length: {MAX_LENGTH}")
#     print(f"  Tokenizer: {MODEL_NAME}")
    
#     # Save datasets temporarily
#     train_dataset.save_to_disk(os.path.join(DATA_DIR, 'train_tokenized'))
#     val_dataset.save_to_disk(os.path.join(DATA_DIR, 'val_tokenized'))
#     tokenizer.save_pretrained(os.path.join(DATA_DIR, 'tokenizer'))
    
#     print(f"\n✓ Tokenized data saved")
#     print("=" * 50)
    
#     return "Data preparation complete"


# def task_train_model(**context):
#     """
#     Task 2: Train the model.
    
#     WARNING: This will take 1-3 hours depending on hardware!
#     """
#     print("=" * 50)
#     print("TASK 2: Training Phi-3 Model")
#     print("=" * 50)
#     print("\n" + "!" * 50)
#     print("⚠️  WARNING: LONG-RUNNING TASK")
#     print("   Expected time: 1-3 hours")
#     print("   This will use significant CPU/GPU resources")
#     print("!" * 50 + "\n")
    
#     from datasets import load_from_disk
#     from transformers import AutoTokenizer
    
#     # Load prepared data
#     print("Loading tokenized datasets...")
#     train_dataset = load_from_disk(os.path.join(DATA_DIR, 'train_tokenized'))
#     val_dataset = load_from_disk(os.path.join(DATA_DIR, 'val_tokenized'))
#     tokenizer = AutoTokenizer.from_pretrained(os.path.join(DATA_DIR, 'tokenizer'))

#     print(f"Original datasets: {len(train_dataset)} train, {len(val_dataset)} val")

#     # MINI-TRAINING MODE: Use only 1000 samples for fast training
#     print("\n🔧 MINI-TRAINING MODE ACTIVATED")
#     print("   Using subset of data for faster training...")
#     train_dataset = train_dataset.shuffle(seed=42).select(range(min(1000, len(train_dataset))))
#     val_dataset = val_dataset.shuffle(seed=42).select(range(min(200, len(val_dataset))))

#     print(f"Mini datasets: {len(train_dataset)} train, {len(val_dataset)} val")    
    
    
#     # Create output directory
#     os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
#     print(f"\nTraining configuration:")
#     print(f"  Model: {MODEL_NAME}")
#     print(f"  Epochs: {NUM_EPOCHS}")
#     print(f"  Batch size: {BATCH_SIZE}")
#     print(f"  Learning rate: {LEARNING_RATE}")
#     print(f"  Output: {MODEL_OUTPUT_DIR}")
    
#     print(f"\n🚀 Starting training...")
#     print(f"   Training on {len(train_dataset)} samples")
#     print(f"   Validating on {len(val_dataset)} samples")
#     print(f"\n   This will take a while. Grab some coffee! ☕\n")
    
#     # Train model
#     train_result = train_model(
#         train_dataset=train_dataset,
#         val_dataset=val_dataset,
#         tokenizer=tokenizer,
#         output_dir=MODEL_OUTPUT_DIR,
#         model_name=MODEL_NAME,
#         num_epochs=NUM_EPOCHS,
#         batch_size=BATCH_SIZE,
#         learning_rate=LEARNING_RATE
#     )
    
#     print("\n" + "=" * 50)
#     print("🎉 TRAINING COMPLETE!")
#     print("=" * 50)
#     print(f"\nFinal training loss: {train_result.training_loss:.4f}")
#     print(f"Model saved to: {MODEL_OUTPUT_DIR}")
    
#     print("\n✅ Stage 5 completed successfully!")
#     print("=" * 50)
    
#     return f"Training complete. Loss: {train_result.training_loss:.4f}"


# # Define tasks
# prepare_task = PythonOperator(
#     task_id='prepare_and_tokenize_data',
#     python_callable=task_prepare_data,
#     dag=dag,
# )

# train_task = PythonOperator(
#     task_id='train_model',
#     python_callable=task_train_model,
#     dag=dag,
# )


# # Set task dependencies
# prepare_task >> train_task

"""
Stage 5 Alternative: Quick Model Setup
Downloads pre-trained CodeBERT instead of training from scratch.
This is for demonstration/learning purposes.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define paths
MODELS_DIR = '/opt/airflow/models'
MODEL_OUTPUT_DIR = os.path.join(MODELS_DIR, 'codebert_bug_detector_final')
MODEL_NAME = "microsoft/codebert-base"

default_args = {
    'owner': 'purvaja',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'stage5_quick_setup',
    default_args=default_args,
    description='Quick setup: Download pre-trained CodeBERT model',
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['bug-detection', 'stage5', 'quick-setup'],
)


def task_download_pretrained_model(**context):
    """
    Download and save pre-trained CodeBERT model.
    This skips training for learning/demo purposes.
    """
    print("=" * 50)
    print("QUICK SETUP: Downloading Pre-trained Model")
    print("=" * 50)
    print("\n📝 NOTE: This uses pre-trained CodeBERT without fine-tuning")
    print("   For learning Airflow orchestration, this is perfectly fine!")
    print("   You can always train later on GPU if needed.\n")
    
    print(f"Downloading {MODEL_NAME}...")
    
    # Create output directory
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    # Download model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2  # Binary classification
    )
    
    print(f"\n✓ Model downloaded successfully")
    print(f"  Parameters: ~125M")
    print(f"  Size: ~500MB")
    
    # Save to models directory
    print(f"\nSaving to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    
    print(f"\n✓ Model saved to {MODEL_OUTPUT_DIR}")
    print("\n✅ Quick setup complete!")
    print("   You can now run Stage 6 (Evaluation)!")
    print("=" * 50)
    
    return "Pre-trained model ready"


# Define task
download_task = PythonOperator(
    task_id='download_pretrained_model',
    python_callable=task_download_pretrained_model,
    dag=dag,
)