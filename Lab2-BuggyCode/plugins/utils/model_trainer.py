"""
Model Training Utilities
Handles data preparation and model training for bug detection.
"""

import os
import logging
from typing import Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import pandas as pd

logger = logging.getLogger(__name__)


def prepare_tokenized_datasets(
    train_path: str,
    val_path: str,
    model_name: str = "microsoft/codebert-base",
    max_length: int = 512
) -> Tuple[Dataset, Dataset, AutoTokenizer]:
    """
    Load data and create tokenized HuggingFace datasets.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        model_name: Model identifier for tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_dataset, val_dataset, tokenizer)
    """
    logger.info(f"Loading data from {train_path} and {val_path}")
    
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df[['func', 'target']])
    val_dataset = Dataset.from_pandas(val_df[['func', 'target']])
    
    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples['func'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    
    logger.info("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Rename target to labels for HuggingFace
    train_dataset = train_dataset.rename_column('target', 'labels')
    val_dataset = val_dataset.rename_column('target', 'labels')
    
    # Set format
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    logger.info("Data preparation complete")
    
    return train_dataset, val_dataset, tokenizer


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer: AutoTokenizer,
    output_dir: str,
    model_name: str = "microsoft/codebert-base",
    num_epochs: int = 3,
    batch_size: int = 4,  
    learning_rate: float = 2e-5
):
    """
    Train the bug detection model.
    
    Args:
        train_dataset: Tokenized training dataset
        val_dataset: Tokenized validation dataset
        tokenizer: Tokenizer instance
        output_dir: Directory to save model
        model_name: Model identifier
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        trust_remote_code=True
    )
    
    # Set pad token id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cpu":
        logger.warning("Training on CPU - this will be VERY slow! Consider using a GPU.")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  
        report_to="none",  
    )
    
    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Device: {device}")
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    
    # Train
    train_result = trainer.train()
    
    logger.info("Training complete!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")
    
    # Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Model saved successfully")
    
    return train_result