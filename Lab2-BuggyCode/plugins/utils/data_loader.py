"""
Data Loader Utilities
Handles loading and saving datasets for the bug detection pipeline.
"""

import os
import logging
from typing import Tuple, Dict
import pandas as pd
from datasets import load_dataset

# Set up logging
logger = logging.getLogger(__name__)


def load_codexglue_dataset() -> Dict[str, pd.DataFrame]:
    """
    Load the CodeXGLUE defect detection dataset from HuggingFace.
    
    Returns:
        Dict containing 'train', 'validation', and 'test' DataFrames
    """
    logger.info("Loading CodeXGLUE defect detection dataset...")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("code_x_glue_cc_defect_detection")
        
        # Convert to pandas DataFrames
        train_df = pd.DataFrame(dataset['train'])
        val_df = pd.DataFrame(dataset['validation'])
        test_df = pd.DataFrame(dataset['test'])
        
        logger.info(f"Loaded dataset - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def save_dataset_splits(datasets: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Save dataset splits to CSV files.
    
    Args:
        datasets: Dictionary with 'train', 'validation', 'test' DataFrames
        output_dir: Directory to save CSV files
    """
    logger.info(f"Saving dataset splits to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split
    for split_name, df in datasets.items():
        output_path = os.path.join(output_dir, f"{split_name}.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {split_name} split to {output_path} ({len(df)} samples)")


def generate_dataset_analysis(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate analysis report for the dataset.
    
    Args:
        datasets: Dictionary with dataset splits
        
    Returns:
        DataFrame containing analysis metrics
    """
    logger.info("Generating dataset analysis...")
    
    analysis_data = []
    
    for split_name, df in datasets.items():
        # Calculate statistics
        total_samples = len(df)
        buggy_samples = df['target'].sum() if 'target' in df.columns else 0
        clean_samples = total_samples - buggy_samples
        
        # Calculate average code length
        avg_code_length = df['func'].str.len().mean() if 'func' in df.columns else 0
        
        # Calculate balance ratio
        balance_ratio = buggy_samples / total_samples if total_samples > 0 else 0
        
        analysis_data.append({
            'split': split_name,
            'total_samples': total_samples,
            'buggy_samples': buggy_samples,
            'clean_samples': clean_samples,
            'balance_ratio': round(balance_ratio, 3),
            'avg_code_length': round(avg_code_length, 2)
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    logger.info("Dataset analysis complete")
    
    return analysis_df


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate that a dataset has the required columns and structure.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises exception otherwise
    """
    required_columns = ['func', 'target']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if len(df) == 0:
        raise ValueError("Dataset is empty")
    
    logger.info(f"Dataset validation passed ({len(df)} samples)")
    return True