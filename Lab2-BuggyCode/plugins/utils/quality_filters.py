"""
Quality Filters for Synthetic Code
Validates and filters generated code samples.
"""

import ast
import logging
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)


def validate_python_syntax(code: str) -> bool:
    """
    Validate that code is syntactically correct Python.
    
    Args:
        code: Python code string
        
    Returns:
        True if valid syntax, False otherwise
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False
    except Exception as e:
        logger.warning(f"Unexpected error in syntax validation: {str(e)}")
        return False


def check_minimum_length(code: str, min_length: int = 50) -> bool:
    """
    Check if code meets minimum length requirement.
    
    Args:
        code: Python code string
        min_length: Minimum number of characters
        
    Returns:
        True if meets minimum length
    """
    return len(code.strip()) >= min_length


def check_is_function(code: str) -> bool:
    """
    Check if code contains a function definition.
    
    Args:
        code: Python code string
        
    Returns:
        True if contains 'def ' keyword
    """
    return 'def ' in code


def check_has_docstring(code: str) -> bool:
    """
    Check if function has a docstring.
    
    Args:
        code: Python code string
        
    Returns:
        True if contains triple quotes (docstring indicator)
    """
    return '"""' in code or "'''" in code


def apply_quality_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all quality filters to a DataFrame of code samples.
    
    Args:
        df: DataFrame with 'func' column containing code
        
    Returns:
        DataFrame with additional boolean columns for each filter
    """
    logger.info(f"Applying quality filters to {len(df)} samples...")
    
    # Apply each filter
    df['syntax_valid'] = df['func'].apply(validate_python_syntax)
    df['min_length'] = df['func'].apply(check_minimum_length)
    df['is_function'] = df['func'].apply(check_is_function)
    df['has_docstring'] = df['func'].apply(check_has_docstring)
    
    # Overall pass: must pass syntax, length, and function checks
    df['passes_filter'] = (
        df['syntax_valid'] & 
        df['min_length'] & 
        df['is_function']
    )
    
    passed = df['passes_filter'].sum()
    total = len(df)
    pass_rate = (passed / total * 100) if total > 0 else 0
    
    logger.info(f"Quality filtering complete: {passed}/{total} samples passed ({pass_rate:.1f}%)")
    
    return df


def generate_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a quality report from filtered data.
    
    Args:
        df: DataFrame with filter results
        
    Returns:
        DataFrame containing quality metrics
    """
    total = len(df)
    
    report_data = {
        'metric': [
            'Total Generated',
            'Syntax Valid',
            'Meets Min Length',
            'Is Function',
            'Has Docstring',
            'Passed All Filters',
            'Pass Rate (%)'
        ],
        'count': [
            total,
            df['syntax_valid'].sum(),
            df['min_length'].sum(),
            df['is_function'].sum(),
            df['has_docstring'].sum(),
            df['passes_filter'].sum(),
            round(df['passes_filter'].sum() / total * 100, 2) if total > 0 else 0
        ]
    }
    
    report_df = pd.DataFrame(report_data)
    logger.info("Quality report generated")
    
    return report_df


def get_code_statistics(df: pd.DataFrame) -> Dict:
    """
    Get statistics about code samples.
    
    Args:
        df: DataFrame with code samples
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'total_samples': len(df),
        'avg_code_length': df['func'].str.len().mean(),
        'min_code_length': df['func'].str.len().min(),
        'max_code_length': df['func'].str.len().max(),
        'samples_with_docstring': df['has_docstring'].sum() if 'has_docstring' in df.columns else 0
    }
    
    return stats