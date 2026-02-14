"""
Code Generator for Synthetic Buggy Code
Generates buggy Python functions using LLMs.
"""

import logging
import random
from typing import List, Dict
from plugins.utils.llm_wrapper import call_llm_with_retry, parse_buggy_code_response

logger = logging.getLogger(__name__)

# Bug types to generate
BUG_TYPES = [
    "off-by-one error",
    "null pointer / None reference",
    "type error",
    "logic error",
    "infinite loop",
    "index out of bounds"
]


def generate_buggy_code_sample(sample_id: int) -> Dict[str, str]:
    """
    Generate a single buggy Python code sample using LLM.
    
    Args:
        sample_id: Unique identifier for this sample
        
    Returns:
        Dict with 'func', 'target', 'bug_type', 'source', 'sample_id'
    """
    logger.info(f"Generating sample {sample_id}...")
    
    # Randomly select a bug type
    bug_type = random.choice(BUG_TYPES)
    
    # Create prompt for LLM
    system_prompt = """You are an expert Python programmer who generates code samples for educational purposes.
Your task is to create Python functions that contain subtle bugs for training bug detection models."""
    
    user_prompt = f"""Generate a Python function with exactly ONE subtle bug.

Bug type to include: {bug_type}

Requirements:
1. The function should be 10-30 lines long
2. Include ONE subtle bug of the specified type
3. The bug should be realistic (something a programmer might actually write)
4. Use clear variable names
5. Add a docstring

Format your response EXACTLY as:
CODE:
<your python function here>

BUG_TYPE: {bug_type}

Do not include any other text or explanations."""

    try:
        # Call LLM
        response = call_llm_with_retry(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.8
        )
        
        # Parse response
        parsed = parse_buggy_code_response(response)
        
        # Validate we got code
        if not parsed['code'] or len(parsed['code']) < 50:
            logger.warning(f"Sample {sample_id}: Generated code too short, retrying...")
            # Retry once
            response = call_llm_with_retry(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.9  # Slightly higher temperature
            )
            parsed = parse_buggy_code_response(response)
        
        return {
            'func': parsed['code'],
            'target': 1,  # 1 = buggy
            'bug_type': parsed['bug_type'],
            'source': 'synthetic_llm',
            'sample_id': sample_id
        }
        
    except Exception as e:
        logger.error(f"Error generating sample {sample_id}: {str(e)}")
        # Return a minimal fallback
        return {
            'func': f"# Failed to generate sample {sample_id}\ndef placeholder():\n    pass",
            'target': 1,
            'bug_type': 'generation_failed',
            'source': 'synthetic_llm_failed',
            'sample_id': sample_id
        }


def generate_batch(batch_id: int, samples_per_batch: int = 10) -> List[Dict]:
    """
    Generate a batch of buggy code samples.
    
    Args:
        batch_id: Identifier for this batch
        samples_per_batch: Number of samples to generate
        
    Returns:
        List of generated samples
    """
    logger.info(f"Starting batch {batch_id} - generating {samples_per_batch} samples")
    
    samples = []
    for i in range(samples_per_batch):
        sample_id = batch_id * 100 + i  # Unique ID: batch_id*100 + sample_num
        sample = generate_buggy_code_sample(sample_id)
        sample['batch_id'] = batch_id
        samples.append(sample)
        
        logger.info(f"Batch {batch_id}: Generated sample {i+1}/{samples_per_batch}")
    
    logger.info(f"Batch {batch_id} complete - {len(samples)} samples generated")
    return samples