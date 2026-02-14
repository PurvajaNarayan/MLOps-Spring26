"""
LLM Wrapper for Synthetic Data Generation
Uses OpenRouter with free models instead of OpenAI.
"""

import os
import time
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class ChatOpenRouter(ChatOpenAI):
    """
    Custom wrapper for OpenRouter API.
    Extends ChatOpenAI to work with OpenRouter's API.
    """
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b:free", **kwargs):
        # Get API key from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        # Pass to parent class with the correct parameter name
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=api_key,  # Changed from api_key to openai_api_key
            model_name=model_name,
            **kwargs
        )

def call_llm_with_retry(
    prompt: str,
    system_prompt: Optional[str] = None,
    model_id: str = "openai/gpt-oss-20b:free",
    temperature: float = 0.8,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> str:
    """
    Call LLM with retry logic for robustness.
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        model_id: Model to use (default: free Gemini)
        temperature: Sampling temperature
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        LLM response text
    """
    logger.info(f"Calling LLM with model: {model_id}")
    
    for attempt in range(max_retries):
        try:
            # Create LLM instance
            llm = ChatOpenRouter(
                model_name=model_id,
                temperature=temperature,
                max_retries=2
            )
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            # Call LLM
            response = llm.invoke(messages)
            
            # Add rate limiting delay
            time.sleep(1)  # 1 second between requests
            
            return response.content
            
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"All {max_retries} attempts failed")
                raise
    
    raise RuntimeError("Failed to get LLM response after all retries")


def parse_buggy_code_response(response: str) -> dict:
    """
    Parse LLM response to extract code and bug type.
    
    Expected format:
    CODE:
    <code here>
    
    BUG_TYPE: <bug type>
    
    Args:
        response: Raw LLM response
        
    Returns:
        Dict with 'code' and 'bug_type'
    """
    try:
        # Split by CODE: and BUG_TYPE:
        code = ""
        bug_type = "unknown"
        
        if "CODE:" in response:
            parts = response.split("CODE:")
            if len(parts) > 1:
                code_section = parts[1]
                
                # Extract code (before BUG_TYPE if it exists)
                if "BUG_TYPE:" in code_section:
                    code = code_section.split("BUG_TYPE:")[0].strip()
                    bug_type = code_section.split("BUG_TYPE:")[1].strip()
                else:
                    code = code_section.strip()
        else:
            # If no CODE: marker, treat entire response as code
            code = response.strip()
        
        # Clean up code (remove markdown backticks if present)
        code = code.replace("```python", "").replace("```", "").strip()
        
        return {
            'code': code,
            'bug_type': bug_type
        }
        
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        return {
            'code': response.strip(),
            'bug_type': 'parse_error'
        }