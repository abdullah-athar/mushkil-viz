from typing import Optional
import os
from openai import OpenAI
from dotenv import load_dotenv

def init_openai_client(api_key: Optional[str] = None, api_base: Optional[str] = None) -> OpenAI:
    """
    Initialize the OpenAI client with the appropriate configuration.
    Uses environment variables if no parameters are provided.
    
    Args:
        api_key: Optional OpenAI API key. If not provided, uses OPENAI_API_KEY from environment.
        api_base: Optional API base URL. If not provided, uses OPENAI_API_BASE from environment.
        
    Returns:
        OpenAI client instance configured for use with OpenRouter
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get configuration from parameters or environment
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    api_base = api_base or os.getenv("OPENAI_API_BASE")
    
    if not api_key:
        raise ValueError("OpenAI API key must be provided either through parameters or OPENAI_API_KEY environment variable")
    
    
    client = OpenAI(
        api_key=api_key,
        base_url=api_base,  
        timeout=60.0,  # Increased timeout for potentially longer operations
    )
    
    return client 