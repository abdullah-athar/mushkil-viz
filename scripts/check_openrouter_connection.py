import requests
import json
import os
import sys
from datetime import datetime

def test_openrouter_api(api_key, model_name):
    """
    Test OpenRouter API key with a simple completion request
    Returns tuple of (success_boolean, response_or_error_message)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://localhost",  
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name, 
        "messages": [
            {"role": "user", "content": "Say 'API test successful' if you can read this message."}
        ]
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=10
        )
        
        if response.status_code == 200:
            return True, "API key is valid and working"
        else:
            return False, f"Error: HTTP {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def main():
    # Get API key from environment variable or user input
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key:
        print("OPENROUTER_API_KEY is not set")
        return
    
    # Check if model name is passed as a command-line argument
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "google/gemini-2.0-pro-exp-02-05:free"  # Default model

    print("\nTesting OpenRouter API key with model:", model_name)
    success, message = test_openrouter_api(api_key, model_name)
    
    if success:
        print("✓ Success:", message)
    else:
        print("✗ Failed:", message)

if __name__ == "__main__":
    main()