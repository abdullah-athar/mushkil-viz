#!/usr/bin/env python3
"""
Simple test script to verify the google-genai API works correctly.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()


def test_genai_import():
    """Test if google-genai can be imported."""
    try:
        from google import genai
        print("✅ google.genai imported successfully")
        return genai
    except ImportError as e:
        print(f"❌ Failed to import google.genai: {e}")
        return None

def test_client_creation(genai):
    """Test client creation."""
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        print("✅ Client created successfully")
        return client
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return None

def test_model_generation(client):
    """Test model generation."""
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Say 'Hello, World!' in one sentence."
        )
        print("✅ Text generation successful")
        print(f"Response: {response.text}")
        return True
    except Exception as e:
        print(f"❌ Failed to generate text: {e}")
        return False

def test_with_config():
    """Test with generation config."""
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client()
        
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=100
        )
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="What is 2+2?",
            config=config
        )
        print("✅ Generation with config successful")
        print(f"Response: {response.text}")
        return True
    except Exception as e:
        print(f"❌ Failed to generate with config: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing google-genai API")
    print("=" * 40)
    
    # Test 1: Import
    genai = test_genai_import()
    if not genai:
        return False
    
    # Test 2: Client creation
    client = test_client_creation(genai)
    if not client:
        return False
    
    # Test 3: Simple generation
    if not test_model_generation(client):
        return False
    
    # Test 4: Generation with config
    if not test_with_config():
        return False
    
    print("\n" + "=" * 40)
    print("✅ All google-genai API tests passed!")
    print("🚀 You can now run the full framework test:")
    print("   python test_framework.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 