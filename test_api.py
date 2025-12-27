"""
Test script for the Ibani Translation API
Run this after starting the FastAPI server to test the endpoints
"""

import requests
import json


API_URL = "http://localhost:8000"


def test_health():
    """Test the health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_single_translation():
    """Test single translation"""
    print("Testing /translate endpoint (English to Ibani)...")
    
    payload = {
        "text": "Hello, how are you?",
        "source_lang": "en",
        "target_lang": "ibani",
        "max_length": 256,
        "num_beams": 4
    }
    
    response = requests.post(f"{API_URL}/translate", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_batch_translation():
    """Test batch translation"""
    print("Testing /batch-translate endpoint...")
    
    payload = {
        "texts": [
            "Good morning",
            "Thank you",
            "How are you?"
        ],
        "source_lang": "en",
        "target_lang": "ibani",
        "max_length": 256,
        "num_beams": 4
    }
    
    response = requests.post(f"{API_URL}/batch-translate", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_reverse_translation():
    """Test Ibani to English translation"""
    print("Testing /translate endpoint (Ibani to English)...")
    
    # Using a sample from the training data
    payload = {
        "text": "Mịị anịị diri bie anị fịnị ḅara Jizọs tádọ́apụ",
        "source_lang": "ibani",
        "target_lang": "en",
        "max_length": 256,
        "num_beams": 4
    }
    
    response = requests.post(f"{API_URL}/translate", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


if __name__ == "__main__":
    print("="*60)
    print("Ibani Translation API Test Suite")
    print("="*60)
    print()
    
    try:
        test_health()
        test_single_translation()
        test_batch_translation()
        test_reverse_translation()
        
        print("="*60)
        print("All tests completed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API server.")
        print("Make sure the server is running at", API_URL)
    except Exception as e:
        print(f"ERROR: {str(e)}")
