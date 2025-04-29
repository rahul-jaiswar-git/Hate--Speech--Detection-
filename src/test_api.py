import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

# Get API key
PERSPECTIVE_API_KEY = os.getenv('PERSPECTIVE_API_KEY')
if not PERSPECTIVE_API_KEY:
    print("⚠️ Perspective API key not found in .env file")
    exit(1)

# Test text
TEST_TEXT = "This is a test message to check available attributes."

# API URL
PERSPECTIVE_API_URL = 'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key=' + PERSPECTIVE_API_KEY

# Test all possible attributes
TEST_ATTRIBUTES = [
    "TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT",
    "SEXUALLY_EXPLICIT", "OBSCENE", "FLIRTATION", "SPAM", "UNSUBSTANTIAL", "HARASSMENT",
    "HATE_SPEECH", "VIOLENCE", "SELF_HARM"
]

def test_attributes():
    print("Testing Perspective API attributes...")
    print("\nAvailable attributes:")
    
    # Test each attribute
    for attr in TEST_ATTRIBUTES:
        data = {
            "comment": {"text": TEST_TEXT},
            "languages": ["en"],
            "requestedAttributes": {attr: {}}
        }
        
        try:
            response = requests.post(PERSPECTIVE_API_URL, json=data)
            if response.status_code == 200:
                print(f"✅ {attr} - Available")
            else:
                print(f"❌ {attr} - Not available (Status code: {response.status_code})")
        except Exception as e:
            print(f"❌ {attr} - Error: {str(e)}")

if __name__ == "__main__":
    test_attributes() 