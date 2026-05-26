#!/usr/bin/env python3
"""
Direct test of the server to bypass frontend issues
"""

import requests
import os

def test_server_direct():
    """Test server endpoints directly"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing server endpoints directly...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test if enrollment endpoint exists
    try:
        # This should return 422 (validation error) not 404
        response = requests.post(f"{base_url}/enroll/testuser", data={})
        print(f"✅ Enrollment endpoint exists: {response.status_code}")
        if response.status_code == 422:
            print("✅ Endpoint is working (validation error expected)")
        elif response.status_code == 404:
            print("❌ Endpoint not found!")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Enrollment test failed: {e}")
    
    # Test verification endpoint  
    try:
        response = requests.post(f"{base_url}/verify/testuser", data={})
        print(f"✅ Verification endpoint exists: {response.status_code}")
        if response.status_code == 422:
            print("✅ Endpoint is working (validation error expected)")
        elif response.status_code == 404:
            print("❌ Endpoint not found!")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Verification test failed: {e}")

if __name__ == "__main__":
    test_server_direct()
