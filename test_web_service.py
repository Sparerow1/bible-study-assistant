#!/usr/bin/env python3
"""
Test script for BibleBot Web Service

This script tests the basic functionality of the web service.
"""

import requests
import time
import sys
from typing import Optional

def test_health(base_url: str) -> bool:
    """Test the health endpoint."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            if data.get('vector_count'):
                print(f"📊 Vector count: {data['vector_count']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_stats(base_url: str) -> bool:
    """Test the stats endpoint."""
    try:
        response = requests.get(f"{base_url}/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats retrieved: {data['total_vectors']} vectors, {data['dimension']} dimensions")
            return True
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return False

def test_chat(base_url: str, message: str) -> bool:
    """Test the chat endpoint."""
    try:
        print(f"🤖 Testing chat with: '{message}'")
        response = requests.post(
            f"{base_url}/chat",
            json={"message": message, "session_id": "test_session"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chat response received ({len(data['answer'])} characters)")
            if data.get('sources'):
                print(f"📚 Sources found: {len(data['sources'])}")
            return True
        else:
            print(f"❌ Chat failed: {response.status_code}")
            if response.text:
                print(f"Error details: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return False

def test_clear_memory(base_url: str) -> bool:
    """Test the clear memory endpoint."""
    try:
        response = requests.post(f"{base_url}/clear-memory", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Memory cleared: {data['message']}")
            return True
        else:
            print(f"❌ Clear memory failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Clear memory error: {e}")
        return False

def wait_for_service(base_url: str, max_attempts: int = 30) -> bool:
    """Wait for the service to become available."""
    print(f"⏳ Waiting for service at {base_url}...")
    
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Service is ready after {attempt + 1} attempts")
                return True
        except:
            pass
        
        print(f"   Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(2)
    
    print(f"❌ Service did not become available after {max_attempts} attempts")
    return False

def main():
    """Run all tests."""
    # Get base URL from command line or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print("🧪 BibleBot Web Service Test Suite")
    print("=" * 50)
    print(f"🌐 Testing service at: {base_url}")
    print()
    
    # Wait for service to be ready
    if not wait_for_service(base_url):
        print("❌ Service is not available. Make sure to start the web service first.")
        print("   Run: python start_web_service.py")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Health Check", lambda: test_health(base_url)),
        ("Statistics", lambda: test_stats(base_url)),
        ("Chat - Simple Question", lambda: test_chat(base_url, "What is love?")),
        ("Chat - Biblical Question", lambda: test_chat(base_url, "What does the Bible say about forgiveness?")),
        ("Clear Memory", lambda: test_clear_memory(base_url)),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        print("-" * 30)
        
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The web service is working correctly.")
        print(f"🌐 You can now access:")
        print(f"   - Web Interface: {base_url}/web")
        print(f"   - API Docs: {base_url}/docs")
        print(f"   - Health Check: {base_url}/health")
    else:
        print("⚠️  Some tests failed. Check the service configuration and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
