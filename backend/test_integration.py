"""
R3MES Backend Integration Test Script

Bu script backend'in temel fonksiyonlarını test eder.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_test(name, passed):
    """Test sonucunu yazdır."""
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{status} - {name}")

def test_health_check():
    """Health check endpoint'ini test et."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        passed = response.status_code == 200
        print_test("Health Check", passed)
        if passed:
            print(f"   Response: {response.json()}")
        return passed
    except Exception as e:
        print_test("Health Check", False)
        print(f"   Error: {e}")
        return False

def test_chain_status():
    """Chain status endpoint'ini test et."""
    try:
        response = requests.get(f"{BASE_URL}/chain/status")
        passed = response.status_code == 200
        print_test("Chain Status", passed)
        if passed:
            print(f"   Response: {response.json()}")
        return passed
    except Exception as e:
        print_test("Chain Status", False)
        print(f"   Error: {e}")
        return False

def test_login():
    """Login endpoint'ini test et."""
    try:
        payload = {
            "wallet_address": "remes1abcdefghijklmnopqrstuvwxyz1234567890",
            "signature": "test_signature_12345"
        }
        response = requests.post(f"{BASE_URL}/auth/login", json=payload)
        passed = response.status_code == 200
        print_test("Login", passed)
        
        if passed:
            data = response.json()
            print(f"   Access Token: {data['access_token'][:50]}...")
            print(f"   Refresh Token: {data['refresh_token'][:50]}...")
            return data['access_token'], data['refresh_token']
        return None, None
    except Exception as e:
        print_test("Login", False)
        print(f"   Error: {e}")
        return None, None

def test_generate_anonymous():
    """Anonymous generate endpoint'ini test et."""
    try:
        payload = {
            "prompt": "Merhaba R3MES!",
            "max_length": 100
        }
        response = requests.post(f"{BASE_URL}/generate", json=payload)
        passed = response.status_code == 200
        print_test("Generate (Anonymous)", passed)
        
        if passed:
            data = response.json()
            print(f"   Output: {data['output'][:100]}...")
            print(f"   Authenticated: {data['authenticated']}")
        return passed
    except Exception as e:
        print_test("Generate (Anonymous)", False)
        print(f"   Error: {e}")
        return False

def test_generate_authenticated(access_token):
    """Authenticated generate endpoint'ini test et."""
    try:
        payload = {
            "prompt": "Merhaba R3MES! (Authenticated)",
            "max_length": 100
        }
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        response = requests.post(f"{BASE_URL}/generate", json=payload, headers=headers)
        passed = response.status_code == 200
        print_test("Generate (Authenticated)", passed)
        
        if passed:
            data = response.json()
            print(f"   Output: {data['output'][:100]}...")
            print(f"   Authenticated: {data['authenticated']}")
        return passed
    except Exception as e:
        print_test("Generate (Authenticated)", False)
        print(f"   Error: {e}")
        return False

def test_chat(access_token):
    """Chat endpoint'ini test et."""
    try:
        payload = {
            "message": "Merhaba, nasılsın?"
        }
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        response = requests.post(f"{BASE_URL}/chat", json=payload, headers=headers)
        passed = response.status_code == 200
        print_test("Chat", passed)
        
        if passed:
            data = response.json()
            print(f"   Conversation ID: {data['conversation_id']}")
            print(f"   Message: {data['message'][:100]}...")
            print(f"   History Length: {data['history_length']}")
        return passed
    except Exception as e:
        print_test("Chat", False)
        print(f"   Error: {e}")
        return False

def test_profile(access_token):
    """Profile endpoint'ini test et."""
    try:
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        response = requests.get(f"{BASE_URL}/user/profile", headers=headers)
        passed = response.status_code == 200
        print_test("User Profile", passed)
        
        if passed:
            data = response.json()
            print(f"   Wallet: {data['wallet_address']}")
        return passed
    except Exception as e:
        print_test("User Profile", False)
        print(f"   Error: {e}")
        return False

def test_refresh_token(refresh_token):
    """Token refresh endpoint'ini test et."""
    try:
        payload = {
            "refresh_token": refresh_token
        }
        response = requests.post(f"{BASE_URL}/auth/refresh", json=payload)
        passed = response.status_code == 200
        print_test("Token Refresh", passed)
        
        if passed:
            data = response.json()
            print(f"   New Access Token: {data['access_token'][:50]}...")
        return passed
    except Exception as e:
        print_test("Token Refresh", False)
        print(f"   Error: {e}")
        return False

def test_input_sanitization():
    """Input sanitization'ı test et."""
    try:
        # XSS attempt
        payload = {
            "prompt": "<script>alert('XSS')</script>",
            "max_length": 100
        }
        response = requests.post(f"{BASE_URL}/generate", json=payload)
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            # XSS kodu sanitize edilmiş olmalı
            sanitized = "<script>" not in data['input']
            print_test("Input Sanitization (XSS)", sanitized)
            print(f"   Original: {payload['prompt']}")
            print(f"   Sanitized: {data['input']}")
            return sanitized
        return False
    except Exception as e:
        print_test("Input Sanitization", False)
        print(f"   Error: {e}")
        return False

def main():
    """Tüm testleri çalıştır."""
    print("=" * 60)
    print("R3MES Backend Integration Tests")
    print("=" * 60)
    print()
    
    print("📋 Public Endpoints")
    print("-" * 60)
    test_health_check()
    test_chain_status()
    print()
    
    print("🔐 Authentication")
    print("-" * 60)
    access_token, refresh_token = test_login()
    print()
    
    if access_token:
        print("🤖 AI Generation")
        print("-" * 60)
        test_generate_anonymous()
        test_generate_authenticated(access_token)
        print()
        
        print("💬 Chat")
        print("-" * 60)
        test_chat(access_token)
        print()
        
        print("👤 User Profile")
        print("-" * 60)
        test_profile(access_token)
        print()
        
        print("🔄 Token Refresh")
        print("-" * 60)
        test_refresh_token(refresh_token)
        print()
    
    print("🛡️ Security")
    print("-" * 60)
    test_input_sanitization()
    print()
    
    print("=" * 60)
    print("Tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    print("\n⚠️  Make sure the backend is running on http://localhost:8000")
    print("   Start with: python main.py\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Test suite failed: {e}")
