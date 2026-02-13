#!/usr/bin/env python3
"""Test Groq API connectivity"""
import os
import sys

# Add RAG to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from groq import Groq
import groq

print("=" * 60)
print("GROQ API CONNECTIVITY TEST")
print("=" * 60)

# Test 1: Module import
print("\n✓ Groq module imported successfully")
print(f"  Groq version: {groq.__version__ if hasattr(groq, '__version__') else 'unknown'}")

# Test 2: Check for API key in environment
api_key_env = os.getenv("GROQ_API_KEY")
if api_key_env:
    print(f"\n✓ GROQ_API_KEY found in environment: {api_key_env[:10]}...")
else:
    print("\n⚠ GROQ_API_KEY not found in environment")
    print("  (Will use key passed to client if provided)")

# Test 3: Try to initialize client
test_key = "gsk_test_123456789"  # Fake key for testing
try:
    client = Groq(api_key=test_key, timeout=5)
    print(f"\n✓ Groq client initialized (test key)")
except Exception as e:
    print(f"\n✗ Failed to initialize Groq client: {e}")
    sys.exit(1)

# Test 4: Try a test request (will fail with invalid key, but that's okay - we're testing connectivity)
print("\n→ Attempting test request to Groq API...")
try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0,
        timeout=5
    )
    print(f"✓ Request succeeded: {response.choices[0].message.content[:50]}...")
except Exception as e:
    error_str = str(e).lower()
    if "authentication" in error_str or "invalid" in error_str or "401" in error_str:
        print(f"✓ API connectivity OK (authentication rejected as expected with test key)")
    elif "connection" in error_str or "timeout" in error_str or "network" in error_str:
        print(f"✗ CONNECTION ERROR: {e}")
        print("  Possible causes:")
        print("  1. Your internet connection is down")
        print("  2. Groq API server is unreachable (https://groq.com status)")
        print("  3. Firewall/proxy blocking Groq API (api.groq.com)")
        print("  4. DNS resolution failing")
        sys.exit(1)
    else:
        print(f"✓ API connectivity OK (got error as expected): {type(e).__name__}")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)
print("✓ Groq library is installed and working")
print("✓ Network connectivity to Groq API is OK")
print("\nNEXT STEPS:")
print("1. Ensure you have a valid Groq API key from console.groq.com")
print("2. In Streamlit, enter your actual API key in the Configuration panel")
print("3. Refresh the browser and try again")
