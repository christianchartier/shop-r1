#!/usr/bin/env python3
"""
Test script to verify GRPO dual-server routing fix.
This script tests that:
1. TRL communicator endpoints go to port 8000
2. Generation requests go to port 8001
"""

import asyncio
import os
import sys
from openai import AsyncOpenAI
import httpx

# Track all HTTP requests
logged_requests = []

def patch_httpx_logging():
    """Patch httpx to log all requests"""
    _orig_async = httpx.AsyncClient.request
    _orig_sync = httpx.Client.request
    
    async def async_request_logged(self, method, url, *args, **kwargs):
        logged_requests.append(f"[httpx-async] {method} {url}")
        print(f"[httpx-async] {method} {url}", flush=True)
        return await _orig_async(self, method, url, *args, **kwargs)
    
    def sync_request_logged(self, method, url, *args, **kwargs):
        logged_requests.append(f"[httpx-sync] {method} {url}")
        print(f"[httpx-sync] {method} {url}", flush=True)
        return _orig_sync(self, method, url, *args, **kwargs)
    
    httpx.AsyncClient.request = async_request_logged
    httpx.Client.request = sync_request_logged

async def test_routing():
    """Test that generation client routes to port 8001"""
    
    # Apply logging patch
    patch_httpx_logging()
    
    # Import after patching
    import verifiers.envs.environment as envmod
    
    # Create the generation client (should go to 8001)
    generation_client = AsyncOpenAI(
        api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
        base_url='http://localhost:8001/v1'
    )
    
    # Apply the routing fix
    _orig_get_model_response = envmod.Environment.get_model_response
    
    async def _route_to_generation_server(self, *args, **kwargs):
        args_list = list(args)
        if args_list:
            args_list[0] = generation_client
        kwargs['client'] = generation_client
        return await _orig_get_model_response(self, *args_list, **kwargs)
    
    envmod.Environment.get_model_response = _route_to_generation_server
    
    # Create a mock environment to test
    class MockEnv(envmod.Environment):
        def __init__(self):
            self.message_type = "chat"
    
    # Test making a request through the environment
    env = MockEnv()
    
    # Create a communicator client (simulating what GRPO does)
    communicator_client = AsyncOpenAI(
        api_key='EMPTY',
        base_url='http://localhost:8000/v1'  # TRL server
    )
    
    print("\n=== Testing Environment Routing ===")
    print("1. Original client passed to env: http://localhost:8000/v1 (TRL)")
    print("2. After routing fix, should use: http://localhost:8001/v1 (OpenAI)")
    print("\n=== Captured Requests ===")
    
    try:
        # Try to make a request (will fail if servers aren't running, but we'll see the URL)
        await env.get_model_response(
            client=communicator_client,  # Pass the wrong client
            model="Qwen/Qwen2.5-0.5B-Instruct",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.6
        )
    except Exception as e:
        print(f"\nRequest failed (expected if servers aren't running): {e}")
    
    print("\n=== Analysis ===")
    generation_requests = [r for r in logged_requests if 'chat/completions' in r]
    if generation_requests:
        for req in generation_requests:
            if ':8001' in req:
                print(f"✅ CORRECT: {req}")
            elif ':8000' in req:
                print(f"❌ WRONG: {req} (should be port 8001)")
            else:
                print(f"❓ UNKNOWN: {req}")
    else:
        print("No generation requests captured. Make sure to run with servers active.")
    
    return generation_requests

if __name__ == "__main__":
    print("Testing GRPO dual-server routing fix...")
    print("Make sure both servers are running:")
    print("  - TRL Communicator on port 8000")
    print("  - OpenAI API on port 8001")
    print("-" * 50)
    
    # Set environment
    os.environ['OPENAI_API_KEY'] = 'EMPTY'
    os.environ['OPENAI_BASE_URL'] = 'http://localhost:8001/v1'
    
    # Run test
    requests = asyncio.run(test_routing())
    
    # Check results
    if any(':8001' in r and 'chat/completions' in r for r in requests):
        print("\n✅ SUCCESS: Generation requests are correctly routed to port 8001")
        sys.exit(0)
    elif any(':8000' in r and 'chat/completions' in r for r in requests):
        print("\n❌ FAILURE: Generation requests are still going to port 8000")
        sys.exit(1)
    else:
        print("\n⚠️  No conclusive result - make sure servers are running")
        sys.exit(2)