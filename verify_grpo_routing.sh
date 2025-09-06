#!/bin/bash
# Quick verification script for GRPO dual-server routing fix
# Run this on the pod after starting both servers

echo "=== GRPO Routing Fix Verification ==="
echo "This will test if generation requests go to the correct server (port 8001)"
echo ""

# Test inline with httpx logging
cat << 'EOF' > /tmp/test_routing.py
import os, sys, runpy
import httpx

# Log all HTTP requests
print("Setting up HTTP request logging...")
_orig_async = httpx.AsyncClient.request
_orig_sync = httpx.Client.request

async def async_log(self, method, url, *args, **kwargs):
    print(f"[httpx-async] {method} {url}", flush=True)
    return await _orig_async(self, method, url, *args, **kwargs)

def sync_log(self, method, url, *args, **kwargs):
    print(f"[httpx-sync] {method} {url}", flush=True)
    return _orig_sync(self, method, url, *args, **kwargs)

httpx.AsyncClient.request = async_log
httpx.Client.request = sync_log

# Apply the routing fix from rl_train_grpo.py
from openai import AsyncOpenAI
import verifiers.envs.environment as envmod

generation_client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'),
    base_url='http://localhost:8001/v1'
)

_orig = envmod.Environment.get_model_response

async def _route(self, *args, **kwargs):
    args_list = list(args)
    if args_list:
        args_list[0] = generation_client
    kwargs['client'] = generation_client
    return await _orig(self, *args_list, **kwargs)

envmod.Environment.get_model_response = _route

print("\nRouting fix applied. Running micro training test...")
print("Watch for HTTP requests below:")
print("-" * 50)

# Run tiny GRPO training
sys.argv = [
    "scripts/rl_train_grpo.py",
    "--model", "Qwen/Qwen2.5-0.5B-Instruct",
    "--dataset", "data/rl.jsonl",
    "--output_dir", "/tmp/test_grpo",
    "--strict",
    "--sim_threshold", "0.75",
    "--alpha", "0.005",
    "--beta", "0.001",
    "--dars_factor", "1000",
    "--temperature", "0.6",
    "--per_device_batch_size", "1",
    "--num_generations", "1",
    "--grad_accum", "1",
    "--max_steps", "1",
    "--save_steps", "25",
    "--eval_steps", "25",
    "--max_seq_len", "1024",
    "--learning_rate", "1e-7",
]

try:
    runpy.run_path("scripts/rl_train_grpo.py", run_name="__main__")
except Exception as e:
    print(f"\nTraining error (check if it's related to routing): {e}")
EOF

echo "Prerequisites:"
echo "1. TRL Communicator server running on port 8000"
echo "2. OpenAI API server running on port 8001"
echo "3. data/rl.jsonl exists"
echo ""
echo "Running test..."
echo "============================================"

cd /workspace/shop-r1 && source .venv/bin/activate

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

CUDA_VISIBLE_DEVICES=1 python /tmp/test_routing.py 2>&1 | tee /tmp/routing_test.log

echo ""
echo "============================================"
echo "Analyzing results..."
echo ""

# Check for correct routing
if grep -q "http://localhost:8001/v1/chat/completions" /tmp/routing_test.log; then
    echo "✅ SUCCESS: Generation requests going to port 8001 (OpenAI server)"
else
    if grep -q "http://localhost:8000/v1/chat/completions" /tmp/routing_test.log; then
        echo "❌ FAILURE: Generation requests still going to port 8000 (TRL server)"
        echo "   The fix is not working correctly."
    else
        echo "⚠️  WARNING: No chat/completions requests detected."
        echo "   Check if both servers are running and dataset exists."
    fi
fi

echo ""
echo "Server endpoints hit:"
grep "\[httpx-" /tmp/routing_test.log | sort | uniq | head -20

echo ""
echo "Full log saved to: /tmp/routing_test.log"