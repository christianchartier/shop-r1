#!/bin/bash
# Full remote testing script for Prime Intellect pod

ssh -p 1234 root@62.169.159.61 << 'ENDSSH'
set -e

echo "=== Shop-R1 Testing on Prime Intellect Pod ==="
echo "Starting at: $(date)"

# 1. Clean setup
echo -e "\n1. Setting up environment..."
cd /ephemeral || cd /workspace || cd ~
rm -rf shop-r1-test
git clone https://github.com/christianchartier/shop-r1.git shop-r1-test
cd shop-r1-test

# 2. Create Python 3.11 environment
echo -e "\n2. Creating Python 3.11 virtual environment..."
python3.11 -m venv .venv311 2>/dev/null || python3.10 -m venv .venv311 2>/dev/null || python3.9 -m venv .venv311
source .venv311/bin/activate
python --version

# 3. Install dependencies
echo -e "\n3. Installing dependencies..."
python -m pip install -U pip setuptools wheel
python -m pip uninstall -y torchvision 2>/dev/null || true
python -m pip install "transformers>=4.55,<5" "trl==0.21.0" "vllm==0.10.1.1"
python -m pip install accelerate>=0.30 peft>=0.11 datasets>=2.19 requests openai
python -m pip install 'verifiers @ git+https://github.com/willccbb/verifiers@main'

# 4. Install shop-r1 environment
echo -e "\n4. Installing shop-r1 environment..."
python -m pip install -e .
vf-install shop-r1

# 5. Test environment loading
echo -e "\n5. Testing environment loading..."
python -c "
import verifiers as vf
env = vf.load_environment('shop-r1')
print(f'✓ Environment loaded successfully')
print(f'  Parser: {env.parser.__class__.__name__}')
print(f'  Rubric functions: {len(env.rubric.funcs)}')
print(f'  Weights: {env.rubric.weights}')
"

# 6. Create test dataset
echo -e "\n6. Creating test dataset..."
mkdir -p data
python -c "
import json

examples = [
    {
        'prompt': [{'role': 'user', 'content': 'You are on Amazon homepage. Search for a laptop.'}],
        'answer': {'type': 'type_and_submit', 'name': 'search_input', 'text': 'gaming laptop'},
        'rationale': 'I need to search for a gaming laptop'
    },
    {
        'prompt': [{'role': 'user', 'content': 'Search results show laptops. Click on the first one.'}],
        'answer': {'type': 'click', 'name': 'product_link_1'},
        'rationale': 'Clicking on the first laptop result'
    },
    {
        'prompt': [{'role': 'user', 'content': 'Product page loaded. Add to cart.'}],
        'answer': {'type': 'click', 'name': 'add_to_cart'},
        'rationale': 'Adding the laptop to my cart'
    },
    {
        'prompt': [{'role': 'user', 'content': 'Item added. Continue shopping or checkout?'}],
        'answer': {'type': 'terminate'},
        'rationale': 'Done shopping for now'
    }
]

with open('data/test.jsonl', 'w') as f:
    for ex in examples:
        f.write(json.dumps(ex) + '\\n')

print(f'✓ Created test dataset with {len(examples)} examples')
"

# 7. Check GPU availability
echo -e "\n7. Checking GPU availability..."
nvidia-smi -L || echo "No GPUs found"
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
echo "Found $NUM_GPUS GPUs"

# 8. Test SFT training (quick test)
echo -e "\n8. Testing SFT training..."
python scripts/sft_train.py \
  --dataset data/test.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir checkpoints/test_sft \
  --epochs 1 \
  --save_steps 2 \
  --logging_steps 1 \
  --per_device_batch_size 1 \
  --grad_accum 1 \
  --max_seq_len 1024 \
  --lr 2e-5 || echo "SFT training failed"

# 9. Test environment evaluation
echo -e "\n9. Testing reward computation..."
python -c "
import json
import verifiers as vf

# Load environment
env = vf.load_environment('shop-r1', debug_rewards=True)

# Test example
with open('data/test.jsonl', 'r') as f:
    example = json.loads(f.readline())

prompt = example['prompt']
answer = example['answer']
completion = json.dumps({
    'rationale': example.get('rationale', 'test'),
    'action': answer
})

# Compute rewards
rewards = []
for func, weight in zip(env.rubric.funcs, env.rubric.weights):
    try:
        r = func(completion, answer, prompt=prompt, info=answer)
        rewards.append((r, weight, r * weight))
    except Exception as e:
        rewards.append((0, weight, 0))
        print(f'  Error in reward func: {e}')

total = sum(r[2] for r in rewards)
print(f'✓ Total reward: {total:.3f}')
print('  Components:')
for i, (raw, weight, weighted) in enumerate(rewards):
    print(f'    Func {i}: raw={raw:.3f}, weight={weight:.3f}, weighted={weighted:.3f}')
"

# 10. If multi-GPU, test GRPO
if [ "$NUM_GPUS" -ge 2 ]; then
    echo -e "\n10. Testing multi-GPU GRPO setup..."
    
    # Kill any existing vLLM servers
    pkill -f "trl.scripts.vllm_serve" 2>/dev/null || true
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 2
    
    # Start vLLM server on GPU 0
    echo "Starting vLLM server on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --host 0.0.0.0 --port 8000 \
        --max-model-len 1024 \
        --gpu-memory-utilization 0.20 &
    VLLM_PID=$!
    
    echo "Waiting for vLLM server to start..."
    sleep 15
    
    # Test endpoints
    echo "Testing vLLM endpoints..."
    curl -L -s -o /dev/null -w "World Size: %{http_code}\n" http://localhost:8000/get_world_size/ || true
    curl -L -s -o /dev/null -w "Init Comm: %{http_code}\n" http://localhost:8000/init_communicator/ || true
    
    # Run GRPO on GPU 1 (very short test)
    echo "Running GRPO test on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --dataset data/test.jsonl \
        --output_dir checkpoints/test_rl \
        --max_steps 2 \
        --save_steps 10 \
        --eval_steps 10 \
        --per_device_batch_size 1 \
        --num_generations 1 \
        --grad_accum 1 \
        --max_seq_len 1024 \
        --learning_rate 1e-7 \
        --temperature 0.6 \
        --alpha 0.13 \
        --beta 0.001 \
        --dars_factor 500 || echo "GRPO training failed"
    
    # Kill vLLM server
    kill $VLLM_PID 2>/dev/null || true
else
    echo -e "\n10. Skipping multi-GPU GRPO test (need 2+ GPUs)"
fi

# 11. Run verifiers evaluation
echo -e "\n11. Running verifiers evaluation..."
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

# Start a simple vLLM server for evaluation
if [ "$NUM_GPUS" -ge 1 ]; then
    echo "Starting vLLM server for evaluation..."
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --host 0.0.0.0 --port 8001 \
        --dtype auto \
        --max-model-len 2048 \
        --gpu-memory-utilization 0.90 &
    EVAL_PID=$!
    
    sleep 15
    
    # Run evaluation
    vf-eval shop-r1 -m Qwen/Qwen2.5-0.5B-Instruct -s -n 4 || echo "Evaluation failed"
    
    # Kill evaluation server
    kill $EVAL_PID 2>/dev/null || true
else
    echo "Skipping GPU evaluation (no GPUs found)"
fi

echo -e "\n=== Testing Complete ==="
echo "Finished at: $(date)"

# Show summary
echo -e "\nSummary:"
ls -la checkpoints/ 2>/dev/null || echo "No checkpoints created"
echo -e "\nTest dataset:"
wc -l data/test.jsonl

ENDSSH