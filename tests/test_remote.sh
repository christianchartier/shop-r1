#!/bin/bash
# Remote testing script for Prime Intellect pod

echo "=== Remote Testing Commands for Prime Intellect Pod ==="
echo ""
echo "SSH into the pod:"
echo "ssh root@62.169.159.61 -p 1234"
echo ""
echo "Once connected, run these commands:"
echo ""
cat << 'REMOTE_COMMANDS'
# 1. Setup environment
cd /ephemeral
rm -rf shop-r1
git clone https://github.com/christianchartier/shop-r1.git
cd shop-r1

# 2. Create Python 3.11 venv (Prime Intellect pods should have it)
python3.11 -m venv .venv311
source .venv311/bin/activate

# 3. Install dependencies
python -m pip install -U pip
python -m pip uninstall -y torchvision || true
python -m pip install "transformers>=4.55,<5" "trl==0.21.0" "vllm==0.10.1.1"
python -m pip install accelerate>=0.30 peft>=0.11 datasets>=2.19 requests
python -m pip install 'verifiers @ git+https://github.com/willccbb/verifiers@main'

# 4. Install the shop-r1 environment
python -m pip install -e .
vf-install shop-r1

# 5. Test environment loading
python -c "
import verifiers as vf
env = vf.load_environment('shop-r1')
print(f'✓ Environment loaded: {env}')
"

# 6. Generate small test dataset
mkdir -p data
python -c "
import json
examples = [
    {
        'prompt': [{'role': 'user', 'content': 'Search for laptop'}],
        'answer': {'type': 'type_and_submit', 'name': 'search', 'text': 'laptop'},
        'rationale': 'Looking for a laptop'
    },
    {
        'prompt': [{'role': 'user', 'content': 'Click add to cart'}],
        'answer': {'type': 'click', 'name': 'add_to_cart'},
        'rationale': 'Adding item to cart'
    }
]
with open('data/test.jsonl', 'w') as f:
    for ex in examples:
        f.write(json.dumps(ex) + '\n')
print(f'✓ Created test dataset with {len(examples)} examples')
"

# 7. Test SFT script (dry run)
python scripts/sft_train.py \
  --dataset data/test.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output_dir checkpoints/test_sft \
  --epochs 1 \
  --save_steps 1 \
  --logging_steps 1 \
  --per_device_batch_size 1 \
  --grad_accum 1 \
  --max_seq_len 2048

# 8. Check GPU availability
nvidia-smi -L

# 9. If 2 GPUs available, test GRPO setup
if [ $(nvidia-smi -L | wc -l) -ge 2 ]; then
    echo "Testing multi-GPU GRPO setup..."
    
    # Terminal 1 (tmux): Start vLLM server on GPU 0
    tmux new -s vllm -d
    tmux send-keys -t vllm "CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve --model Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000 --max-model-len 2048 --gpu-memory-utilization 0.20" C-m
    
    sleep 10
    
    # Test endpoints
    curl -L -s -o /dev/null -w "WS %{http_code}\n" http://localhost:8000/get_world_size/
    curl -L -s -o /dev/null -w "IC %{http_code}\n" http://localhost:8000/init_communicator/
    
    # Terminal 2: Test GRPO on GPU 1
    CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --dataset data/test.jsonl \
      --output_dir checkpoints/test_rl \
      --max_steps 2 \
      --save_steps 1 \
      --eval_steps 1 \
      --per_device_batch_size 1 \
      --num_generations 1 \
      --grad_accum 1 \
      --max_seq_len 2048
fi

REMOTE_COMMANDS

echo ""
echo "=== End of Remote Commands ==="