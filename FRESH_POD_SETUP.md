# 🚀 FRESH POD SETUP - Complete Working Instructions

## Step 1: SSH into Fresh Pod
```bash
ssh -p 1234 root@[YOUR_POD_IP]

# ============================================
# ssh -p 1234 root@38.128.232.106
# ============================================
```

## Step 2: Run Complete Setup Script

```bash
cd /workspace
wget https://raw.githubusercontent.com/christianchartier/shop-r1/main/RUNPOD_QUICK_SETUP.sh
chmod +x RUNPOD_QUICK_SETUP.sh
./RUNPOD_QUICK_SETUP.sh
```

Note: The setup script automatically pins compatible Transformers/TRL versions and registers the environment.

## Step 3: Test SFT Training

```bash
# Quick test with minimal dataset
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
  --lr 2e-5

# Check results
ls -la checkpoints/test_sft/
```

## Step 4: Test GRPO (Optional - Requires 2 GPUs)

**Note**: GRPO has known dataset iteration issues in the current environment. Skip to Step 5 (Evaluation) for more critical validation.

```bash
# Install vLLM and wandb for GRPO
python -m pip install "vllm==0.10.1.1" wandb

# Create larger test dataset
cat > data/test_large.jsonl << 'EOF'
{"prompt": [{"role": "user", "content": "Search for laptop"}], "answer": {"type": "type_and_submit", "name": "search", "text": "laptop"}, "rationale": "Looking for a laptop"}
{"prompt": [{"role": "user", "content": "Click add to cart"}], "answer": {"type": "click", "name": "add_to_cart"}, "rationale": "Adding to cart"}
{"prompt": [{"role": "user", "content": "Done shopping"}], "answer": {"type": "terminate"}, "rationale": "Finished"}
{"prompt": [{"role": "user", "content": "Search for headphones"}], "answer": {"type": "type_and_submit", "name": "search", "text": "headphones"}, "rationale": "Looking for headphones"}
{"prompt": [{"role": "user", "content": "Click product link"}], "answer": {"type": "click", "name": "product_link"}, "rationale": "Checking product"}
{"prompt": [{"role": "user", "content": "Add to wishlist"}], "answer": {"type": "click", "name": "wishlist"}, "rationale": "Saving for later"}
{"prompt": [{"role": "user", "content": "Search for books"}], "answer": {"type": "type_and_submit", "name": "search", "text": "books"}, "rationale": "Looking for books"}
{"prompt": [{"role": "user", "content": "End session"}], "answer": {"type": "terminate"}, "rationale": "Finished shopping"}
EOF

# Terminal 1: Start vLLM server (in tmux)
tmux new -s vllm
# Inside tmux:
cd /workspace/shop-r1 && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.20
# Press Ctrl+B, then D to detach

# Terminal 2: Run GRPO (main terminal)
sleep 15
CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/test_large.jsonl \
  --output_dir checkpoints/test_rl \
  --max_steps 2 \
  --save_steps 10 \
  --eval_steps 10 \
  --per_device_batch_size 1 \
  --num_generations 1 \
  --grad_accum 1 \
  --max_seq_len 1024 \
  --learning_rate 1e-7

# Clean up
tmux kill-session -t vllm
```

## Step 5: Run Evaluation (Most Important Test)

### Terminal 1: Start Evaluation vLLM Server
```bash
# Check GPU memory first
nvidia-smi

# Start tmux session
tmux new -s eval

# Inside tmux:
cd /workspace/shop-r1 && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --dtype auto \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.70 \
  --disable-log-requests \
  --enforce-eager

# Press Ctrl+B, then D to detach
```

### Terminal 2: Run Evaluation
```bash
# Wait for server to load
sleep 30

# Set environment
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

# Test server connection
curl -s http://localhost:8001/v1/models | jq .

# Run evaluation (IMPORTANT: must specify -b and -k flags)
# Add logprobs/top_logprobs to enable self‑certainty; json_object improves formatting
vf-eval shop-r1 \
  -m Qwen/Qwen2.5-0.5B-Instruct \
  -b http://localhost:8001/v1 -k EMPTY \
  -S '{"logprobs":true,"top_logprobs":5,"temperature":0,"response_format":{"type":"json_object"}}' \
  -n 5 -r 1

# Clean up
tmux kill-session -t eval
```

## Troubleshooting

### If Python 3.11 installation hangs:
```bash
# Kill the hanging process and use the alternative method:
pkill -f apt
DEBIAN_FRONTEND=noninteractive apt-get install -y python3.11 python3.11-venv python3.11-dev
```

### If TRL installation conflicts with transformers:
```bash
# Use validated pins for compatibility
pip uninstall -y transformers trl || true
pip install --no-cache-dir "transformers==4.56.1" "trl==0.21.0"
vf-install shop-r1
```

### If vLLM server fails to start:
```bash
# Use reduced memory settings
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --dtype auto \
  --max-model-len 512 \
  --gpu-memory-utilization 0.50 \
  --disable-log-requests \
  --enforce-eager \
  --max-num-batched-tokens 512
```

## Known Issues

1. **GRPO Dataset Iterator**: GRPO training fails with "no single sample in epoch_iterator" error. This is a compatibility issue between verifiers GRPO trainer and the current environment. SFT training works correctly.

2. **FlashAttention2**: The setup automatically patches scripts to use SDPA attention instead of FlashAttention2, which is not available in the RunPod environment.

3. **TRL/Transformers Conflicts**: The setup script pins compatible versions automatically. If you later change packages and hit conflicts, re-run the troubleshooting pin commands to restore transformers==4.56.1 and trl==0.21.0.

4. **SFT Tensor Size Mismatch**: Fixed - The setup now automatically patches the SFT training script to use DataCollatorForSeq2Seq instead of default_data_collator to handle sequences of different lengths.

## Summary

✅ **Working Components:**
- Python 3.11 installation
- Shop-R1 repository setup
- Verified pins: transformers 4.56.1; trl 0.21.0
- SFT training pipeline
- vLLM server setup
- Evaluation framework

⚠️ **Known Limitations:**
- GRPO training (dataset iteration issue)
- FlashAttention2 (using SDPA workaround)

The setup is sufficient for testing Shop-R1's core functionality, with SFT training and evaluation being the most critical components for validating the implementation.
