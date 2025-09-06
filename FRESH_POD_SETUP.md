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

## Step 4: GRPO Training (Requires 2 GPUs)

**Critical Architecture**: GRPO requires TWO vLLM servers running simultaneously:
1. **TRL Communicator** (port 8000, GPU 0): Handles distributed training coordination
2. **OpenAI API Server** (port 8001, GPU 1): Handles model generation requests

### Quick Option: Use the Complete Setup Script
```bash
# Download and run the all-in-one script
wget https://raw.githubusercontent.com/christianchartier/shop-r1/main/run_grpo_complete.sh
chmod +x run_grpo_complete.sh

# Quick test (1 step)
./run_grpo_complete.sh --quick

# Full training (50 steps)
./run_grpo_complete.sh
```

### Manual Setup Option

### 4.1 Install Dependencies
```bash
cd /workspace/shop-r1 && source .venv/bin/activate
python -m pip install "vllm==0.10.1.1" wandb
```

### 4.2 Create RL Dataset
```bash
# Generate a small dataset for testing (or use your own)
python environments/shop_r1/synthesize.py -o data/rl.jsonl -n 200 --seed 7
```

### 4.3 Start Both Servers

**Terminal A - TRL Communicator Server (GPU 0, port 8000):**
```bash
tmux new -d -s vllm_trl "cd /workspace/shop-r1 && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.60"

# Wait and verify TRL server is running
sleep 15
curl -s -i http://localhost:8000/get_world_size/
# Should return: {"world_size":1}
```

**Terminal B - OpenAI API Server (GPU 1, port 8001):**
```bash
tmux new -d -s vllm_oai "cd /workspace/shop-r1 && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --dtype auto \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.50 \
  --disable-log-requests \
  --enforce-eager \
  --max-num-batched-tokens 512"

# Wait and verify OpenAI server is running
sleep 25
curl -s http://localhost:8001/v1/models | python -m json.tool
# Should show model information
```

### 4.4 Run GRPO Training

**Quick Test (Verified Working):**
```bash
cd /workspace/shop-r1 && source .venv/bin/activate
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

# Minimal test - 1 step to verify setup
CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/rl.jsonl \
  --output_dir checkpoints/rl_test \
  --strict \
  --sim_threshold 0.75 \
  --alpha 0.005 \
  --beta 0.001 \
  --dars_factor 1000 \
  --temperature 0.6 \
  --per_device_batch_size 1 \
  --num_generations 1 \
  --grad_accum 1 \
  --max_steps 1 \
  --save_steps 100 \
  --eval_steps 100 \
  --max_seq_len 1024 \
  --learning_rate 1e-7
```

**Full Training Run:**
```bash
# After verifying the test works, run full training
CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/rl.jsonl \
  --output_dir checkpoints/rl_shop_r1 \
  --strict \
  --sim_threshold 0.75 \
  --alpha 0.005 \
  --beta 0.001 \
  --dars_factor 1000 \
  --temperature 0.6 \
  --per_device_batch_size 1 \
  --num_generations 1 \
  --grad_accum 8 \
  --max_steps 50 \
  --save_steps 25 \
  --eval_steps 25 \
  --max_seq_len 1024 \
  --learning_rate 1e-7
```

### 4.5 Monitor Training

```bash
# Check server logs (in separate terminals)
tmux capture-pane -pt vllm_trl | tail -n 30  # Should show only /get_world_size, /init_communicator
tmux capture-pane -pt vllm_oai | tail -n 30  # Should show /v1/chat/completions 200 OK

# Watch GPU usage
watch -n 1 nvidia-smi
```

### 4.6 Cleanup
```bash
# When done, stop servers
tmux kill-session -t vllm_trl
tmux kill-session -t vllm_oai
```

### Troubleshooting

**If you see 404 errors:** Generation requests are going to wrong server. Ensure you're using the latest code with routing fix.

**If you see max_tokens errors:** The model context is being exceeded. The fix limits max_tokens to 160.

**If you see batch size errors:** Use `num_generations=1` with `per_device_batch_size=1` as shown above.

**Verify servers are on correct ports:**
```bash
netstat -tuln | grep -E "8000|8001"
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

## Known Issues (Resolved)

1. ~~**GRPO Dataset Iterator**: GRPO training fails with "no single sample in epoch_iterator" error.~~ **FIXED**: Resolved by implementing dual-server architecture with proper routing.

2. **FlashAttention2**: The setup automatically patches scripts to use SDPA attention instead of FlashAttention2, which is not available in the RunPod environment.

3. **TRL/Transformers Conflicts**: The setup script pins compatible versions automatically. If you later change packages and hit conflicts, re-run the troubleshooting pin commands to restore transformers==4.56.1 and trl==0.21.0.

4. ~~**SFT Tensor Size Mismatch**:~~ **FIXED**: The setup now automatically patches the SFT training script to use DataCollatorForSeq2Seq instead of default_data_collator to handle sequences of different lengths.

5. ~~**GRPO 404 Routing Error**:~~ **FIXED**: Generation requests now correctly route to OpenAI server on port 8001 while communicator stays on port 8000.

6. ~~**GRPO max_tokens Error**:~~ **FIXED**: max_tokens is now limited to 160 to prevent exceeding model context window.

## Summary

✅ **Working Components:**
- Python 3.11 installation
- Shop-R1 repository setup
- Verified pins: transformers 4.56.1; trl 0.21.0
- SFT training pipeline
- **GRPO training with dual-server architecture** ✅
- vLLM server setup (both TRL communicator and OpenAI API)
- Evaluation framework

⚠️ **Minor Limitations:**
- FlashAttention2 (using SDPA workaround - works fine)

The setup now fully supports Shop-R1's complete training pipeline including both SFT and GRPO reinforcement learning. The dual-server architecture ensures proper routing of generation requests while maintaining TRL communication.
