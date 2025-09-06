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

## Step 4: Test GRPO (2 GPUs Required - Dual Server Architecture)

Paper‑faithful settings: α=0.005, β=0.001, DARS=1000, temperature≈0.6, similarity threshold=0.75, strict JSON encouraged.

**IMPORTANT**: GRPO requires TWO servers running simultaneously:
- TRL Communicator Server (port 8000, GPU 0): Handles `/get_world_size` and `/init_communicator` 
- OpenAI API Server (port 8001, GPU 1): Handles `/v1/chat/completions` for model generation

```bash
# 4.1 Install vLLM and wandb
python -m pip install "vllm==0.10.1.1" wandb

# 4.2 Create a small RL dataset (or bring your own)
python environments/shop_r1/synthesize.py -o data/rl.jsonl -n 200 --seed 7

# 4.3 Terminal A: Start TRL Communicator Server (GPU 0, port 8000)
tmux new -d -s vllm_trl "cd /workspace/shop-r1 && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.60"

# Verify TRL server is running
sleep 10
curl -s -i http://localhost:8000/get_world_size/

# 4.4 Terminal B: Start OpenAI API Server (GPU 1, port 8001) 
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

# Verify OpenAI server is running
sleep 20
curl -s http://localhost:8001/v1/models | python -m json.tool

# 4.5 Terminal C: Run GRPO Training (GPU 1)
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

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
  --num_generations 2 \
  --grad_accum 2 \
  --max_steps 5 \
  --save_steps 25 \
  --eval_steps 25 \
  --max_seq_len 1024 \
  --learning_rate 1e-7

# 4.6 Monitor server logs (optional, in separate terminals)
# tmux capture-pane -pt vllm_trl | tail -n 60  # Should show only communicator endpoints
# tmux capture-pane -pt vllm_oai | tail -n 60  # Should show /v1/chat/completions 200 OK

# 4.7 Clean up
tmux kill-session -t vllm_trl || true
tmux kill-session -t vllm_oai || true
```

Notes
- The training script now requests logprobs/top_logprobs and enforces JSON object responses for self‑certainty and format rewards.
- Use two GPUs (one for server, one for trainer) to avoid memory contention. On a single GPU, reduce `--num_generations`, `--max_model_len`, or run smaller models.
- If your dataset is larger, increase `--max_steps` and consider checkpointing frequency.
- GPU assignment tips:
  - `CUDA_VISIBLE_DEVICES=0` makes the process use physical GPU 0; `CUDA_VISIBLE_DEVICES=1` uses physical GPU 1.
  - Inside the process, the visible device index starts at 0 regardless of mapping. Verify mapping via `echo $CUDA_VISIBLE_DEVICES`.
  - Monitor usage with `watch -n 1 nvidia-smi` or `nvidia-smi -i 0,1 pmon -c 1` to see which PID is active on each GPU.

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
