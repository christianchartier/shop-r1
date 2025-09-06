#!/bin/bash
# Quick test script for GRPO with proper max_tokens setting

echo "=== Quick GRPO Test with Fixed max_tokens ==="
echo ""

cd /workspace/shop-r1
source .venv/bin/activate

# Update to latest code
echo "Pulling latest changes..."
git fetch origin main
git reset --hard origin/main

# Check if servers are running
echo "Checking servers..."
if curl -s http://localhost:8000/get_world_size/ >/dev/null 2>&1; then
    echo "✅ TRL server running on port 8000"
else
    echo "❌ TRL server not running. Starting it..."
    tmux new -d -s vllm_trl "cd /workspace/shop-r1 && source .venv/bin/activate && \
      CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --host 0.0.0.0 --port 8000 \
      --max-model-len 1024 \
      --gpu-memory-utilization 0.60"
    sleep 15
fi

if curl -s http://localhost:8001/v1/models >/dev/null 2>&1; then
    echo "✅ OpenAI server running on port 8001"
else
    echo "❌ OpenAI server not running. Starting it..."
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
    sleep 20
fi

# Create minimal dataset if needed
if [ ! -f data/rl.jsonl ]; then
    echo "Creating test dataset..."
    python environments/shop_r1/synthesize.py -o data/rl.jsonl -n 10 --seed 7
fi

# Set environment
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

echo ""
echo "Running GRPO training (2 steps)..."
echo "================================"

# Run GRPO with minimal settings
CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/rl.jsonl \
  --output_dir /tmp/grpo_test_$(date +%Y%m%d_%H%M%S) \
  --strict \
  --sim_threshold 0.75 \
  --alpha 0.005 \
  --beta 0.001 \
  --dars_factor 1000 \
  --temperature 0.6 \
  --per_device_batch_size 1 \
  --num_generations 2 \
  --grad_accum 1 \
  --max_steps 2 \
  --save_steps 25 \
  --eval_steps 25 \
  --max_seq_len 1024 \
  --learning_rate 1e-7 2>&1 | tee /tmp/grpo_run.log

echo ""
echo "================================"
echo "Checking results..."
echo ""

# Check for errors
if grep -q "BadRequestError.*max_tokens" /tmp/grpo_run.log; then
    echo "❌ Still getting max_tokens error"
    grep "max_tokens" /tmp/grpo_run.log | head -2
elif grep -q "404 Not Found" /tmp/grpo_run.log; then
    echo "❌ Still getting 404 routing errors"
elif grep -q "steps/iteration" /tmp/grpo_run.log; then
    echo "✅ SUCCESS! GRPO training is running correctly!"
    echo ""
    echo "Training metrics:"
    grep -E "(loss|rewards|kl)" /tmp/grpo_run.log | tail -5
else
    echo "⚠️  Unclear result. Check /tmp/grpo_run.log for details"
fi

echo ""
echo "Log saved to: /tmp/grpo_run.log"
echo ""
echo "To monitor servers:"
echo "  tmux attach -t vllm_trl  (Ctrl+B,D to detach)"
echo "  tmux attach -t vllm_oai  (Ctrl+B,D to detach)"