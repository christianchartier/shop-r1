#!/bin/bash
# Simplest possible GRPO test

echo "=== Minimal GRPO Test ==="
echo ""

cd /workspace/shop-r1
source .venv/bin/activate

# Ensure servers are running
echo "Verifying servers..."
curl -s http://localhost:8000/get_world_size/ >/dev/null 2>&1 || {
    echo "TRL server not running. Please start it first."
    exit 1
}
curl -s http://localhost:8001/v1/models >/dev/null 2>&1 || {
    echo "OpenAI server not running. Please start it first."
    exit 1
}

echo "✅ Both servers are running"
echo ""

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

echo "Running minimal GRPO training..."
echo "================================="

# Most minimal configuration - 1 generation, 1 batch, 1 step
CUDA_VISIBLE_DEVICES=1 python scripts/rl_train_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset data/rl.jsonl \
  --output_dir /tmp/grpo_minimal_$(date +%s) \
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
  --learning_rate 1e-7 2>&1 | tail -20

echo "================================="
echo ""
echo "If you see loss values above, GRPO is working!"