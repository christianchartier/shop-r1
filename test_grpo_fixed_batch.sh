#!/bin/bash
# GRPO test with proper batch size configuration

echo "=== GRPO Test with Correct Batch Configuration ==="
echo ""

cd /workspace/shop-r1
source .venv/bin/activate

# Check servers
echo "Checking servers..."
if ! curl -s http://localhost:8000/get_world_size/ >/dev/null 2>&1; then
    echo "Starting TRL server..."
    tmux kill-session -t vllm_trl 2>/dev/null || true
    tmux new -d -s vllm_trl "cd /workspace/shop-r1 && source .venv/bin/activate && \
      CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --host 0.0.0.0 --port 8000 \
      --max-model-len 1024 \
      --gpu-memory-utilization 0.60"
    sleep 15
fi

if ! curl -s http://localhost:8001/v1/models >/dev/null 2>&1; then
    echo "Starting OpenAI server..."
    tmux kill-session -t vllm_oai 2>/dev/null || true
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

# Create dataset if needed
if [ ! -f data/rl.jsonl ]; then
    python environments/shop_r1/synthesize.py -o data/rl.jsonl -n 10 --seed 7
fi

export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

echo ""
echo "Running GRPO training with corrected batch settings..."
echo "======================================================"
echo ""

# Run with batch size 2 to match num_generations
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
  --per_device_batch_size 2 \
  --num_generations 2 \
  --grad_accum 1 \
  --max_steps 2 \
  --save_steps 25 \
  --eval_steps 25 \
  --max_seq_len 1024 \
  --learning_rate 1e-7 2>&1 | tee /tmp/grpo_batch_test.log

echo ""
echo "======================================================"
echo "Checking results..."
echo ""

# Check for successful training
if grep -q "{'loss':" /tmp/grpo_batch_test.log; then
    echo "✅ SUCCESS! GRPO training completed successfully!"
    echo ""
    echo "Training metrics:"
    grep -E "loss|rewards|kl" /tmp/grpo_batch_test.log | tail -5
elif grep -q "RuntimeError.*shape" /tmp/grpo_batch_test.log; then
    echo "❌ Batch size mismatch error"
    grep "RuntimeError" /tmp/grpo_batch_test.log | head -2
elif grep -q "404 Not Found" /tmp/grpo_batch_test.log; then
    echo "❌ Routing error (should be fixed)"
else
    echo "⚠️  Check /tmp/grpo_batch_test.log for details"
fi

echo ""
echo "Full log: /tmp/grpo_batch_test.log"