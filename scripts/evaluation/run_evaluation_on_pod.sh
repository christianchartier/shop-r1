#!/bin/bash
# Complete evaluation script for running on RunPod
# Run this after training models

set -e

echo "=========================================="
echo "   Shop-R1 Evaluation on RunPod"
echo "=========================================="
echo ""

# Navigate to project
cd /workspace/shop-r1
source .venv/bin/activate

# Step 1: Check prerequisites
echo "Checking prerequisites..."
if [ ! -d "checkpoints" ]; then
    echo "❌ No checkpoints directory found. Train models first!"
    echo "   Run: ./run_grpo_complete.sh"
    exit 1
fi

# Step 2: Create test dataset if needed
if [ ! -f "data/test.jsonl" ]; then
    echo "Creating test dataset (100 examples)..."
    python environments/shop_r1/synthesize.py -o data/test.jsonl -n 100 --seed 42
else
    # Check how many examples are in the dataset
    num_lines=$(wc -l < data/test.jsonl)
    echo "Found existing test dataset with $num_lines examples"
    if [ "$num_lines" -lt 10 ]; then
        echo "Dataset too small, regenerating with 100 examples..."
        python environments/shop_r1/synthesize.py -o data/test.jsonl -n 100 --seed 42
    fi
fi

# Step 3: Start evaluation server
echo "Starting vLLM evaluation server..."
tmux kill-session -t eval_server 2>/dev/null || true
tmux new -d -s eval_server "cd /workspace/shop-r1 && source .venv/bin/activate && \
  CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --dtype auto \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.70 \
  --disable-log-requests \
  --enforce-eager"

# Wait for server
echo "Waiting for server to start..."
sleep 20
for i in {1..30}; do
    if curl -s http://localhost:8001/v1/models >/dev/null 2>&1; then
        echo "✅ Evaluation server ready"
        break
    fi
    echo -n "."
    sleep 2
done

# Step 4: Set environment
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

# Step 5: Run evaluations
echo ""
echo "Running evaluations..."
echo "================================"

# Create results directory
mkdir -p results/evaluation

# 5a. Quick zero-shot baseline
# First check how many examples we have
num_examples=$(wc -l < data/test.jsonl)
eval_count=$((num_examples < 10 ? num_examples : 10))
echo "1. Zero-shot baseline ($eval_count examples from $num_examples total)..."
python scripts/eval_paper_metrics.py \
  --dataset data/test.jsonl \
  --model_alias local-qwen \
  --max_examples $eval_count \
  --output results/evaluation/zero_shot_quick.json

# 5b. Check if we have trained models
if [ -d "checkpoints/sft_shop_r1" ] || [ -d "checkpoints/rl_shop_r1" ]; then
    echo ""
    echo "2. Full evaluation with trained models..."
    ./scripts/run_paper_evaluation.sh data/test.jsonl results/evaluation/table2
else
    echo ""
    echo "⚠️  No trained models found. Skipping full evaluation."
    echo "   Train models first with:"
    echo "   - SFT: python scripts/sft_train.py ..."
    echo "   - GRPO: ./run_grpo_complete.sh"
fi

# Step 6: Display results
echo ""
echo "================================"
echo "Results Summary:"
echo "================================"

if [ -f "results/evaluation/zero_shot_quick.json" ]; then
    echo ""
    echo "Zero-shot Performance (baseline):"
    python -c "
import json
with open('results/evaluation/zero_shot_quick.json') as f:
    m = json.load(f)
    print(f\"  Exact Action: {m['exact_action_acc']:.2%}\")
    print(f\"  Action Type:  {m['action_type_acc']:.2%}\")  
    print(f\"  F1 Score:     {m['action_type_f1']:.2%}\")
"
fi

if [ -f "results/evaluation/table2/consolidated_table2.json" ]; then
    echo ""
    echo "Full Evaluation Results:"
    python -c "
import json
with open('results/evaluation/table2/consolidated_table2.json') as f:
    results = json.load(f)
    print('')
    print('Model                    Exact    Type     F1')
    print('----------------------------------------------')
    for name, metrics in results.items():
        print(f\"{name:<24} {metrics['exact_action_acc']:6.2%}  {metrics['action_type_acc']:6.2%}  {metrics['action_type_f1']:6.2%}\")
"
fi

# Step 7: Cleanup option
echo ""
echo "================================"
echo "Evaluation Complete!"
echo ""
echo "Server management:"
echo "  View logs:  tmux attach -t eval_server"
echo "  Stop:       tmux kill-session -t eval_server"
echo ""
echo "Results saved in: results/evaluation/"
echo "================================"