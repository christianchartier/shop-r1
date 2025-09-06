#!/bin/bash
# Run improved zero-shot evaluation with proper action format instructions
# This addresses the issue where small models don't understand Shop-R1's action format

set -e

echo "==========================================="
echo "   Shop-R1 Zero-Shot Evaluation (Improved)"
echo "==========================================="
echo ""

# Default values
DATASET="${1:-data/test.jsonl}"
MAX_EXAMPLES="${2:-100}"
OUTPUT_DIR="results/evaluation"

# Ensure we're in the right directory
cd /workspace/shop-r1
source .venv/bin/activate

# Set environment variables
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8001/v1

# Check if server is running
if ! curl -s http://localhost:8001/v1/models >/dev/null 2>&1; then
    echo "Starting evaluation server..."
    tmux new -d -s eval_server "cd /workspace/shop-r1 && source .venv/bin/activate && \
      CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --host 0.0.0.0 --port 8001 \
      --dtype auto \
      --max-model-len 1024 \
      --gpu-memory-utilization 0.70 \
      --disable-log-requests \
      --enforce-eager"
    
    echo "Waiting for server to start..."
    sleep 20
    for i in {1..30}; do
        if curl -s http://localhost:8001/v1/models >/dev/null 2>&1; then
            echo "✅ Server ready"
            break
        fi
        echo -n "."
        sleep 2
    done
fi

# Create output directory
mkdir -p $OUTPUT_DIR

echo ""
echo "Running improved zero-shot evaluation..."
echo "Dataset: $DATASET"
echo "Examples: $MAX_EXAMPLES"
echo ""

# Run improved evaluation with explicit action format instructions
python scripts/evaluation/fix_zero_shot_prompting.py \
    --dataset $DATASET \
    --max_examples $MAX_EXAMPLES

echo ""
echo "==========================================="
echo "Comparison with Original Prompting:"
echo "==========================================="

# Run original evaluation for comparison
echo ""
echo "Running original evaluation (without explicit instructions)..."
python scripts/eval_paper_metrics.py \
    --dataset $DATASET \
    --model_alias local-qwen \
    --max_examples $MAX_EXAMPLES \
    --output $OUTPUT_DIR/zero_shot_original.json \
    2>/dev/null | grep -A 20 "PRIMARY METRICS" || true

echo ""
echo "==========================================="
echo "Results Summary:"
echo "==========================================="
echo ""
echo "Original prompting (no instructions):"
if [ -f "$OUTPUT_DIR/zero_shot_original.json" ]; then
    python -c "
import json
with open('$OUTPUT_DIR/zero_shot_original.json') as f:
    m = json.load(f)
    print(f'  Exact Action: {m[\"exact_action_acc\"]:.2%}')
    print(f'  Action Type:  {m[\"action_type_acc\"]:.2%}')
    print(f'  F1 Score:     {m[\"action_type_f1\"]:.2%}')
"
fi

echo ""
echo "Improved prompting (with explicit action format):"
if [ -f "$OUTPUT_DIR/zero_shot_improved.json" ]; then
    python -c "
import json
with open('$OUTPUT_DIR/zero_shot_improved.json') as f:
    m = json.load(f)
    print(f'  Exact Action: {m[\"exact_action_acc\"]:.2%}')
    print(f'  Action Type:  {m[\"action_type_acc\"]:.2%}')
    print(f'  F1 Score:     {m[\"action_type_f1\"]:.2%}')
"
fi

echo ""
echo "==========================================="
echo "Key Insight:"
echo "Small models need explicit instruction about"
echo "Shop-R1's action format (click, type_and_submit, terminate)"
echo "to achieve reasonable performance."
echo "==========================================="